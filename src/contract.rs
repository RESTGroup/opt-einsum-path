//! Contains the primary optimization and contraction routines.

use crate::{paths::PathOptimizer, *};

/* #region OptimizeKind */

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum OptimizeKind {
    Optimized,
    BranchBound(paths::BranchBound),
}

impl paths::PathOptimizer for OptimizeKind {
    fn optimize_path(
        &mut self,
        inputs: &[&BTreeSet<char>],
        output: &BTreeSet<char>,
        size_dict: &BTreeMap<char, usize>,
        memory_limit: Option<usize>,
    ) -> PathType {
        match self {
            OptimizeKind::Optimized => paths::optimal(inputs, output, size_dict, memory_limit),
            OptimizeKind::BranchBound(optimizer) => optimizer.optimize_path(inputs, output, size_dict, memory_limit),
        }
    }
}

impl From<&str> for OptimizeKind {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "optimized" => OptimizeKind::Optimized,
            "branchbound" | "branch_bound" | "branch-bound" | "branch bound" => {
                OptimizeKind::BranchBound(paths::BranchBound::default())
            },
            _ => panic!("Unknown optimization kind: {s}"),
        }
    }
}

/* #endregion */

/* #region PathInfo */

#[derive(Debug, Clone)]
pub struct PathInfo {
    pub contraction_list: Vec<ContractionType>,
    pub input_subscripts: String,
    pub output_subscript: String,
    pub indices: ArrayIndexType,
    pub path: PathType,
    pub scale_list: Vec<usize>,
    pub naive_cost: f64,
    pub opt_cost: f64,
    pub speedup: f64,
    pub size_list: Vec<usize>,
    pub size_dict: SizeDictType,
    pub shapes: Vec<Vec<usize>>,
    pub equation: String,
    pub largest_intermediate: f64,
}

impl PathInfo {
    pub fn new(
        contraction_list: Vec<ContractionType>,
        input_subscripts: String,
        output_subscript: String,
        indices: ArrayIndexType,
        path: PathType,
        scale_list: Vec<usize>,
        naive_cost: f64,
        opt_cost: f64,
        size_list: Vec<usize>,
        size_dict: SizeDictType,
    ) -> Self {
        let speedup = naive_cost / opt_cost.max(1.0);
        let shapes = input_subscripts.split(',').map(|ks| ks.chars().map(|k| size_dict[&k]).collect()).collect();
        let equation = format!("{input_subscripts}->{output_subscript}");
        let largest_intermediate = *size_list.iter().max().unwrap_or(&1) as f64;

        Self {
            contraction_list,
            input_subscripts,
            output_subscript,
            indices,
            path,
            scale_list,
            naive_cost,
            opt_cost,
            speedup,
            size_list,
            size_dict,
            shapes,
            equation,
            largest_intermediate,
        }
    }
}

impl std::fmt::Display for PathInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Header section
        writeln!(f, "  Complete contraction:  {}", self.equation)?;
        writeln!(f, "         Naive scaling:  {}", self.indices.len())?;
        writeln!(f, "     Optimized scaling:  {}", self.scale_list.iter().max().unwrap_or(&0))?;
        writeln!(f, "      Naive FLOP count:  {:.3e}", self.naive_cost)?;
        writeln!(f, "  Optimized FLOP count:  {:.3e}", self.opt_cost)?;
        writeln!(f, "   Theoretical speedup:  {:.3e}", self.speedup)?;
        writeln!(f, "  Largest intermediate:  {:.3e} elements", self.largest_intermediate)?;
        writeln!(f, "{}", "-".repeat(80))?;
        writeln!(f, "{:>6} {:>11} {:>22} {:>37}", "scaling", "BLAS", "current", "remaining")?;
        writeln!(f, "{}", "-".repeat(80))?;

        // Contraction steps
        for (n, ContractionType { einsum_str, remaining, do_blas, .. }) in self.contraction_list.iter().enumerate() {
            let remaining_str = match remaining {
                Some(remaining) => format!("{}->{}", remaining.join(","), self.output_subscript),
                None => "...".to_string(),
            };

            let size_remaining = 56usize.saturating_sub(22.max(einsum_str.len()));
            let scale = self.scale_list.get(n).unwrap_or(&0);
            let blas_str = if *do_blas { "BLAS" } else { "" };

            writeln!(f, "\n{scale:>4} {blas_str:>14} {einsum_str:>22}    {remaining_str:>size_remaining$}")?;
        }

        Ok(())
    }
}

/* #endregion */

pub fn contract_path<Opt>(
    subscripts: &str,
    operands: &[TensorShapeType],
    use_blas: bool,
    mut optimize: Opt,
    memory_limit: Option<usize>,
) -> Result<(PathType, PathInfo), String>
where
    Opt: PathOptimizer,
{
    // Parse einsum input
    let (input_subscripts, output_subscript, input_shapes) = parser::parse_einsum_input(subscripts, operands)?;

    // Build useful collections
    let input_list: Vec<&str> = input_subscripts.split(',').collect_vec();
    let input_sets: Vec<ArrayIndexType> = input_list.iter().map(|s| s.chars().collect()).collect();
    let input_sets_ref: Vec<&ArrayIndexType> = input_sets.iter().collect();
    let output_set: ArrayIndexType = output_subscript.chars().collect();
    let indices: ArrayIndexType = input_subscripts.chars().filter(|&c| c != ',').collect();

    // Build size dictionary
    let mut size_dict = SizeDictType::new();
    for (tnum, term) in input_list.iter().enumerate() {
        let sh = &input_shapes[tnum];

        if sh.len() != term.len() {
            return Err(format!(
                "Einstein sum subscript '{}' does not contain the correct number of indices for operand {tnum}.",
                input_list[tnum]
            ));
        }

        for (cnum, char) in term.chars().enumerate() {
            let dim = sh[cnum];

            if let Some(&existing_dim) = size_dict.get(&char) {
                // For broadcasting cases we always want the largest dim size
                if existing_dim == 1 {
                    size_dict.insert(char, dim);
                } else if dim != 1 && dim != existing_dim {
                    return Err(format!(
                        "Size of label '{char}' for operand {tnum} ({existing_dim}) does not match previous terms ({dim})."
                    ));
                }
            } else {
                size_dict.insert(char, dim);
            }
        }
    }

    // Compute size of each input array plus the output array
    let _size_list: Vec<f64> = input_list
        .iter()
        .chain(std::iter::once(&output_subscript.as_str()))
        .map(|&term| helpers::compute_size_by_dict(term.chars().collect_vec().iter(), &size_dict))
        .collect();

    let memory_arg = match memory_limit {
        Some(limit) if limit == usize::MAX => None,
        Some(limit) => Some(limit),
        None => None,
    };

    let num_ops = input_list.len();

    // Compute naive cost
    let inner_product = (input_sets.iter().map(|s| s.len()).sum::<usize>() - indices.len()) > 0;
    let naive_cost = helpers::flop_count(indices.iter(), inner_product, num_ops, &size_dict);

    // Compute the path
    let path_tuple = if num_ops <= 2 {
        vec![(0..num_ops).collect()]
    } else {
        optimize.optimize_path(&input_sets_ref, &output_set, &size_dict, memory_arg)
    };

    let mut cost_list = Vec::new();
    let mut scale_list = Vec::new();
    let mut size_intermediates = Vec::new();
    let mut contraction_list = Vec::new();

    let mut current_input_list = input_list.iter().map(|s| s.to_string()).collect_vec();
    let mut current_input_sets = input_sets.clone();
    let mut current_input_shapes = input_shapes.clone();

    for (cnum, contract_inds) in path_tuple.iter().enumerate() {
        // Make sure we remove inds from right to left
        let mut sorted_contract_inds = contract_inds.clone();
        sorted_contract_inds.sort_by(|a, b| b.cmp(a));

        let (out_inds, new_input_sets, idx_removed, idx_contract) = helpers::find_contraction(
            &sorted_contract_inds,
            &current_input_sets.iter().collect::<Vec<_>>(),
            &output_set,
        );

        // Compute cost, scale, and size
        let cost = helpers::flop_count(idx_contract.iter(), !idx_removed.is_empty(), contract_inds.len(), &size_dict);
        cost_list.push(cost);
        scale_list.push(idx_contract.len());
        size_intermediates.push(helpers::compute_size_by_dict(out_inds.iter(), &size_dict));

        let tmp_inputs: Vec<String> =
            sorted_contract_inds.iter().map(|&i| current_input_list.remove(i).to_string()).collect();
        let tmp_shapes: Vec<TensorShapeType> =
            sorted_contract_inds.iter().map(|&i| current_input_shapes.remove(i)).collect();

        let do_blas = if use_blas {
            // TODO: Implement BLAS check
            false
        } else {
            false
        };

        // Last contraction
        let idx_result = if cnum == path_tuple.len() - 1 {
            output_subscript.clone()
        } else {
            // use tensordot order to minimize transpositions
            let all_input_inds: String = tmp_inputs.join("");
            let mut sorted_out_inds: Vec<char> = out_inds.iter().cloned().collect();
            sorted_out_inds.sort_by_key(|c| all_input_inds.find(*c));
            sorted_out_inds.into_iter().collect()
        };

        let shp_result = parser::find_output_shape(
            &tmp_inputs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &tmp_shapes,
            &idx_result,
        );

        current_input_list.push(idx_result.clone());
        current_input_shapes.push(shp_result);
        current_input_sets = new_input_sets;

        let einsum_str = format!("{}->{}", tmp_inputs.join(","), idx_result);

        // for large expressions saving the remaining terms at each step can
        // incur a large memory footprint
        let remaining = if current_input_list.len() <= 20 { Some(current_input_list.clone()) } else { None };

        let contraction = ContractionType {
            indices: sorted_contract_inds.iter().join(","),
            idx_rm: idx_removed,
            einsum_str,
            remaining: remaining.map(|v| v.iter().map(|s| s.to_string()).collect()),
            do_blas,
        };
        contraction_list.push(contraction);
    }

    let opt_cost = cost_list.iter().sum();
    let speedup = naive_cost / opt_cost;
    let largest_intermediate = size_intermediates.iter().cloned().reduce(f64::max).unwrap_or(0.0);

    let path_info = PathInfo {
        contraction_list,
        input_subscripts: input_subscripts.clone(),
        output_subscript: output_subscript.clone(),
        indices,
        path: path_tuple.clone(),
        scale_list,
        naive_cost,
        opt_cost,
        speedup,
        size_list: size_intermediates.iter().map(|&x| x as usize).collect(),
        size_dict,
        shapes: input_shapes,
        equation: format!("{input_subscripts}->{output_subscript}"),
        largest_intermediate,
    };

    Ok((path_tuple, path_info))
}

#[test]
fn test_contract_path() {
    let a_shape = vec![4, 4];
    let b_shape = vec![4, 4];
    let subscripts = "ij,jk->ik";
    let (path, path_info) =
        contract_path(subscripts, &[a_shape, b_shape], true, OptimizeKind::Optimized, None).unwrap();

    assert_eq!(path.len(), 1);
    assert_eq!(path_info.input_subscripts, "ij,jk");
    assert_eq!(path_info.output_subscript, "ik");
    println!("{path_info}");
}
