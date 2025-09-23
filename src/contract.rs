//! Contains the primary optimization and contraction routines.

use crate::*;

/* #region PathInfo */

#[derive(Debug, Clone)]
pub struct PathInfo {
    pub contraction_list: Vec<ContractionType>,
    pub input_subscripts: String,
    pub output_subscript: String,
    pub indices: ArrayIndexType,
    pub path: PathType,
    pub scale_list: Vec<usize>,
    pub naive_cost: SizeType,
    pub opt_cost: SizeType,
    pub speedup: f64,
    pub size_list: Vec<SizeType>,
    pub size_dict: SizeDictType,
    pub shapes: Vec<Vec<usize>>,
    pub equation: String,
    pub largest_intermediate: SizeType,
}

impl PathInfo {
    pub fn new(
        contraction_list: Vec<ContractionType>,
        input_subscripts: String,
        output_subscript: String,
        indices: ArrayIndexType,
        path: PathType,
        scale_list: Vec<usize>,
        naive_cost: SizeType,
        opt_cost: SizeType,
        size_list: Vec<SizeType>,
        size_dict: SizeDictType,
    ) -> Self {
        let speedup = naive_cost.to_f64().unwrap() / opt_cost.to_f64().unwrap().max(1.0);
        let shapes = input_subscripts.split(',').map(|ks| ks.chars().map(|k| size_dict[&k]).collect()).collect();
        let equation = format!("{input_subscripts}->{output_subscript}");
        let largest_intermediate = size_list.clone().into_iter().reduce(SizeType::max).unwrap_or(SizeType::one());

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
            let blas_str = do_blas.unwrap_or("");

            writeln!(f, "{scale:>4} {blas_str:>14} {einsum_str:>22}    {remaining_str:>size_remaining$}")?;
        }

        Ok(())
    }
}

/* #endregion */

#[doc = include_str!("contract_path.md")]
pub fn contract_path(
    subscripts: &str,
    operands: &[TensorShapeType],
    mut optimize: impl PathOptimizer,
    memory_limit: impl Into<SizeLimitType>,
) -> Result<(PathType, PathInfo), String> {
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
        let term_chars = term.chars().collect_vec();
        if sh.len() != term_chars.len() {
            return Err(format!(
                "Einstein sum subscript '{}' does not contain the correct number of indices for operand {tnum}.",
                input_list[tnum]
            ));
        }

        for (cnum, &char) in term_chars.iter().enumerate() {
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
    let size_list: Vec<SizeType> = input_list
        .iter()
        .chain([output_subscript.as_str()].iter())
        .map(|&term| helpers::compute_size_by_dict(term.chars(), &size_dict))
        .collect();
    let size_list_max = size_list.iter().cloned().reduce(SizeType::max).unwrap_or(SizeType::zero());
    let memory_arg = match memory_limit.into() {
        SizeLimitType::None => None,
        SizeLimitType::MaxInput => Some(size_list_max),
        SizeLimitType::Size(size) => Some(size),
    };

    let num_ops = input_list.len();

    // Compute naive cost
    let inner_product = (input_sets.iter().map(|s| s.len()).sum::<usize>() - indices.len()) > 0;
    let naive_cost = helpers::flop_count(indices.iter(), inner_product, num_ops, &size_dict);

    // Compute the path
    let path_tuple = if num_ops <= 2 {
        // Nothing to be optimized
        vec![(0..num_ops).collect()]
    } else {
        // Use the optimizer to find the best contraction path
        optimize.optimize_path(&input_sets_ref, &output_set, &size_dict, memory_arg)?
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
        let mut contract_inds = contract_inds.iter().sorted().rev().cloned().collect_vec();

        let (out_inds, new_input_sets, idx_removed, idx_contract) =
            helpers::find_contraction(&contract_inds, &current_input_sets.iter().collect_vec(), &output_set);

        // Compute cost, scale, and size
        let cost = helpers::flop_count(idx_contract.iter(), !idx_removed.is_empty(), contract_inds.len(), &size_dict);
        cost_list.push(cost);
        scale_list.push(idx_contract.len());
        size_intermediates.push(helpers::compute_size_by_dict(out_inds.iter(), &size_dict));

        let mut tmp_inputs = contract_inds.iter().map(|&i| current_input_list.remove(i)).collect_vec();
        let mut tmp_shapes = contract_inds.iter().map(|&i| current_input_shapes.remove(i)).collect_vec();

        let do_blas = blas::can_blas(
            &tmp_inputs.iter().map(|s| s.as_str()).collect_vec(),
            &out_inds.iter().join(""),
            &idx_removed,
            Some(&tmp_shapes),
        );

        // Last contraction
        let idx_result = if cnum == path_tuple.len() - 1 {
            output_subscript.clone()
        } else {
            // use tensordot order to minimize transpositions
            let all_input_inds = tmp_inputs.join("");
            let mut sorted_out_inds = out_inds.iter().cloned().collect_vec();
            sorted_out_inds.sort_by_key(|c| all_input_inds.find(*c));
            sorted_out_inds.into_iter().collect()
        };

        let shp_result = parser::find_output_shape(
            &tmp_inputs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &tmp_shapes,
            &idx_result,
        );

        // Revert the order to normal
        contract_inds.reverse();
        tmp_inputs.reverse();
        tmp_shapes.reverse();

        current_input_list.push(idx_result.clone());
        current_input_shapes.push(shp_result);
        current_input_sets = new_input_sets;

        let einsum_str = format!("{}->{}", tmp_inputs.join(","), idx_result);

        // for large expressions saving the remaining terms at each step can
        // incur a large memory footprint
        let remaining = if current_input_list.len() <= 20 { Some(current_input_list.clone()) } else { None };

        let contraction =
            ContractionType { indices: contract_inds, idx_rm: idx_removed, einsum_str, remaining, do_blas };
        contraction_list.push(contraction);
    }

    let opt_cost = cost_list.iter().sum::<SizeType>();
    let speedup = naive_cost.to_f64().unwrap() / opt_cost.to_f64().unwrap().max(1.0);
    let largest_intermediate = size_intermediates.iter().cloned().reduce(SizeType::max).unwrap_or(SizeType::zero());

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
        size_list: size_intermediates,
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
        contract_path(subscripts, &[a_shape, b_shape], OptimizeKind::from("optimal"), None).unwrap();

    assert_eq!(path.len(), 1);
    assert_eq!(path_info.input_subscripts, "ij,jk");
    assert_eq!(path_info.output_subscript, "ik");
    println!("{path:?}");
    println!("{path_info}");
    println!("{path_info:?}");
}

#[test]
fn test_contract_path_issue_254() {
    let b = 64;
    let g = 8;
    let k = 4096;
    let d = 128;

    let a_shape = vec![b, g, k];
    let v_shape = vec![b, k, d];
    let s_shape = vec![b, k];
    let subscripts = "bgk,bkd,bk->bgd";
    let time = std::time::Instant::now();
    let (path, path_info) =
        contract_path(subscripts, &[a_shape, v_shape, s_shape], OptimizeKind::from("optimal"), None).unwrap();
    println!("Time: {:?}", time.elapsed());

    assert_eq!(path.len(), 2);
    println!("{path:?}");
    println!("{path_info}");
}

#[test]
fn test_contract_path_issue_248() {
    let no = 300;
    let naux = 50;
    let nx = 2000;

    let m = vec![nx, nx];
    let eta = vec![no, nx];
    let eta1 = vec![naux, nx];
    let theta = vec![naux, no];
    let subscripts = "iP,sP,PQ,jQ,tQ,ti,sj->";
    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(
        subscripts,
        &[eta.clone(), eta1.clone(), m.clone(), eta.clone(), eta1.clone(), theta.clone(), theta.clone()],
        OptimizeKind::from("optimal"),
        None,
    )
    .unwrap();
    println!("Time: {:?}", time.elapsed());

    println!("{path:?}");
    println!("{path_info}");
}

#[test]
fn test_greedy_issue_248() {
    let no = 300;
    let naux = 50;
    let nx = 2000;

    let m = vec![nx, nx];
    let eta = vec![no, nx];
    let eta1 = vec![naux, nx];
    let theta = vec![naux, no];
    let subscripts = "iP,sP,PQ,jQ,tQ,ti,sj->";
    let shapes = vec![eta.clone(), eta1.clone(), m.clone(), eta.clone(), eta1.clone(), theta.clone(), theta.clone()];

    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(subscripts, &shapes, OptimizeKind::from("optimal"), None).unwrap();
    println!("Time (optimal): {:?}", time.elapsed());
    println!("{path:?}");
    println!("{path_info}");

    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(subscripts, &shapes, OptimizeKind::from("greedy"), None).unwrap();
    println!("Time (greedy): {:?}", time.elapsed());
    println!("{path:?}");
    println!("{path_info}");

    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(subscripts, &shapes, OptimizeKind::from("dp"), None).unwrap();
    println!("Time (dp): {:?}", time.elapsed());
    println!("{path:?}");
    println!("{path_info}");

    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(subscripts, &shapes, OptimizeKind::from("random-greedy-128"), None).unwrap();
    println!("Time (dp): {:?}", time.elapsed());
    println!("{path:?}");
    println!("{path_info}");
}

#[test]
fn test_greedy_non_optimal() {
    let m = vec![35, 37, 59];
    let a = vec![35, 51, 59];
    let b = vec![37, 51, 51, 59];
    let c = vec![59, 27];
    let shapes = vec![m, a, b, c];

    let subscripts = "xyf,xtf,ytpf,fr->tpr";
    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(subscripts, &shapes, OptimizeKind::from("greedy"), None).unwrap();
    println!("Time: {:?}", time.elapsed());
    println!("{path:?}");
    println!("{path_info}");

    let subscripts = "xyf,xtf,ytpf,fr->tpr";
    let time = std::time::Instant::now();
    let (path, path_info) = contract_path(subscripts, &shapes, OptimizeKind::from("optimal"), None).unwrap();
    println!("Time: {:?}", time.elapsed());
    println!("{path:?}");
    println!("{path_info}");
}

#[test]
fn test_pathtype_optimizer() {
    let subscripts = "qgcf,sotr,klb,jlretia,hpn,nseha,jgoqm,ipkb,cdfm,d->";
    let shapes = vec![
        vec![5, 2, 9, 4],
        vec![4, 9, 5, 9],
        vec![5, 4, 2],
        vec![5, 4, 9, 7, 5, 3, 6],
        vec![5, 2, 8],
        vec![8, 4, 7, 5, 6],
        vec![5, 2, 9, 5, 8],
        vec![3, 2, 5, 2],
        vec![9, 3, 4, 8],
        vec![3],
    ];
    let path_inp = [[0, 8], [3, 4], [1, 4], [5, 6], [1, 5], [0, 4], [0, 3], [1, 2], [0, 1]];
    let (path, path_info) = contract_path(subscripts, &shapes, path_inp, None).unwrap();
    assert_eq!(path, path_inp);
    println!("{path_info}");

    let path_inp = vec![[0, 8], [3, 4], [1, 4], [5, 6], [1, 5], [0, 4], [0, 3], [1, 3], [0, 1]];
    let path_inp = path_inp.iter().map(|a| a.to_vec()).collect_vec();
    assert!(contract_path(subscripts, &shapes, path_inp.clone(), None).is_err());
}
