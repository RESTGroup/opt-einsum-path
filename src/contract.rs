//! Contains the primary optimization and contraction routines.

use crate::*;

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
