//! Contains the path technology behind opt_einsum in addition to several path helpers.

pub mod branch_bound;
pub mod no_optimize;
pub mod optimal;
pub mod util;

use crate::*;

pub trait PathOptimizer {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType;
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum OptimizeKind {
    Optimized,
    BranchBound(paths::branch_bound::BranchBound),
}

impl paths::PathOptimizer for OptimizeKind {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        match self {
            OptimizeKind::Optimized => {
                let mut optimizer = paths::optimal::Optimal::default();
                optimizer.optimize_path(inputs, output, size_dict, memory_limit)
            },
            OptimizeKind::BranchBound(optimizer) => optimizer.optimize_path(inputs, output, size_dict, memory_limit),
        }
    }
}

impl From<&str> for OptimizeKind {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "optimized" => OptimizeKind::Optimized,
            "branch-all" => OptimizeKind::BranchBound(Default::default()),
            _ => panic!("Unknown optimization kind: {s}"),
        }
    }
}
