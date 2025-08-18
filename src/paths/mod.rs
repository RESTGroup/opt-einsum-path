//! Contains the path technology behind opt_einsum in addition to several path helpers.

pub mod branch_bound;
pub mod greedy;
pub mod no_optimize;
pub mod optimal;
pub mod util;

pub(crate) use paths::util::*;

use std::str::FromStr;

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
    Optimized(paths::optimal::Optimal),
    NoOptimize(paths::no_optimize::NoOptimize),
    BranchBound(paths::branch_bound::BranchBound),
    Greedy(paths::greedy::Greedy),
}

impl paths::PathOptimizer for OptimizeKind {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        use OptimizeKind::*;
        match self {
            Optimized(optimizer) => optimizer.optimize_path(inputs, output, size_dict, memory_limit),
            NoOptimize(optimizer) => optimizer.optimize_path(inputs, output, size_dict, memory_limit),
            BranchBound(optimizer) => optimizer.optimize_path(inputs, output, size_dict, memory_limit),
            Greedy(optimizer) => optimizer.optimize_path(inputs, output, size_dict, memory_limit),
        }
    }
}

impl FromStr for OptimizeKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use OptimizeKind::*;
        let optimizer = match s.to_lowercase().replace("_", "-").as_str() {
            "optimal" | "optimized" => Optimized(Default::default()),
            "no-optimize" => NoOptimize(Default::default()),
            "branch-all" => BranchBound(Default::default()),
            "branch-2" => BranchBound("branch-2".into()),
            "branch-1" => BranchBound("branch-1".into()),
            "greedy" => Greedy(Default::default()),
            _ => Err("Unknown optimization kind: {s}")?,
        };
        Ok(optimizer)
    }
}

impl From<&str> for OptimizeKind {
    fn from(s: &str) -> Self {
        OptimizeKind::from_str(s).unwrap()
    }
}

impl From<bool> for OptimizeKind {
    fn from(b: bool) -> Self {
        match b {
            true => OptimizeKind::Optimized(Default::default()),
            false => OptimizeKind::NoOptimize(Default::default()),
        }
    }
}
