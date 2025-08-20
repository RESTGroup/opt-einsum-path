//! Contains the path technology behind opt_einsum in addition to several path helpers.

pub mod branch_bound;
pub mod dp;
pub mod greedy;
pub mod greedy_random;
pub mod no_optimize;
pub mod optimal;
pub mod util;

use crate::*;
use paths::util::*;
use std::str::FromStr;

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
#[derive(Debug)]
pub enum OptimizeKind {
    Optimal(paths::optimal::Optimal),
    NoOptimize(paths::no_optimize::NoOptimize),
    BranchBound(paths::branch_bound::BranchBound),
    Greedy(paths::greedy::Greedy),
    DynamicProgramming(paths::dp::DynamicProgramming),
    RandomGreedy(paths::greedy_random::RandomGreedy),
    Auto(Auto),
    AutoHq(AutoHq),
}

#[derive(Debug, Default, Clone)]
pub struct Auto {}

#[derive(Debug, Default, Clone)]
pub struct AutoHq {}

impl PathOptimizer for Auto {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        let mut optimizer: Box<dyn PathOptimizer> = match inputs.len() {
            ..5 => Box::new(paths::optimal::Optimal::default()),
            5..7 => Box::new(paths::branch_bound::BranchBound::from("branch-all")),
            7..9 => Box::new(paths::branch_bound::BranchBound::from("branch-2")),
            9..15 => Box::new(paths::branch_bound::BranchBound::from("branch-1")),
            15.. => Box::new(paths::greedy::Greedy::default()),
        };
        optimizer.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

impl PathOptimizer for AutoHq {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        let mut optimizer: Box<dyn PathOptimizer> = match inputs.len() {
            ..6 => Box::new(paths::optimal::Optimal::default()),
            6..17 => Box::new(paths::dp::DynamicProgramming::default()),
            17.. => Box::new(paths::greedy_random::RandomGreedy::from("random-greedy-128")),
        };
        optimizer.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

impl OptimizeKind {
    pub fn optimizer(&mut self) -> &mut dyn PathOptimizer {
        use OptimizeKind::*;
        match self {
            Optimal(optimizer) => optimizer,
            NoOptimize(optimizer) => optimizer,
            BranchBound(optimizer) => optimizer,
            Greedy(optimizer) => optimizer,
            DynamicProgramming(optimizer) => optimizer,
            RandomGreedy(optimizer) => optimizer,
            Auto(optimizer) => optimizer,
            AutoHq(optimizer) => optimizer,
        }
    }
}

impl paths::PathOptimizer for OptimizeKind {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        self.optimizer().optimize_path(inputs, output, size_dict, memory_limit)
    }
}

impl FromStr for OptimizeKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use OptimizeKind::*;

        // special handling for dp;
        if s.starts_with("dp-") {
            return Ok(DynamicProgramming(s.into()));
        }

        // general handling
        let optimizer = match s.replace(['_', ' '], "-").to_lowercase().as_str() {
            "optimal" | "optimized" => Optimal(Default::default()),
            "no-optimize" => NoOptimize(Default::default()),
            "branch-all" => BranchBound(Default::default()),
            "branch-2" => BranchBound("branch-2".into()),
            "branch-1" => BranchBound("branch-1".into()),
            "greedy" | "eager" | "opportunistic" => Greedy(Default::default()),
            "dp" | "dynamic-programming" => DynamicProgramming(Default::default()),
            "random-greedy" => RandomGreedy(Default::default()),
            "random-greedy-128" => RandomGreedy("random-greedy-128".into()),
            "auto" => Auto(Default::default()),
            "auto-hq" => AutoHq(Default::default()),
            _ => Err(format!("Unknown optimization kind: {s}"))?,
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
            true => "auto-hq".into(),
            false => "no-optimize".into(),
        }
    }
}

impl From<Option<bool>> for OptimizeKind {
    fn from(b: Option<bool>) -> Self {
        b.unwrap_or(true).into()
    }
}

impl PathOptimizer for &str {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        let mut optimizer = OptimizeKind::from(*self);
        optimizer.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

impl PathOptimizer for bool {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        let mut optimizer = OptimizeKind::from(*self);
        optimizer.optimize_path(inputs, output, size_dict, memory_limit)
    }
}
