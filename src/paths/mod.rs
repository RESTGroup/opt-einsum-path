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
    ) -> Result<PathType, String>;
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
    ) -> Result<PathType, String> {
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
    ) -> Result<PathType, String> {
        let mut optimizer: Box<dyn PathOptimizer> = match inputs.len() {
            ..6 => Box::new(paths::optimal::Optimal::default()),
            6..20 => Box::new(paths::dp::DynamicProgramming::default()),
            20.. => Box::new(paths::greedy_random::RandomGreedy::from("random-greedy-128")),
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

impl PathOptimizer for OptimizeKind {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        // capture panics from the optimizer and convert to error
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.optimizer().optimize_path(inputs, output, size_dict, memory_limit)
        }))
        .unwrap_or_else(|err| Err(format!("Optimizer panicked: {err:?}")))
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
        if s.starts_with("random-greedy") {
            return Ok(RandomGreedy(s.into()));
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
    ) -> Result<PathType, String> {
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
    ) -> Result<PathType, String> {
        let mut optimizer = OptimizeKind::from(*self);
        optimizer.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

/* #region special impl for PathType (Vec<Vec<usize>> or related) */

impl PathOptimizer for PathType {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        _output: &ArrayIndexType,
        _size_dict: &SizeDictType,
        _memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        // simple validation
        {
            let mut n = inputs.len();
            for indices in self.iter() {
                if indices.is_empty() {
                    return Err("Empty path step found".to_string());
                }
                let mut indices = indices.to_vec();
                indices.sort_unstable();
                // check largest index is less than n
                if indices.last().unwrap() >= &n {
                    return Err(format!("Path step index {} out of bounds for {n} inputs", indices.last().unwrap()));
                }
                // update n by removing the contracted indices
                n -= indices.len() - 1;
            }
            if n != 1 {
                return Err(format!("Path does not reduce to single output, ended with {n} tensors"));
            }
        }
        Ok(self.clone())
    }
}

#[duplicate::duplicate_item(ImplType; [&[Vec<usize>]]; [&[&[usize]]]; [Vec<[usize; 2]>]; [&[[usize; 2]]])]
impl PathOptimizer for ImplType {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        let mut path = self.iter().map(|step| step.to_vec()).collect::<PathType>();
        path.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

#[duplicate::duplicate_item(ImplType; [[Vec<usize>; N]]; [[[usize; 2]; N]])]
impl<const N: usize> PathOptimizer for ImplType {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        let mut path = self.iter().map(|step| step.to_vec()).collect::<PathType>();
        path.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

#[duplicate::duplicate_item(ImplType; [Vec<(usize, usize)>]; [&[(usize, usize)]])]
impl PathOptimizer for ImplType {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        let mut path = self.iter().map(|step| vec![step.0, step.1]).collect::<PathType>();
        path.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

#[duplicate::duplicate_item(ImplType; [[(usize, usize); N]])]
impl<const N: usize> PathOptimizer for ImplType {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        let mut path = self.iter().map(|step| vec![step.0, step.1]).collect::<PathType>();
        path.optimize_path(inputs, output, size_dict, memory_limit)
    }
}

/* #endregion */
