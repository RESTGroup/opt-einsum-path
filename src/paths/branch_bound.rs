use crate::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizeStrategy {
    FlopsFirst,
    SizeFirst,
}

/// functions for comparing which of two paths is 'better'.
pub fn get_better_fn(key: MinimizeStrategy) -> fn(SizeType, SizeType, SizeType, SizeType) -> bool {
    match key {
        MinimizeStrategy::FlopsFirst => {
            |flops, size, best_flops, best_size| flops < best_flops || (flops == best_flops && size < best_size)
        },
        MinimizeStrategy::SizeFirst => {
            |flops, size, best_flops, best_size| size < best_size || (size == best_size && flops < best_flops)
        },
    }
}

/// Returns the appropriate cost function based on the enum variant
pub fn memory_removed(jitter: bool) -> fn(SizeType, SizeType, SizeType, usize, usize, usize) -> SizeType {
    match jitter {
        false => |size12, size1, size2, _k12, _k1, _k2| size12 - size1 - size2,
        true => |size12, size1, size2, _k12, _k1, _k2| {
            let mut rng = rand::rng();
            SizeType::from_f64(rng.random_range(0.99..1.01) * (size12 - size1 - size2).to_f64().unwrap()).unwrap()
        },
    }
}

#[derive(Debug, Clone)]
pub struct BranchBoundBest {
    pub flops: SizeType,
    pub size: SizeType,
    pub ssa_path: Option<PathType>,
}

impl Default for BranchBoundBest {
    fn default() -> Self {
        Self { flops: SizeType::MAX, size: SizeType::MAX, ssa_path: None }
    }
}

#[derive(Debug, Clone)]
pub struct BranchBound {
    pub nbranch: Option<usize>,
    pub cutoff_flops_factor: SizeType,
    pub better_fn: fn(SizeType, SizeType, SizeType, SizeType) -> bool,
    pub cost_fn: fn(SizeType, SizeType, SizeType, usize, usize, usize) -> SizeType,
    pub best: BranchBoundBest,
    pub best_progress: BTreeMap<usize, SizeType>,
}

impl Default for BranchBound {
    fn default() -> Self {
        Self {
            nbranch: None,
            cutoff_flops_factor: SizeType::from_f64(4.0).unwrap(),
            better_fn: get_better_fn(MinimizeStrategy::FlopsFirst),
            cost_fn: memory_removed(false),
            best: BranchBoundBest::default(),
            best_progress: BTreeMap::new(),
        }
    }
}

impl BranchBound {
    pub fn path(&self) -> PathType {
        paths::util::ssa_to_linear(self.best.ssa_path.as_ref().unwrap_or(&Vec::new()))
    }
}

impl PathOptimizer for BranchBound {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        // Reset best state for new optimization
        self.best = BranchBoundBest::default();
        self.best_progress.clear();

        let mut size_cache: BTreeMap<ArrayIndexType, SizeType> =
            inputs.iter().map(|&k| (k.clone(), helpers::compute_size_by_dict(k.iter(), size_dict))).collect();

        #[allow(clippy::type_complexity)]
        let mut result_cache: BTreeMap<(ArrayIndexType, ArrayIndexType), (ArrayIndexType, SizeType)> = BTreeMap::new();

        // Convert inputs to Vec of owned sets for easier manipulation
        let inputs: Vec<ArrayIndexType> = inputs.iter().map(|s| (*s).clone()).collect();

        // Inner recursive function
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::type_complexity)]
        fn branch_iterate(
            branch_bound: &mut BranchBound,
            path: &[TensorShapeType],
            inputs: Vec<ArrayIndexType>,
            remaining: Vec<usize>,
            flops: SizeType,
            size: SizeType,
            size_cache: &mut BTreeMap<ArrayIndexType, SizeType>,
            result_cache: &mut BTreeMap<(ArrayIndexType, ArrayIndexType), (ArrayIndexType, SizeType)>,
            output: &ArrayIndexType,
            size_dict: &SizeDictType,
            memory_limit: Option<SizeType>,
        ) {
            // Reached end of path (only get here if flops is best found so far)
            if remaining.len() == 1 {
                branch_bound.best.flops = flops;
                branch_bound.best.size = size;
                branch_bound.best.ssa_path = Some(path.to_vec());
                return;
            }

            #[derive(Debug, Clone, PartialEq, PartialOrd)]
            struct BranchBoundCandidate {
                cost: SizeType,
                flops12: SizeType,
                new_flops: SizeType,
                new_size: SizeType,
                pair: (usize, usize),
                k12: ArrayIndexType,
            }

            impl Eq for BranchBoundCandidate {}

            #[allow(clippy::derive_ord_xor_partial_ord)]
            impl Ord for BranchBoundCandidate {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.cost.partial_cmp(&other.cost).unwrap().reverse()
                }
            }

            // Assess a candidate contraction
            #[allow(clippy::too_many_arguments)]
            #[allow(clippy::type_complexity)]
            fn assess_candidate(
                branch_bound: &mut BranchBound,
                k1: &ArrayIndexType,
                k2: &ArrayIndexType,
                i: usize,
                j: usize,
                inputs: &[ArrayIndexType],
                remaining: &[usize],
                output: &ArrayIndexType,
                size_dict: &SizeDictType,
                size_cache: &mut BTreeMap<ArrayIndexType, SizeType>,
                result_cache: &mut BTreeMap<(ArrayIndexType, ArrayIndexType), (ArrayIndexType, SizeType)>,
                path: &[TensorShapeType],
                flops: SizeType,
                size: SizeType,
                memory_limit: Option<SizeType>,
            ) -> Option<BranchBoundCandidate> {
                let key = (k1.clone(), k2.clone());
                let (k12, flops12) = result_cache
                    .entry(key)
                    .or_insert_with(|| {
                        paths::util::calc_k12_flops(
                            &inputs.iter().collect::<Vec<_>>(),
                            output,
                            remaining,
                            i,
                            j,
                            size_dict,
                        )
                    })
                    .clone();

                let size12 = *size_cache
                    .entry(k12.clone())
                    .or_insert_with(|| helpers::compute_size_by_dict(k12.iter(), size_dict));

                let new_flops = flops + flops12;
                let new_size = size.max(size12);

                // Sieve based on current best
                if !(branch_bound.better_fn)(new_flops, new_size, branch_bound.best.flops, branch_bound.best.size) {
                    return None;
                }

                // Sieve based on progress relative to best
                let current_len = inputs.len();
                let best_progress = branch_bound.best_progress.entry(current_len).or_insert(SizeType::MAX);
                if new_flops < *best_progress {
                    *best_progress = new_flops;
                } else if new_flops > branch_bound.cutoff_flops_factor * *best_progress {
                    return None;
                }

                // Sieve based on memory limit
                if let Some(limit) = memory_limit
                    && size12 > limit
                {
                    // Terminate path here, but check all-terms contract first
                    let oversize_flops = flops
                        + paths::util::compute_oversize_flops(
                            &inputs.iter().collect::<Vec<_>>(),
                            remaining,
                            output,
                            size_dict,
                        );
                    if oversize_flops < branch_bound.best.flops {
                        branch_bound.best.flops = oversize_flops;
                        let mut new_path = path.to_vec();
                        new_path.push(remaining.to_vec());
                        branch_bound.best.ssa_path = Some(new_path);
                    }
                    return None;
                }

                // Calculate cost heuristic
                let size1 = size_cache[k1];
                let size2 = size_cache[k2];
                let cost = (branch_bound.cost_fn)(size12, size1, size2, k12.len(), k1.len(), k2.len());

                Some(BranchBoundCandidate { cost, flops12, new_flops, new_size, pair: (i, j), k12 })
            }

            // Check all possible remaining paths
            let mut candidates = BinaryHeap::new();
            for (i, j) in remaining.iter().tuple_combinations() {
                let (i, j) = if i < j { (*i, *j) } else { (*j, *i) };
                let k1 = &inputs[i];
                let k2 = &inputs[j];

                // Initially ignore outer products
                if k1.is_disjoint(k2) {
                    continue;
                }

                if let Some(candidate) = assess_candidate(
                    branch_bound,
                    k1,
                    k2,
                    i,
                    j,
                    &inputs,
                    &remaining,
                    output,
                    size_dict,
                    size_cache,
                    result_cache,
                    path,
                    flops,
                    size,
                    memory_limit,
                ) {
                    candidates.push(Reverse(candidate));
                }
            }

            // Assess outer products if nothing left
            if candidates.is_empty() {
                for (i, j) in remaining.iter().tuple_combinations() {
                    let (i, j) = if i < j { (*i, *j) } else { (*j, *i) };
                    let k1 = &inputs[i];
                    let k2 = &inputs[j];

                    if let Some(candidate) = assess_candidate(
                        branch_bound,
                        k1,
                        k2,
                        i,
                        j,
                        &inputs,
                        &remaining,
                        output,
                        size_dict,
                        size_cache,
                        result_cache,
                        path,
                        flops,
                        size,
                        memory_limit,
                    ) {
                        candidates.push(Reverse(candidate));
                    }
                }
            }

            // Recurse into all or some of the best candidate contractions
            let mut bi = 0;
            while (branch_bound.nbranch.is_none() || bi < branch_bound.nbranch.unwrap()) && !candidates.is_empty() {
                let Reverse(candidate) = candidates.pop().unwrap();
                let BranchBoundCandidate { new_flops, new_size, pair: (i, j), k12, .. } = candidate;

                let mut new_remaining = remaining.clone();
                new_remaining.retain(|&x| x != i && x != j);
                new_remaining.push(inputs.len());

                let mut new_inputs = inputs.clone();
                new_inputs.push(k12);

                let mut new_path = path.to_vec();
                new_path.push(vec![i, j]);

                branch_iterate(
                    branch_bound,
                    &new_path,
                    new_inputs,
                    new_remaining,
                    new_flops,
                    new_size,
                    size_cache,
                    result_cache,
                    output,
                    size_dict,
                    memory_limit,
                );

                bi += 1;
            }
        }

        // Start the recursive process
        let inputs_len = inputs.len();
        branch_iterate(
            self,
            &Vec::new(),
            inputs,
            (0..inputs_len).collect(),
            SizeType::zero(),
            SizeType::zero(),
            &mut size_cache,
            &mut result_cache,
            output,
            size_dict,
            memory_limit,
        );

        self.path()
    }
}

/* #endregion */
