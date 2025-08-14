//! Contains the path technology behind opt_einsum in addition to several path helpers.

use crate::*;
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet};

pub trait PathOptimizer {
    fn optimize_path(
        &self,
        inputs: &[&BTreeSet<char>],
        output: &BTreeSet<char>,
        size_dict: &BTreeMap<char, usize>,
        memory_limit: Option<usize>,
    ) -> PathType;
}

/// Convert a path with static single assignment ids to a path with recycled linear ids.
///
/// # Counterpart in Python
///
/// `opt_einsum.paths.ssa_to_linear`
///
/// # Example
///
/// ```
/// # use opt_einsum_path::paths::ssa_to_linear;
/// let ssa_path = vec![vec![0, 3], vec![2, 4], vec![1, 5]];
/// let linear_path = ssa_to_linear(ssa_path);
/// assert_eq!(linear_path, vec![vec![0, 3], vec![1, 2], vec![0, 1]]);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import ssa_to_linear
/// >>> ssa_to_linear([(0, 3), (2, 4), (1, 5)])
/// [(0, 3), (1, 2), (0, 1)]
/// ```
pub fn ssa_to_linear(ssa_path: PathType) -> PathType {
    let n = ssa_path.iter().map(|x| x.len()).sum::<usize>() + 1 - ssa_path.len();
    let mut ids = (0..n).collect_vec();
    let mut path = vec![];
    let mut ssa = n;

    for scon in ssa_path {
        // Sort and find positions using search
        let con = scon.iter().map(|&s| ids.binary_search(&s).unwrap()).sorted_unstable().collect_vec();

        // Remove elements in reverse order to avoid index shifting issues
        for &j in con.iter().rev() {
            ids.remove(j);
        }

        ids.push(ssa);
        path.push(con);
        ssa += 1;
    }

    path
}

/// Convert a path with recycled linear ids to a path with static single assignment ids.
///
/// # Counterpart in Python
///
/// `opt_einsum.paths.linear_to_ssa`
///
/// # Example
///
/// ```
/// # use opt_einsum_path::paths::linear_to_ssa;
/// let linear_path = vec![vec![0, 3], vec![1, 2], vec![0, 1]];
/// let ssa_path = linear_to_ssa(linear_path);
/// assert_eq!(ssa_path, vec![vec![0, 3], vec![2, 4], vec![1, 5]]);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import linear_to_ssa
/// >>> linear_to_ssa([(0, 3), (1, 2), (0, 1)])
/// [(0, 3), (2, 4), (1, 5)]
/// ```
pub fn linear_to_ssa(path: PathType) -> PathType {
    let num_inputs = path.iter().map(|x| x.len()).sum::<usize>() + 1 - path.len();
    let mut linear_to_ssa = (0..num_inputs).collect_vec();
    let mut new_id = num_inputs;
    let mut ssa_path = vec![];

    for ids in path {
        // Convert linear IDs to SSA IDs
        let ssa_ids = ids.iter().map(|&id| linear_to_ssa[id]).collect_vec();
        ssa_path.push(ssa_ids);

        // Remove used IDs in reverse order
        for &id in ids.iter().sorted_unstable().rev() {
            linear_to_ssa.remove(id);
        }

        // Add new SSA ID
        linear_to_ssa.push(new_id);
        new_id += 1;
    }

    ssa_path
}

/// Calculate the resulting indices and flops for a potential pairwise contraction - used in the
/// recursive (optimal/branch) algorithms.
///
/// # Parameters
///
/// - `inputs`: The indices of each tensor in this contraction, note this includes tensors
///   unavailable to contract as static single assignment is used: contracted tensors are not
///   removed from the list.
/// - `output`: The set of output indices for the whole contraction.
/// - `remaining`: The set of indices (corresponding to `inputs`) of tensors still available to
///   contract.
/// - `i`: Index of potential tensor to contract.
/// - `j`: Index of potential tensor to contract.
/// - `size_dict`: Size mapping of all the indices.
///
/// # Returns
///
/// - `k12`: The resulting indices of the potential tensor.
/// - `cost`: Estimated flop count of operation.
///
/// # Counterpart in Python
///
/// `opt_einsum.paths.calc_k12_flops`
///
/// # Example
///
/// ```rust
/// # use std::collections::BTreeMap;
/// # use opt_einsum_path::paths::calc_k12_flops;
/// let inputs = [&"abcd".chars().collect(), &"ac".chars().collect(), &"bdc".chars().collect()];
/// let output = "ad".chars().collect();
/// let remaining = [0, 1, 2];
/// let size_dict = BTreeMap::from([('a', 5), ('b', 2), ('c', 3), ('d', 4)]);
/// let (k12, cost) = calc_k12_flops(&inputs, &output, &remaining, 0, 2, &size_dict);
/// assert_eq!(k12, "acd".chars().collect());
/// assert_eq!(cost, 240);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import calc_k12_flops
/// >>> inputs = [set("abcd"), set("ac"), set("bdc")]
/// >>> output = frozenset("ad")
/// >>> remaining = set([0, 1, 2])
/// >>> size_dict = {'a': 5, 'b': 2, 'c': 3, 'd': 4}
/// >>> calc_k12_flops(inputs, output, remaining, 0, 2, size_dict)
/// ({'a', 'c', 'd'}, 240)
/// ```
pub fn calc_k12_flops(
    inputs: &[&BTreeSet<char>],
    output: &BTreeSet<char>,
    remaining: &[usize],
    i: usize,
    j: usize,
    size_dict: &BTreeMap<char, usize>,
) -> (BTreeSet<char>, usize) {
    let k1 = inputs[i];
    let k2 = inputs[j];

    // Compute union and intersection
    let either = k1 | k2;
    let shared = k1 & k2;

    // Compute keep set
    let mut keep = output.clone();
    for &idx in remaining {
        if idx != i && idx != j {
            keep.extend(inputs[idx]);
        }
    }

    // Compute k12 and cost
    let k12 = &either & &keep;
    let inner = !shared.difference(&keep).collect_vec().is_empty();
    let cost = helpers::flop_count(either.iter(), inner, 2, size_dict);

    (k12, cost)
}

/// Compute the flop count for a contraction of all remaining arguments. This
/// is used when a memory limit means that no pairwise contractions can be made.
///
/// # Parameters
///
/// - `inputs`: The indices of each tensor in the contraction
/// - `remaining`: Indices of tensors still available to contract
/// - `output`: The set of output indices for the whole contraction
/// - `size_dict`: Size mapping of all the indices
///
/// # Returns
///
/// Estimated flop count for contracting all remaining tensors at once
///
/// # Counterpart in Python
///
/// `opt_einsum.paths._compute_oversize_flops`
///
/// # Example
///
/// ```rust
/// # use std::collections::BTreeMap;
/// # use opt_einsum_path::paths::_compute_oversize_flops;
/// let inputs = [&"ab".chars().collect(), &"bc".chars().collect(), &"cd".chars().collect()];
/// let remaining = [0, 1, 2];
/// let output = "ad".chars().collect();
/// let size_dict = BTreeMap::from([('a', 2), ('b', 3), ('c', 4), ('d', 5)]);
/// let flops = _compute_oversize_flops(&inputs, &remaining, &output, &size_dict);
/// assert_eq!(flops, 360);  // abcd->ad
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import _compute_oversize_flops
/// >>> inputs = [frozenset("ab"), frozenset("bc"), frozenset("cd")]
/// >>> remaining = [0, 1, 2]
/// >>> output = frozenset("ad")
/// >>> size_dict = {'a': 2, 'b': 3, 'c': 4, 'd': 5}
/// >>> _compute_oversize_flops(inputs, remaining, output, size_dict)
/// 360
/// ```
pub fn _compute_oversize_flops(
    inputs: &[&BTreeSet<char>],
    remaining: &[usize],
    output: &BTreeSet<char>,
    size_dict: &BTreeMap<char, usize>,
) -> usize {
    let num_terms = remaining.len();
    let idx_contraction: BTreeSet<char> = remaining.iter().flat_map(|&i| inputs[i].clone()).collect();
    let inner = !idx_contraction.difference(output).collect_vec().is_empty();
    helpers::flop_count(idx_contraction.iter(), inner, num_terms, size_dict)
}

struct _OptimalIterConsts {
    output: BTreeSet<char>,
    size_dict: BTreeMap<char, usize>,
    memory_limit: Option<usize>,
}

#[allow(clippy::type_complexity)]
struct _OptimalIterCaches {
    best_flops: usize,
    best_ssa_path: Vec<Vec<usize>>,
    size_cache: BTreeMap<BTreeSet<char>, usize>,
    result_cache: BTreeMap<(BTreeSet<char>, BTreeSet<char>), (BTreeSet<char>, usize)>,
}

fn _optimal_iterate(
    path: Vec<Vec<usize>>,
    remaining: &[usize],
    inputs: &[&BTreeSet<char>],
    flops: usize,
    consts: &_OptimalIterConsts,
    caches: &mut _OptimalIterCaches,
) {
    let _OptimalIterConsts { output, size_dict, memory_limit } = &consts;

    // Reached end of path (only get here if flops is best found so far)
    if remaining.len() == 1 {
        caches.best_flops = flops;
        caches.best_ssa_path = path;
        return;
    }

    // Generate all possible pairs
    for i in 0..remaining.len() {
        for j in (i + 1)..remaining.len() {
            let a = remaining[i];
            let b = remaining[j];
            let (i, j) = if a < b { (a, b) } else { (b, a) };

            let key = (inputs[i].clone(), inputs[j].clone());
            let (k12, flops12) = caches
                .result_cache
                .entry(key.clone())
                .or_insert_with(|| calc_k12_flops(inputs, output, remaining, i, j, size_dict))
                .clone();

            // Sieve based on current best flops
            let new_flops = flops + flops12;
            if new_flops >= caches.best_flops {
                continue;
            }

            // Sieve based on memory limit
            if let Some(limit) = memory_limit {
                let size12 = caches
                    .size_cache
                    .entry(k12.clone())
                    .or_insert_with(|| helpers::compute_size_by_dict(k12.iter(), size_dict));

                // Possibly terminate this path with an all-terms einsum
                if *size12 > *limit {
                    let oversize_flops = flops + _compute_oversize_flops(inputs, remaining, output, size_dict);
                    if oversize_flops < caches.best_flops {
                        caches.best_flops = oversize_flops;
                        let mut new_path = path.clone();
                        new_path.push(remaining.to_vec());
                        caches.best_ssa_path = new_path;
                    }
                    continue;
                }
            }

            // Add contraction and recurse
            let mut new_remaining = remaining.to_vec();
            new_remaining.retain(|&x| x != i && x != j);
            new_remaining.push(inputs.len());
            let mut new_inputs = inputs.to_vec();
            new_inputs.push(&k12);

            let mut new_path = path.clone();
            new_path.push(vec![i, j]);

            _optimal_iterate(new_path, &new_remaining, &new_inputs, new_flops, consts, caches);
        }
    }
}

/// Computes all possible pair contractions in a depth-first recursive manner,
/// sieving results based on `memory_limit` and the best path found so far.
///
/// # Parameters
///
/// - `inputs`: List of sets that represent the lhs side of the einsum subscript
/// - `output`: Set that represents the rhs side of the overall einsum subscript
/// - `size_dict`: Dictionary of index sizes
/// - `memory_limit`: The maximum number of elements in a temporary array
///
/// # Returns
///
/// The optimal contraction order within the memory limit constraint
///
/// # Example
///
/// ```rust
/// # use std::collections::BTreeMap;
/// # use opt_einsum_path::paths::optimal;
/// let inputs = [&"abd".chars().collect(), &"ac".chars().collect(), &"bdc".chars().collect()];
/// let output = "".chars().collect();
/// let size_dict = BTreeMap::from([('a', 1), ('b', 2), ('c', 3), ('d', 4)]);
/// let path = optimal(&inputs, &output, &size_dict, Some(5000));
/// assert_eq!(path, vec![vec![0, 2], vec![0, 1]]);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import optimal
/// >>> isets = [set('abd'), set('ac'), set('bdc')]
/// >>> oset = set('')
/// >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
/// >>> optimal(isets, oset, idx_sizes, 5000)
/// [(0, 2), (0, 1)]
/// ```
pub fn optimal(
    inputs: &[&BTreeSet<char>],
    output: &BTreeSet<char>,
    size_dict: &BTreeMap<char, usize>,
    memory_limit: Option<usize>,
) -> PathType {
    let best_flops = usize::MAX;
    let best_ssa_path: Vec<Vec<usize>> = (0..inputs.len()).map(|i| vec![i]).collect();
    let size_cache = BTreeMap::new();
    let result_cache = BTreeMap::new();
    let consts = _OptimalIterConsts { output: output.clone(), size_dict: size_dict.clone(), memory_limit };
    let mut caches = _OptimalIterCaches { best_flops, best_ssa_path, size_cache, result_cache };

    _optimal_iterate(Vec::new(), &(0..inputs.len()).collect_vec(), inputs, 0, &consts, &mut caches);
    let best_ssa_path = caches.best_ssa_path;
    ssa_to_linear(best_ssa_path)
}

#[test]
fn playground() {
    use std::collections::BTreeMap;
    let time = std::time::Instant::now();
    let inputs = [&"abd".chars().collect(), &"ac".chars().collect(), &"bdc".chars().collect()];
    let output = "".chars().collect();
    let size_dict = BTreeMap::from([('a', 1), ('b', 2), ('c', 3), ('d', 4)]);
    let path = optimal(&inputs, &output, &size_dict, Some(5000));
    assert_eq!(path, vec![vec![0, 2], vec![0, 1]]);
    let duration = time.elapsed();
    println!("Optimal path found in: {duration:?}");
}
