use crate::*;

struct _OptimalIterConsts {
    output: ArrayIndexType,
    size_dict: SizeDictType,
    memory_limit: Option<SizeType>,
}

#[allow(clippy::type_complexity)]
struct _OptimalIterCaches {
    best_flops: SizeType,
    best_ssa_path: PathType,
    size_cache: BTreeMap<ArrayIndexType, SizeType>,
}

fn _optimal_iterate(
    path: PathType,
    remaining: &[usize],
    inputs: &[&ArrayIndexType],
    flops: SizeType,
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

            let (k12, flops12) = paths::util::calc_k12_flops(inputs, output, remaining, i, j, size_dict);

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
                    let oversize_flops =
                        flops + paths::util::compute_oversize_flops(inputs, remaining, output, size_dict);
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
/// # use opt_einsum_path::typing::*;
/// # use num::FromPrimitive;
/// # use opt_einsum_path::paths::optimal::optimal;
/// let inputs = [&"abd".chars().collect(), &"ac".chars().collect(), &"bdc".chars().collect()];
/// let output = "".chars().collect();
/// let size_dict = BTreeMap::from([('a', 1), ('b', 2), ('c', 3), ('d', 4)]);
/// let path = optimal(&inputs, &output, &size_dict, Some(SizeType::from_usize(5000).unwrap()));
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
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    memory_limit: Option<SizeType>,
) -> PathType {
    let best_flops = SizeType::MAX;
    let best_ssa_path = (0..inputs.len()).map(|i| vec![i]).collect();
    let size_cache = BTreeMap::new();
    let consts = _OptimalIterConsts { output: output.clone(), size_dict: size_dict.clone(), memory_limit };
    let mut caches = _OptimalIterCaches { best_flops, best_ssa_path, size_cache };

    _optimal_iterate(Vec::new(), &(0..inputs.len()).collect_vec(), inputs, SizeType::zero(), &consts, &mut caches);
    paths::util::ssa_to_linear(&caches.best_ssa_path)
}

pub struct Optimal;

impl PathOptimizer for Optimal {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        optimal(inputs, output, size_dict, memory_limit)
    }
}

#[test]
fn playground() {
    use std::collections::BTreeMap;
    let time = std::time::Instant::now();
    let inputs = [&"abd".chars().collect(), &"ac".chars().collect(), &"bdc".chars().collect()];
    let output = "".chars().collect();
    let size_dict = BTreeMap::from([('a', 1), ('b', 2), ('c', 3), ('d', 4)]);
    let path = optimal(&inputs, &output, &size_dict, Some(SizeType::from_usize(5000).unwrap()));
    assert_eq!(path, vec![vec![0, 2], vec![0, 1]]);
    let duration = time.elapsed();
    println!("Optimal path found in: {duration:?}");
}

#[test]
fn playground_issue() {
    use std::collections::BTreeMap;
    let time = std::time::Instant::now();
    let inputs = [&"bgk".chars().collect(), &"bkd".chars().collect(), &"bk".chars().collect()];
    let output = "bgd".chars().collect();
    let size_dict = BTreeMap::from([('b', 64), ('g', 8), ('k', 4096), ('d', 128)]);
    let path = optimal(&inputs, &output, &size_dict, None);
    println!("{path:?}");
    let duration = time.elapsed();
    println!("Optimal path found in: {duration:?}");
}
