use crate::*;

/// Convert a path with static single assignment ids to a path with recycled linear ids.
///
/// # Counterpart in Python
///
/// `opt_einsum.paths.ssa_to_linear`
///
/// # Example
///
/// ```
/// # use opt_einsum_path::paths::util::ssa_to_linear;
/// let ssa_path = [vec![0, 3], vec![2, 4], vec![1, 5]];
/// let linear_path = ssa_to_linear(&ssa_path);
/// assert_eq!(linear_path, &[vec![0, 3], vec![1, 2], vec![0, 1]]);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import ssa_to_linear
/// >>> ssa_to_linear([(0, 3), (2, 4), (1, 5)])
/// [(0, 3), (1, 2), (0, 1)]
/// ```
pub fn ssa_to_linear(ssa_path: &[TensorShapeType]) -> PathType {
    let n = ssa_path.iter().map(|x| x.len()).sum::<usize>() + 1 - ssa_path.len();
    let mut ids = (0..n).collect_vec();
    let mut path = vec![];
    let mut ssa = n;

    println!("Converting SSA path to linear path: {ssa_path:?}");
    println!("Initial ids: {ids:?}");
    for scon in ssa_path {
        // Sort and find positions using search
        let con = scon.iter().map(|&s| ids.binary_search(&s).unwrap()).sorted_unstable().collect_vec();
        println!("con: {con:?}");

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
/// # use opt_einsum_path::paths::util::linear_to_ssa;
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
/// # use num::ToPrimitive;
/// # use opt_einsum_path::paths::util::calc_k12_flops;
/// let inputs = [&"abcd".chars().collect(), &"ac".chars().collect(), &"bdc".chars().collect()];
/// let output = "ad".chars().collect();
/// let remaining = [0, 1, 2];
/// let size_dict = BTreeMap::from([('a', 5), ('b', 2), ('c', 3), ('d', 4)]);
/// let (k12, cost) = calc_k12_flops(&inputs, &output, &remaining, 0, 2, &size_dict);
/// assert_eq!(k12, "acd".chars().collect());
/// assert_eq!(cost.to_usize().unwrap(), 240);
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
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    remaining: &[usize],
    i: usize,
    j: usize,
    size_dict: &SizeDictType,
) -> (ArrayIndexType, SizeType) {
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
/// `opt_einsum.paths.compute_oversize_flops`
///
/// # Example
///
/// ```rust
/// # use std::collections::BTreeMap;
/// # use num::ToPrimitive;
/// # use opt_einsum_path::paths::util::compute_oversize_flops;
/// let inputs = [&"ab".chars().collect(), &"bc".chars().collect(), &"cd".chars().collect()];
/// let remaining = [0, 1, 2];
/// let output = "ad".chars().collect();
/// let size_dict = BTreeMap::from([('a', 2), ('b', 3), ('c', 4), ('d', 5)]);
/// let flops = compute_oversize_flops(&inputs, &remaining, &output, &size_dict);
/// assert_eq!(flops.to_usize().unwrap(), 360);  // abcd->ad
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.paths import compute_oversize_flops
/// >>> inputs = [frozenset("ab"), frozenset("bc"), frozenset("cd")]
/// >>> remaining = [0, 1, 2]
/// >>> output = frozenset("ad")
/// >>> size_dict = {'a': 2, 'b': 3, 'c': 4, 'd': 5}
/// >>> compute_oversize_flops(inputs, remaining, output, size_dict)
/// 360
/// ```
pub fn compute_oversize_flops(
    inputs: &[&ArrayIndexType],
    remaining: &[usize],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
) -> SizeType {
    let num_terms = remaining.len();
    let idx_contraction: ArrayIndexType = remaining.iter().flat_map(|&i| inputs[i].clone()).collect();
    let inner = !idx_contraction.difference(output).collect_vec().is_empty();
    helpers::flop_count(idx_contraction.iter(), inner, num_terms, size_dict)
}
