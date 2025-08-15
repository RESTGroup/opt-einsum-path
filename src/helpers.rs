//! Contains helper functions for opt_einsum testing scripts.

use crate::*;

/// Computes the product of the elements in indices based on the dictionary
/// idx_dict.
///
/// # Parameters
///
/// * `indices` - Indices to base the product on.
/// * `idx_dict` - Dictionary of index sizes
///
/// # Returns
///
/// The resulting product as an integer.
///
/// # Counterpart in Python
///
/// `opt_einsum.helpers.compute_size_by_dict`
///
/// # Example
///
/// **Using HashMap as a dictionary**
///
/// In this case, indices and dictionary are both passed by reference.
///
/// ```rust
/// # use itertools::Itertools;
/// # use num::ToPrimitive;
/// # use std::collections::{BTreeMap, BTreeSet};
/// # use opt_einsum_path::helpers::compute_size_by_dict;
/// let indices = "abbc".chars();
/// let idx_dict = BTreeMap::from([('a', 2), ('b', 3), ('c', 5)]);
/// let size = compute_size_by_dict(indices, &idx_dict);
/// assert_eq!(size.to_usize().unwrap(), 90);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> opt_einsum.helpers.compute_size_by_dict('abbc', {'a': 2, 'b': 3, 'c': 5})
/// 90
/// ```
pub fn compute_size_by_dict<T>(indices: impl Iterator<Item = T>, idx_dict: &SizeDictType) -> SizeType
where
    T: Borrow<char>,
{
    indices.map(|k| SizeType::from_usize(idx_dict[k.borrow()]).unwrap()).product()
}

/// Finds the contraction details for a given set of input indices, output indices, and positions of
/// terms to contract.
///
/// # Parameters
///
/// * `positions` - Indices of the terms (from `input_sets`) to be contracted.
/// * `input_sets` - List of sets where each set represents the indices of an input term in the
///   einsum expression.
/// * `output_set` - Set representing the indices of the final output of the overall einsum
///   expression.
///
/// # Returns
///
/// A tuple containing:
/// 1. `new_result`: The indices of the resulting intermediate tensor from the contraction.
/// 2. `remaining`: The list of input sets after removing the contracted terms, with `new_result`
///    appended.
/// 3. `idx_removed`: The indices that are summed over (removed) during this contraction.
/// 4. `idx_contract`: All indices involved in the current contraction (from the contracted terms).
///
/// # Counterpart in Python
///
/// `opt_einsum.helpers.find_contraction`
///
/// # Examples
///
/// **A simple dot product test case**
///
/// ```rust
/// # use std::collections::BTreeSet;
/// # use opt_einsum_path::helpers::find_contraction;
/// let positions = [0, 1];
/// let input_sets = ["ab".chars().collect(), "bc".chars().collect()];
/// let input_sets = input_sets.iter().collect::<Vec<_>>();
/// let output_set = "ac".chars().collect();
/// let (new_result, remaining, idx_removed, idx_contract) =
///     find_contraction(&positions, &input_sets, &output_set);
/// assert_eq!(new_result, "ac".chars().collect());
/// assert_eq!(remaining, vec!["ac".chars().collect()]);
/// assert_eq!(idx_removed, "b".chars().collect());
/// assert_eq!(idx_contract, "abc".chars().collect());
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.helpers import find_contraction
/// >>> pos = (0, 1)
/// >>> isets = [set('ab'), set('bc')]
/// >>> oset = set('ac')
/// >>> find_contraction(pos, isets, oset)
/// ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})
/// ```
///
/// **A more complex case with additional terms in the contraction**
///
/// ```rust
/// # use std::collections::BTreeSet;
/// # use opt_einsum_path::helpers::find_contraction;
/// let positions = [0, 2];
/// let input_sets = ["abd".chars().collect(), "ac".chars().collect(), "bdc".chars().collect()];
/// let input_sets = input_sets.iter().collect::<Vec<_>>();
/// let output_set = "ac".chars().collect();
/// let (new_result, remaining, idx_removed, idx_contract) =
///     find_contraction(&positions, &input_sets, &output_set);
/// assert_eq!(new_result, "ac".chars().collect());
/// assert_eq!(remaining, vec!["ac".chars().collect(), "ac".chars().collect()]);
/// assert_eq!(idx_removed, "bd".chars().collect());
/// assert_eq!(idx_contract, "abcd".chars().collect());
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.helpers import find_contraction
/// >>> pos = (0, 2)
/// >>> isets = [set('abd'), set('ac'), set('bdc')]
/// >>> oset = set('ac')
/// >>> find_contraction(pos, isets, oset)
/// ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
/// ```
pub fn find_contraction(
    positions: &[usize],
    input_sets: &[&ArrayIndexType],
    output_set: &ArrayIndexType,
) -> (ArrayIndexType, Vec<ArrayIndexType>, ArrayIndexType, ArrayIndexType) {
    // To developers:
    // - If performance is a concern, consider using `ByteSet` from crate `byte_set` instead (u8 type
    //   string only).
    // - Though Vec<S> is somehow faster, it is not convenient to use, compared to `BTreeSet`.
    // - `HashSet` is not faster in small sets, and it is not ordered.
    let positions = positions.as_ref().to_vec();
    let mut remaining = vec![];
    let mut idx_contract = BTreeSet::new();
    let mut idx_remain = output_set.clone();
    for (i, &set) in input_sets.as_ref().iter().enumerate() {
        match positions.contains(&i) {
            true => idx_contract.extend(set.clone()),
            false => {
                idx_remain.extend(set.clone());
                remaining.push(set.clone());
            },
        }
    }

    let new_result = &idx_remain & &idx_contract;
    let idx_removed = &idx_contract - &new_result;
    remaining.push(new_result.clone());
    (new_result, remaining, idx_removed, idx_contract)
}

/// Computes the number of FLOPS required for a contraction.
///
/// # Parameters
///
/// - `idx_contraction`: Indices involved in the contraction
/// - `inner`: Whether this contraction requires an inner product
/// - `num_terms`: Number of terms in the contraction
/// - `size_dictionary`: Dictionary mapping indices to their sizes
///
/// # Returns
///
/// Total number of FLOPS
///
/// # Counterpart in Python
///
/// `opt_einsum.helpers.flop_count`
///
/// # Examples
///
/// ```rust
/// # use std::collections::BTreeMap;
/// # use num::ToPrimitive;
/// # use opt_einsum_path::helpers::flop_count;
/// # use itertools::Itertools;
/// let mut size_dict = BTreeMap::from([('a', 2), ('b', 3), ('c', 5)]);
/// let flops = flop_count("abc".chars(), false, 1, &size_dict);
/// assert_eq!(flops.to_usize().unwrap(), 30);
/// let flops = flop_count("abc".chars(), true, 2, &size_dict);
/// assert_eq!(flops.to_usize().unwrap(), 60);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> from opt_einsum.helpers import flop_count
/// >>> flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
/// 30
/// >>> flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
/// 60
/// ```
pub fn flop_count<T>(
    idx_contraction: impl Iterator<Item = T>,
    inner: bool,
    num_terms: usize,
    size_dictionary: &SizeDictType,
) -> SizeType
where
    T: Borrow<char>,
{
    let overall_size = compute_size_by_dict(idx_contraction, size_dictionary);
    // let mut op_factor = std::cmp::max(1, num_terms - 1); // may underflow
    let mut op_factor = std::cmp::max(2, num_terms) - 1;
    if inner {
        op_factor += 1;
    }
    overall_size * SizeType::from_usize(op_factor).unwrap()
}

#[test]
fn playground() {
    let positions = [0, 1];
    let input_sets = ["ab".chars().collect(), "bc".chars().collect()];
    let input_sets = input_sets.iter().collect::<Vec<_>>();
    let output_set = "ac".chars().collect();
    let (new_result, remaining, idx_removed, idx_contract) = find_contraction(&positions, &input_sets, &output_set);
    assert_eq!(new_result, "ac".chars().collect());
    assert_eq!(remaining, vec!["ac".chars().collect()]);
    assert_eq!(idx_removed, "b".chars().collect());
    assert_eq!(idx_contract, "abc".chars().collect());
}
