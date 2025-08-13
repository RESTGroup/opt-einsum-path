//! Contains helper functions for opt_einsum testing scripts.

use std::collections::BTreeSet;
use std::ops::Index;

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
/// # use std::collections::HashMap;
/// # use opt_einsum_path::helpers::compute_size_by_dict;
/// let indices = "abbc".chars();
/// let idx_dict = HashMap::from([('a', 2), ('b', 3), ('c', 5)]);
/// let size = compute_size_by_dict(indices, &idx_dict);
/// assert_eq!(size, 90);
/// ```
///
/// Python equivalent:
///
/// ```python
/// >>> opt_einsum.helpers.compute_size_by_dict('abbc', {'a': 2, 'b': 3, 'c': 5})
/// 90
/// ```
pub fn compute_size_by_dict<S, D>(indices: S, idx_dict: &D) -> usize
where
    S: IntoIterator<Item = char>,
    D: for<'a> Index<&'a char, Output = usize>,
{
    indices.into_iter().map(|k| idx_dict[&k]).product()
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
/// let positions = vec![0, 1];
/// let input_sets = vec!["ab".chars(), "bc".chars()];
/// let output_set = "ac".chars();
/// let (new_result, remaining, idx_removed, idx_contract) =
///     find_contraction(positions, input_sets, output_set);
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
/// let positions = vec![0, 2];
/// let input_sets = vec!["abd".chars(), "ac".chars(), "bdc".chars()];
/// let output_set = "ac".chars();
/// let (new_result, remaining, idx_removed, idx_contract) =
///     find_contraction(positions, input_sets, output_set);
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
pub fn find_contraction<S>(
    positions: impl AsRef<[usize]>,
    input_sets: impl AsRef<[S]>,
    output_set: S,
) -> (BTreeSet<char>, Vec<BTreeSet<char>>, BTreeSet<char>, BTreeSet<char>)
where
    S: Iterator<Item = char> + Clone,
{
    // To developers:
    // - If performance is a concern, consider using `ByteSet` from crate `byte_set` instead (u8 type
    //   string only).
    // - Though Vec<S> is somehow faster, it is not convenient to use, compared to `BTreeSet`.
    // - `HashSet` is not faster in small sets, and it is not ordered.
    let positions = positions.as_ref().to_vec();
    let mut remaining = vec![];
    let mut idx_contract = BTreeSet::new();
    let mut idx_remain = output_set.clone().collect::<BTreeSet<char>>();
    for (i, set) in input_sets.as_ref().iter().enumerate() {
        match positions.contains(&i) {
            true => idx_contract.extend(set.clone()),
            false => {
                idx_remain.extend(set.clone());
                remaining.push(set.clone().collect());
            },
        }
    }

    let new_result = idx_remain.intersection(&idx_contract).cloned().collect();
    let idx_removed = idx_contract.difference(&new_result).cloned().collect();
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
/// # use std::collections::HashMap;
/// # use opt_einsum_path::helpers::flop_count;
/// # use itertools::Itertools;
/// let mut size_dict = HashMap::from([('a', 2), ('b', 3), ('c', 5)]);
/// assert_eq!(flop_count("abc".chars(), false, 1, &size_dict), 30);
/// assert_eq!(flop_count("abc".chars(), true, 2, &size_dict), 60);
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
pub fn flop_count<S, D>(idx_contraction: S, inner: bool, num_terms: usize, size_dictionary: &D) -> usize
where
    S: IntoIterator<Item = char>,
    D: for<'a> Index<&'a char, Output = usize>,
{
    let overall_size = compute_size_by_dict(idx_contraction, size_dictionary);
    // let mut op_factor = std::cmp::max(1, num_terms - 1); // may underflow
    let mut op_factor = std::cmp::max(2, num_terms) - 1;
    if inner {
        op_factor += 1;
    }
    overall_size * op_factor
}

#[test]
fn playground() {
    use std::collections::HashMap;
    let indices = "abbc".chars();
    let idx_dict = HashMap::from([('a', 2), ('b', 3), ('c', 5)]);
    let size = compute_size_by_dict(indices, &idx_dict);
    assert_eq!(size, 90);
}
