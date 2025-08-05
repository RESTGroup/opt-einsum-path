//! Contains helper functions for opt_einsum testing scripts.

use itertools::Itertools;
use std::collections::{HashMap, HashSet};

/// Computes the product of the elements in integer indices based on a list of sizes.
///
/// # Parameters
/// - `indices`: Indices to base the product on (integer indices into the size list)
/// - `idx_dict`: List where each position contains the size for that index
///
/// # Returns
/// The resulting product of sizes
///
/// # Examples
/// ```
/// # use opt_einsum_path::helper::compute_size_by_int_indices;
/// let idx_dict = [2, 3, 5];
/// let indices = [0, 1, 1, 2];
/// assert_eq!(compute_size_by_int_indices(&indices, &idx_dict), 2 * 3 * 3 * 5);
/// ```
pub fn compute_size_by_int_indices(indices: &[usize], idx_dict: &[usize]) -> usize {
    indices.iter().map(|&i| idx_dict[i]).product()
}

/// Computes the product of the elements in character indices based on a dictionary of sizes.
///
/// # Parameters
/// - `indices`: Indices to base the product on (character indices)
/// - `idx_dict`: Dictionary mapping characters to their sizes
///
/// # Returns
/// The resulting product of sizes
///
/// # Examples
/// ```
/// # use std::collections::HashMap;
/// # use opt_einsum_path::helper::compute_size_by_char_indices;
/// let mut idx_dict = HashMap::new();
/// idx_dict.insert('a', 2);
/// idx_dict.insert('b', 3);
/// idx_dict.insert('c', 5);
///
/// let indices = ['a', 'b', 'b', 'c'];
/// assert_eq!(compute_size_by_char_indices(&indices, &idx_dict), 2 * 3 * 3 * 5);
/// ```
pub fn compute_size_by_char_indices(indices: &[char], idx_dict: &HashMap<char, usize>) -> usize {
    indices.iter().map(|c| idx_dict[c]).product()
}

/// Finds the contraction details for a given set of input and output indices.
///
/// # Parameters
/// - `positions`: Integer positions of terms used in the contraction
/// - `input_sets`: List of sets representing the LHS indices of the einsum subscript
/// - `output_set`: Set representing the RHS indices of the overall einsum subscript
///
/// # Returns
/// A tuple containing:
/// - new_result: Indices of the resulting contraction
/// - remaining: List of remaining index sets with new result appended
/// - idx_removed: Indices removed from the entire contraction
/// - idx_contraction: Indices used in the current contraction
///
/// # Examples
/// ```
/// # use std::collections::HashSet;
/// # use opt_einsum_path::helper::find_contraction;
/// // Simple dot product test case
/// let positions = &[0, 1];
/// let input_sets = vec![['a', 'b'].into(), ['b', 'c'].into()];
/// let output_set = ['a', 'c'].into();
///
/// let (new_result, remaining, idx_removed, idx_contraction) =
///     find_contraction(positions, &input_sets, &output_set);
///
/// assert_eq!(new_result, ['a', 'c'].into());
/// assert_eq!(remaining, vec![['a', 'c'].into()]);
/// assert_eq!(idx_removed, ['b'].into());
/// assert_eq!(idx_contraction, ['a', 'b', 'c'].into());
/// ```
pub fn find_contraction(
    positions: &[usize],
    input_sets: &[HashSet<char>],
    output_set: &HashSet<char>,
) -> (HashSet<char>, Vec<HashSet<char>>, HashSet<char>, HashSet<char>) {
    let mut remaining: Vec<HashSet<char>> = input_sets.to_vec();

    // Sort positions in reverse to avoid index shifting when removing elements
    let mut sorted_positions = positions.to_vec();
    sorted_positions.sort_by(|a, b| b.cmp(a));

    // Collect the input sets being contracted
    let contracted_sets = sorted_positions.iter().map(|&i| remaining.remove(i)).collect_vec();

    // Compute all indices involved in this contraction
    let idx_contraction = contracted_sets.iter().flatten().cloned().collect();

    // Compute remaining indices across output and unused input sets
    let mut idx_remain = output_set.clone();
    for set in &remaining {
        idx_remain.extend(set.iter().cloned());
    }

    // Indices that will remain after this contraction
    let new_result = idx_remain.intersection(&idx_contraction).cloned().collect();

    // Indices that are being summed over/removed
    let idx_removed = idx_contraction.difference(&new_result).cloned().collect();

    // Add the new result to remaining sets
    remaining.push(new_result.clone());

    (new_result, remaining, idx_removed, idx_contraction)
}

/// Computes the number of FLOPS required for a contraction.
///
/// # Parameters
/// - `idx_contraction`: Indices involved in the contraction
/// - `inner`: Whether this contraction requires an inner product
/// - `num_terms`: Number of terms in the contraction
/// - `size_dictionary`: Dictionary mapping indices to their sizes
///
/// # Returns
/// Total number of FLOPS
///
/// # Examples
/// ```
/// # use std::collections::HashMap;
/// # use opt_einsum_path::helper::flop_count;
/// let mut size_dict = HashMap::new();
/// size_dict.insert('a', 2);
/// size_dict.insert('b', 3);
/// size_dict.insert('c', 5);
///
/// assert_eq!(flop_count(&['a', 'b', 'c'], false, 1, &size_dict), 30);
/// assert_eq!(flop_count(&['a', 'b', 'c'], true, 2, &size_dict), 60);
/// ```
pub fn flop_count(
    idx_contraction: &[char],
    inner: bool,
    num_terms: usize,
    size_dictionary: &HashMap<char, usize>,
) -> usize {
    let overall_size = compute_size_by_char_indices(idx_contraction, size_dictionary);
    let mut op_factor = std::cmp::max(1, num_terms - 1);

    if inner {
        op_factor += 1;
    }

    overall_size * op_factor
}
