//! Determines if a contraction can use BLAS or not.

use crate::*;

/// Checks if we can use a BLAS call for the given contraction pattern.
///
/// # Arguments
/// * `inputs` - Specifies the subscripts for summation (e.g., ["ij", "jk"])
/// * `result` - Resulting summation subscript (e.g., "ik")
/// * `idx_removed` - Indices that are removed in the summation
/// * `shapes` - If given, checks that none of the indices are broadcast dimensions
///
/// # Returns
/// Option<&'static str> where:
/// - Some(blas_type): The type of BLAS call that can be used
/// - None: No suitable BLAS call available
///
/// # Notes
/// - We assume several operations are not efficient such as a transposed DDOT
/// - Returns blas type appended with "/EINSUM" when they can still be done with tensordot
///
/// # Examples
/// ```
/// # use opt_einsum_path::blas::can_blas;
/// # use std::collections::BTreeSet;
/// let idx = BTreeSet::from(['j']);
/// assert_eq!(can_blas(&["ij", "jk"], "ik", &idx, None), Some("GEMM"));
/// ```
pub fn can_blas(
    inputs: &[&str],
    result: &str,
    idx_removed: &ArrayIndexType,
    shapes: Option<&[TensorShapeType]>,
) -> Option<&'static str> {
    // Can only do two inputs
    if inputs.len() != 2 {
        return None;
    }

    let input_left = inputs[0];
    let input_right = inputs[1];

    // Check for invalid index patterns
    for c in input_left.chars().chain(input_right.chars()).collect::<BTreeSet<_>>() {
        let nl = input_left.chars().filter(|&x| x == c).count();
        let nr = input_right.chars().filter(|&x| x == c).count();

        // Can't deal with repeated indices on same input or more than 2 total
        if (nl > 1) || (nr > 1) || (nl + nr > 2) {
            return None;
        }

        // Can't do implicit summation or dimension collapse
        let in_result = result.contains(c);
        if nl + nr - 1 == in_result as usize {
            return None;
        }
    }

    // Check for broadcast indices
    if let Some(shapes) = shapes {
        for c in idx_removed {
            let left_pos = input_left.chars().position(|x| x == *c).unwrap();
            let right_pos = input_right.chars().position(|x| x == *c).unwrap();

            if shapes[0][left_pos] != shapes[1][right_pos] {
                return None;
            }
        }
    }

    // Prefer einsum if not removing indices
    if idx_removed.is_empty() {
        return Some("OUTER/EINSUM");
    }

    // Build temporaries
    let left_set = input_left.chars().collect();
    let right_set = input_right.chars().collect();
    let right_vec = input_right.chars().collect_vec();
    let keep_left = &left_set - idx_removed;
    let keep_right = &right_set - idx_removed;
    let rs = idx_removed.len();

    // DDOT cases
    if input_left == input_right {
        return Some("DOT");
    } else if left_set == right_set {
        return Some("DOT/EINSUM");
    }

    // GEMM cases
    if input_left.ends_with(&right_vec[..rs])
        || input_left.starts_with(&right_vec[right_vec.len() - rs..])
        || input_left.ends_with(&right_vec[right_vec.len() - rs..])
        || input_left.starts_with(&right_vec[..rs])
    {
        return Some("GEMM");
    }

    // GEMV/EINSUM case
    if keep_left.is_empty() || keep_right.is_empty() {
        return Some("GEMV/EINSUM");
    }

    // Conventional tensordot
    Some("TDOT")
}
