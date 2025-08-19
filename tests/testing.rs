use opt_einsum_path::typing::*;

// Constants from original Python code
const _NO_COLLISION_CHARS: &str = "\u{1B58}\u{1B59}\u{1B5A}\u{1B5B}\u{1B5C}\u{1B5D}\u{1B5E}";
const _VALID_CHARS: &str = "abcdefghijklmnopqABC\u{1B58}\u{1B59}\u{1B5A}\u{1B5B}\u{1B5C}\u{1B5D}\u{1B5E}";
const _SIZES: &[usize] = &[2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4, 9, 10, 2, 4, 5, 3, 2, 6];

/// Creates the default dimension dictionary mapping characters to sizes
fn default_dim_dict() -> SizeDictType {
    _VALID_CHARS.chars().zip(_SIZES.iter().copied()).collect()
}

/// Builds tensor shapes from a string specification
pub fn build_shapes(
    string: &str,
    dimension_dict: Option<&SizeDictType>,
    replace_ellipsis: bool,
) -> Result<Vec<TensorShapeType>, String> {
    let dim_dict = dimension_dict.cloned().unwrap_or(default_dim_dict());

    // Handle ellipsis
    if string.contains("...") && !replace_ellipsis {
        return Err("Ellipsis found in string but `replace_ellipsis` is false".to_string());
    }

    let ellipse_replace = &_NO_COLLISION_CHARS[0..3];
    let processed_string = string.replace("...", ellipse_replace);

    // Parse equation components
    let terms_part = processed_string.split("->").next().ok_or("Invalid equation format (missing '->' or empty)")?;
    let terms: Vec<&str> = terms_part.split(',').collect();

    // Generate shapes for each term
    let mut shapes = Vec::with_capacity(terms.len());
    for term in terms {
        let mut shape = TensorShapeType::with_capacity(term.len());
        for c in term.chars() {
            let size = dim_dict.get(&c).ok_or_else(|| format!("Dimension '{c}' not found in dictionary"))?;
            shape.push(*size);
        }
        shapes.push(shape);
    }

    Ok(shapes)
}
