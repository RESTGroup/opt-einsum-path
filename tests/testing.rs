use opt_einsum_path::parser::get_symbol;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::{BTreeMap, BTreeSet};

// Type definitions matching the provided specifications
pub type SizeType = f64;
pub type TensorShapeType = Vec<usize>;
pub type PathType = Vec<TensorShapeType>;
pub type ArrayIndexType = BTreeSet<char>;
pub type SizeDictType = BTreeMap<char, usize>;

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

/// Builds random arrays (views) from a string specification
pub fn build_views<F>(
    string: &str,
    dimension_dict: Option<&SizeDictType>,
    array_function: Option<F>,
    replace_ellipsis: bool,
) -> Result<Vec<Vec<SizeType>>, String>
where
    F: Fn(&[usize]) -> Vec<SizeType>,
{
    let shapes = build_shapes(string, dimension_dict, replace_ellipsis)?;

    // Default array generator: random f64 values
    fn default_generator(shape: &[usize]) -> Vec<SizeType> {
        let len: usize = shape.iter().product();
        let mut rng = rand::rng();
        (0..len).map(|_| rng.random()).collect()
    }

    let mut views = Vec::with_capacity(shapes.len());
    for shape in shapes {
        if shape.is_empty() {
            // Handle scalars (0-dimensional arrays)
            let mut rng = rand::rng();
            views.push(vec![rng.random()]);
        } else {
            match &array_function {
                Some(f) => views.push(f(&shape)),
                None => views.push(default_generator(&shape)),
            }
        }
    }

    Ok(views)
}

/// Generates a random contraction equation and corresponding shapes
pub fn rand_equation<R>(
    n: usize,
    regularity: usize,
    n_out: usize,
    d_min: usize,
    d_max: usize,
    mut rng: R,
    global_dim: bool,
    return_size_dict: bool,
) -> Result<(String, PathType, Option<SizeDictType>), String>
where
    R: Rng,
{
    if d_min > d_max {
        return Err("d_min must be less than or equal to d_max".to_string());
    }
    if n == 0 {
        return Err("Number of arrays (n) must be greater than 0".to_string());
    }

    // Initialize random number generator

    // Calculate total number of indices
    let num_inds = n * regularity / 2 + n_out;

    // Create size dictionary for indices
    let mut size_dict = SizeDictType::new();
    for i in 0..num_inds {
        let c = get_symbol(i as u32); // Assume get_symbol is safe
        let size = rng.random_range(d_min..=d_max);
        size_dict.insert(c, size);
    }

    // Generate index sequence: output indices (n_out) followed by bond indices (each twice)
    let mut indices = Vec::new();
    // Add output indices
    for i in 0..n_out {
        let c = get_symbol(i as u32);
        indices.push(c);
    }
    // Add bond indices (each appears twice)
    for i in n_out..num_inds {
        let c = get_symbol(i as u32);
        indices.push(c);
        indices.push(c);
    }

    // Randomly permute indices
    indices.shuffle(&mut rng);

    // Initialize input terms with at least one index each
    let mut inputs: Vec<Vec<char>> = (0..n).map(|_| Vec::new()).collect();
    for i in 0..n {
        if let Some(&c) = indices.get(i) {
            inputs[i].push(c);
        } else {
            return Err(format!("Not enough indices to initialize {n} input terms"));
        }
    }

    // Distribute remaining indices, avoiding duplicates in the same term
    for &c in indices.iter().skip(n) {
        let mut where_idx = rng.random_range(0..n);
        while inputs[where_idx].contains(&c) {
            where_idx = rng.random_range(0..n);
        }
        inputs[where_idx].push(c);
    }

    // Add global dimension if requested
    let global_char = if global_dim {
        let gdim = get_symbol(num_inds as u32);
        size_dict.insert(gdim, rng.random_range(d_min..=d_max));
        // Add to all inputs
        for input in &mut inputs {
            input.push(gdim);
        }
        Some(gdim)
    } else {
        None
    };

    // Build and shuffle output indices
    let mut output: Vec<char> = (0..n_out).map(|i| get_symbol(i as u32)).collect();
    if let Some(gdim) = global_char {
        output.push(gdim);
    }
    output.shuffle(&mut rng);

    // Construct equation string
    let inputs_str: Vec<String> = inputs.iter().map(|chars| chars.iter().collect()).collect();
    let output_str: String = output.iter().collect();
    let eq = format!("{}->{}", inputs_str.join(","), output_str);

    // Generate shapes for each input term
    let shapes: PathType = inputs.iter().map(|chars| chars.iter().map(|c| size_dict[c]).collect()).collect();

    let size_dict = if return_size_dict { Some(size_dict) } else { None };

    Ok((eq, shapes, size_dict))
}

/// Builds random arrays from a path of shapes
pub fn build_arrays_from_tuples(path: &PathType) -> Vec<Vec<SizeType>> {
    let mut rng = rand::rng();
    path.iter()
        .map(|shape| {
            let len: usize = shape.iter().product();
            (0..len).map(|_| rng.random()).collect()
        })
        .collect()
}
