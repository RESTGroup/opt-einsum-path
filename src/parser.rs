use crate::*;

/// Get the symbol corresponding to int ``i`` - runs through the usual 52 letters before resorting
/// to unicode characters, starting at ``chr(192)`` and skipping surrogates.
///
/// # Examples
///
/// ```rust
/// # use opt_einsum_path::parser::get_symbol;
/// assert_eq!(get_symbol(2), 'c');
/// assert_eq!(get_symbol(200), 'Ŕ');
/// assert_eq!(get_symbol(20000), '京');
/// ```
pub fn get_symbol(i: u32) -> char {
    match i {
        0..26 => char::from(b'a' + i as u8),
        26..52 => char::from(b'A' + (i - 26) as u8),
        52..55296 => char::from_u32(i + 140).unwrap(),
        55296.. => char::from_u32(i + 2048).unwrap(),
    }
}

/// Generate `n` unused symbols not in `used`.
///
/// # Examples
///
/// ```rust
/// # use std::collections::BTreeSet;
/// # use opt_einsum_path::parser::gen_unused_symbols;
/// let unused = gen_unused_symbols("abd", 2);
/// assert_eq!(unused, "ce".to_string());
/// ```
pub fn gen_unused_symbols(used: &str, n: usize) -> String {
    let mut unused = Vec::with_capacity(n);
    let mut i = 0;
    while unused.len() < n {
        let c = get_symbol(i);
        if !used.contains(c) {
            unused.push(c);
        }
        i += 1;
    }
    unused.into_iter().collect()
}

/// Find the output string for the inputs `subscripts` under canonical Einstein summation rules.
/// That is, repeated indices are summed over by default.
///
/// # Examples
///
/// ```
/// # use opt_einsum_path::parser::find_output_str;
/// assert_eq!(find_output_str("ab,bc"), "ac");
/// assert_eq!(find_output_str("a,b"), "ab");
/// assert_eq!(find_output_str("a,a,b,b"), "");
/// ```
pub fn find_output_str(subscripts: &str) -> String {
    let tmp_subscripts: String = subscripts.split(',').collect();
    let mut unique_chars = BTreeSet::new();
    let mut repeated_chars = BTreeSet::new();

    // Track which characters appear exactly once
    for c in tmp_subscripts.chars() {
        if !unique_chars.insert(c) {
            repeated_chars.insert(c);
        }
    }

    // Collect characters that appear exactly once (not in repeated_chars)
    (&unique_chars - &repeated_chars).into_iter().collect()
}

/// Find the output shape for given inputs, shapes and output string, taking
/// into account broadcasting.
///
/// # Examples
///
/// ```
/// # use opt_einsum_path::parser::find_output_shape;
/// assert_eq!(find_output_shape(&["ab", "bc"], &[vec![2, 3], vec![3, 4]], "ac"), vec![2, 4]);
/// // Broadcasting is accounted for
/// assert_eq!(find_output_shape(&["a", "a"], &[vec![4], vec![1]], "a"), vec![4]);
/// ```
pub fn find_output_shape(inputs: &[&str], shapes: &[TensorShapeType], output: &str) -> TensorShapeType {
    output
        .chars()
        .map(|c| {
            shapes
                .iter()
                .zip(inputs.iter())
                .filter_map(|(shape, input)| input.chars().position(|x| x == c).map(|loc| shape.get(loc).copied()))
                .flatten()
                .max()
                .unwrap() // unwrap should be safe for valid einsum
        })
        .collect()
}

/// Convert user custom subscripts list to subscript string according to `symbol_map`.
///
/// # Examples
/// ```
/// # use std::collections::BTreeMap;
/// # use opt_einsum_path::parser::convert_subscripts;
/// let symbol_map = BTreeMap::from([("abc", "a"), ("def", "b")]);
/// assert_eq!(convert_subscripts(&["abc", "def"], &symbol_map), "ab");
/// // with ellipsis
/// let symbol_map = BTreeMap::from([("ast", "a")]);
/// assert_eq!(convert_subscripts(&["ast", "ast", "..."], &symbol_map), "aa...");
/// ```
pub fn convert_subscripts(old_sub: &[&str], symbol_map: &BTreeMap<&str, &str>) -> String {
    old_sub.iter().map(|&s| if s == "..." { "..." } else { symbol_map[s] }).collect()
}

/// A reproduction of einsum's parsing logic in Rust (originally from Python/Numpy).
///
/// # Arguments
///
/// * `subscripts` - The einsum subscript string (e.g., "ij,jk->ik").
/// * `shapes` - Array shapes or tensors to contract.
///
/// # Returns
///
/// - `input_subscripts` Parsed input subscripts (e.g., "ij,jk").
/// - `output_subscript` Parsed output subscript (e.g., "ik").
/// - `operands` The operands (shapes) to use in contraction.
///
/// # Examples
///
/// ```rust
/// # use opt_einsum_path::parser::parse_einsum_input;
/// let a_shape = vec![4, 4];
/// let b_shape = vec![4, 4, 4];
/// let subscripts = "...a, ...a -> ...";
/// let (input_subscripts, output_subscript, operands) = parse_einsum_input(subscripts, &[a_shape, b_shape]).unwrap();
/// assert_eq!(input_subscripts, "da,cda");
/// assert_eq!(output_subscript, "cd");
/// assert_eq!(operands, vec![vec![4, 4], vec![4, 4, 4]]);
/// ```
///
/// # Errors
///
/// - Subscripts contain invalid '->' patterns (larger than one '->', or contains separated `-` or
///   `>` characters).
/// - Malformed ellipses ("...").
/// - Output characters are non-unique or missing from inputs.
/// - Operand count doesn't match subscript terms.
pub fn parse_einsum_input(
    subscripts: &str,
    operands: &[TensorShapeType],
) -> Result<(String, String, Vec<TensorShapeType>), String> {
    // remove whitespace
    let einsum_str = subscripts.split_whitespace().collect::<String>();

    // Check for proper "->"
    if (einsum_str.contains('-') || einsum_str.contains('>'))
        && (einsum_str.matches('-').count() > 1
            || einsum_str.matches('>').count() > 1
            || einsum_str.matches("->").count() != 1)
    {
        return Err("Subscripts can only contain one '->'.".to_string());
    }

    // Parse ellipses
    let mut subscripts = subscripts.to_string();
    if einsum_str.contains('.') {
        let used: String = einsum_str.replace(".", "").replace(",", "").replace("->", "");
        let ellipse_inds = gen_unused_symbols(&used, operands.iter().map(|s| s.len()).max().unwrap_or(0));
        let mut longest = 0;

        // Do we have an output to account for?
        let (split_subscripts, out_sub) = if einsum_str.contains("->") {
            let parts = einsum_str.split("->").collect_vec();
            let input_tmp = parts[0];
            (input_tmp.split(',').collect_vec(), true)
        } else {
            (einsum_str.split(',').collect_vec(), false)
        };

        let mut processed_subscripts = Vec::new();
        for (num, sub) in split_subscripts.iter().enumerate() {
            if sub.contains('.') {
                if sub.matches('.').count() != 3 || !sub.contains("...") {
                    return Err("Invalid Ellipses.".to_string());
                }

                // Take into account numerical values
                let ellipse_count = if operands[num].is_empty() {
                    0
                } else if operands[num].len() >= sub.len() - 3 {
                    operands[num].len() - (sub.len() - 3)
                } else {
                    return Err("Ellipses lengths do not match.".to_string());
                };
                longest = longest.max(ellipse_count);

                if ellipse_count == 0 {
                    processed_subscripts.push(sub.replace("...", ""));
                } else {
                    let replacement = &ellipse_inds[ellipse_inds.len() - ellipse_count..];
                    processed_subscripts.push(sub.replace("...", replacement));
                }
            } else {
                processed_subscripts.push(sub.to_string());
            }
        }

        subscripts = processed_subscripts.join(",");

        // Figure out output ellipses
        let out_ellipse = if longest == 0 { "" } else { &ellipse_inds[ellipse_inds.len() - longest..] };

        subscripts = if out_sub {
            let parts: Vec<&str> = einsum_str.split("->").collect();
            let output_sub = parts[1];
            format!("{subscripts}->{}", output_sub.replace("...", out_ellipse))
        } else {
            // Special care for outputless ellipses
            let output_subscript = find_output_str(&subscripts);
            let normal_inds: String = output_subscript.chars().filter(|c| !out_ellipse.contains(*c)).collect();
            format!("{subscripts}->{out_ellipse}{normal_inds}")
        };
    }

    // Build output string if does not exist
    let (input_subscripts, output_subscript) = if subscripts.contains("->") {
        let parts: Vec<&str> = subscripts.split("->").collect();
        (parts[0].to_string(), parts[1].to_string())
    } else {
        (subscripts.clone(), find_output_str(&subscripts))
    };

    // Make sure output subscripts are unique and in the input
    for char in output_subscript.chars() {
        if output_subscript.matches(char).count() != 1 {
            return Err(format!("Output character '{char}' appeared more than once in the output."));
        }
        if !input_subscripts.contains(char) {
            return Err(format!("Output character '{char}' did not appear in the input"));
        }
    }

    // Make sure number operands is equivalent to the number of terms
    let input_count = input_subscripts.split(',').count();
    if input_count != operands.len() {
        return Err(format!(
            "Number of einsum subscripts, {input_count}, must be equal to the number of operands, {}.",
            operands.len()
        ));
    }

    Ok((input_subscripts, output_subscript, operands.to_vec()))
}

#[test]
fn playground() {
    for i in 0..500 {
        println!("{} -> '{:?}'", i, char::from_u32(i));
    }
}
