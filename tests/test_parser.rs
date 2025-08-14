use opt_einsum_path::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_symbol() {
        assert_eq!(parser::get_symbol(2), 'c');
        assert_eq!(parser::get_symbol(200000), '\u{31540}');
        assert_eq!(parser::get_symbol(55296), '\u{e000}');
        assert_eq!(parser::get_symbol(57343), '\u{e7ff}');
    }

    #[test]
    #[should_panic]
    fn test_get_symbol_panic() {
        // This should panic because it maps to a surrogate
        parser::get_symbol(55295);
    }

    #[test]
    fn test_parse_einsum_input() {
        let eq = "ab,bc,cd";
        let shapes = [vec![2, 3], vec![3, 4], vec![4, 5]];
        let (input_subscripts, output_subscript, operands) = parser::parse_einsum_input(eq, &shapes).unwrap();
        assert_eq!(input_subscripts, eq);
        assert_eq!(output_subscript, "ad");
        assert_eq!(operands, shapes);
    }

    #[test]
    fn test_parse_with_ellisis() {
        let eq = "...a,ab";
        let shapes = [vec![2, 3], vec![3, 4]];
        let (input_subscripts, output_subscript, operands) = parser::parse_einsum_input(eq, &shapes).unwrap();
        assert_eq!(input_subscripts, "da,ab");
        assert_eq!(output_subscript, "db");
        assert_eq!(operands, shapes);
    }
}
