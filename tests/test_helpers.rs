use itertools::Itertools;
use opt_einsum_path::helpers::*;
use std::collections::BTreeMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_by_dict() {
        let size_dict = BTreeMap::from([('a', 2), ('b', 5), ('c', 9), ('d', 11), ('e', 13), ('z', 0)]);
        let cases = [("", 1), ("a", 2), ("b", 5), ("z", 0), ("az", 0), ("zbc", 0), ("aaae", 104), ("abcde", 12870)];
        for (s, expected) in cases {
            let size = compute_size_by_dict(s.chars().collect_vec().iter(), &size_dict);
            assert_eq!(size, expected as f64);
        }
    }

    #[test]
    fn test_flop_cost() {
        let size_dict = BTreeMap::from([('a', 10), ('b', 10), ('c', 10), ('d', 10), ('e', 10), ('f', 10)]);
        let cases = [
            ("a", false, 1, 10),
            ("a", false, 2, 10),
            ("ab", false, 2, 100),
            ("a", true, 2, 20),
            ("ab", true, 2, 200),
            ("a", true, 3, 30),
            ("abc", true, 2, 2000),
        ];
        for (s, inner, num_terms, expected) in cases {
            let flop_cost = flop_count(s.chars().collect_vec().iter(), inner, num_terms, &size_dict);
            assert_eq!(flop_cost, expected as f64);
        }
    }
}
