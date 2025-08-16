use itertools::Itertools;
use num::{FromPrimitive, Zero};
use opt_einsum_path::paths::*;
use opt_einsum_path::typing::*;
use std::collections::BTreeMap;

#[cfg(test)]
mod tests {

    use super::*;

    fn prepare_explicit_path_tests(case: &str) -> (Vec<&str>, &str, SizeDictType) {
        match case {
            "GEMM1" => (vec!["abd", "ac", "bdc"], "", BTreeMap::from([('a', 1), ('b', 2), ('c', 3), ('d', 4)])),
            "Inner1" => (vec!["abcd", "abc", "bc"], "", BTreeMap::from([('a', 5), ('b', 2), ('c', 3), ('d', 4)])),
            _ => unreachable!("Unknown test case"),
        }
    }

    #[test]
    fn test_path_optimal() {
        let (inputs, output, size_dict) = prepare_explicit_path_tests("GEMM1");
        let inputs = inputs.iter().map(|s| s.chars().collect()).collect_vec();
        let inputs_ref = inputs.iter().collect_vec();
        let path = optimal::optimal(
            &inputs_ref,
            &output.chars().collect(),
            &size_dict,
            Some(SizeType::from_usize(5000).unwrap()),
        );
        assert_eq!(path, vec![vec![0, 2], vec![0, 1]]);
        let path = optimal::optimal(&inputs_ref, &output.chars().collect(), &size_dict, Some(SizeType::zero()));
        assert_eq!(path, vec![vec![0, 1, 2]]);
    }
}
