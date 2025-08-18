use itertools::Itertools;
use num::{FromPrimitive, Zero};
use opt_einsum_path::contract;
use opt_einsum_path::paths::*;
use opt_einsum_path::typing::*;
use std::collections::BTreeMap;
mod testing;

#[cfg(test)]
mod tests {
    use opt_einsum_path::contract::contract_path;

    use crate::testing::build_shapes;

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

    #[test]
    #[should_panic(expected = "Unknown optimization kind: optimall")]
    fn test_bad_path_option() {
        contract::contract_path("a,b,c", &[vec![1], vec![2], vec![3]], true, "optimall", None).unwrap();
    }

    #[test]
    fn test_memory_paths() {
        let expression = "abc,bdef,fghj,cem,mhk,ljk->adgl";
        let views = build_shapes(expression, None, true).unwrap();

        // Test tiny memory limit
        let path_ret = contract_path(expression, &views, true, "optimal", Some(5 as _)).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3, 4, 5]]);

        // Check the possibilities, greedy is capped
        let path_ret = contract_path(expression, &views, true, "greedy", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 3], vec![0, 4], vec![0, 2], vec![0, 2], vec![0, 1]]);

        let path_ret = contract_path(expression, &views, true, "greedy", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 3], vec![0, 4], vec![0, 2], vec![0, 2], vec![0, 1]]);
    }
}
