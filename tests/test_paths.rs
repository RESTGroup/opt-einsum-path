use crate::testing::build_shapes;
use opt_einsum_path::contract::contract_path;
use opt_einsum_path::typing::*;
use rstest::rstest;
use std::collections::BTreeMap;
mod testing;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    #[should_panic(expected = "Unknown optimization kind: optimall")]
    fn test_bad_path_option() {
        contract_path("a,b,c", &[vec![1], vec![2], vec![3]], true, "optimall", None).unwrap();
    }

    #[rstest]
    #[case("optimal", "abd,ac,bdc->", Some(5000 as _), vec![vec![0, 2], vec![0, 1]])]
    #[case("optimal", "abd,ac,bdc->", Some(0 as _), vec![vec![0, 1, 2]])]
    fn test_path_optimal(
        #[case] name: &str,
        #[case] expr: &str,
        #[case] memory_limit: Option<SizeType>,
        #[case] expected: PathType,
    ) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, true, name, memory_limit).unwrap();
        assert_eq!(path_ret.0, expected);
    }

    #[rstest]
    #[case("greedy", "abd,ac,bdc->", Some(5000 as _), vec![vec![0, 2], vec![0, 1]])]
    #[case("greedy", "abd,ac,bdc->", Some(0 as _), vec![vec![0, 1, 2]])]
    fn test_path_greedy(
        #[case] name: &str,
        #[case] expr: &str,
        #[case] memory_limit: Option<SizeType>,
        #[case] expected: PathType,
    ) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, true, name, memory_limit).unwrap();
        assert_eq!(path_ret.0, expected);
    }

    #[test]
    fn test_memory_paths() {
        let expression = "abc,bdef,fghj,cem,mhk,ljk->adgl";
        let views = build_shapes(expression, None, true).unwrap();

        // Test tiny memory limit
        let path_ret = contract_path(expression, &views, true, "optimal", Some(5 as _)).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3, 4, 5]]);

        let path_ret = contract_path(expression, &views, true, "greedy", Some(5 as _)).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3, 4, 5]]);

        // Check the possibilities, greedy is capped
        let path_ret = contract_path(expression, &views, true, "greedy", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 3], vec![0, 4], vec![0, 2], vec![0, 2], vec![0, 1]]);

        let path_ret = contract_path(expression, &views, true, "greedy", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 3], vec![0, 4], vec![0, 2], vec![0, 2], vec![0, 1]]);
    }

    #[rstest]
    #[case("greedy", "eb,cb,fb->cef", vec![vec![0, 2], vec![0, 1]])]
    #[case("branch-all", "eb,cb,fb->cef", vec![vec![0, 2], vec![0, 1]])]
    #[case("branch-2", "eb,cb,fb->cef", vec![vec![0, 2], vec![0, 1]])]
    #[case("optimal", "eb,cb,fb->cef", vec![vec![0, 2], vec![0, 1]])]
    // #[case("dp", "eb,cb,fb->cef", vec![vec![1, 2], vec![0, 1]])]
    #[case("greedy", "dd,fb,be,cdb->cef", vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("branch-all", "dd,fb,be,cdb->cef", vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("branch-2", "dd,fb,be,cdb->cef", vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("optimal", "dd,fb,be,cdb->cef", vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    // #[case("dp", "dd,fb,be,cdb->cef", vec![vec![0, 3], vec![0, 2], vec![0, 1]])]
    #[case("greedy", "bca,cdb,dbf,afc->", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("branch-all", "bca,cdb,dbf,afc->", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("branch-2", "bca,cdb,dbf,afc->", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("optimal", "bca,cdb,dbf,afc->", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    // #[case("dp", "bca,cdb,dbf,afc->", vec![vec![1, 2], vec![1, 2], vec![0, 1]])]
    #[case("greedy", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 1], vec![0, 1]])]
    #[case("branch-all", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("branch-2", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("optimal", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    // #[case("dp", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    fn test_path_edge_cases(#[case] name: &str, #[case] expr: &str, #[case] expected: PathType) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, true, name, None).unwrap();
        assert_eq!(path_ret.0, expected);
    }

    #[rstest]
    #[case("optimal", "a,->a", 1)]
    #[case("optimal", "ab->ab", 1)]
    #[case("optimal", ",a,->a", 2)]
    #[case("optimal", ",,a,->a", 3)]
    #[case("optimal", ",,->", 2)]
    fn test_path_scalar_cases(#[case] name: &str, #[case] expr: &str, #[case] expected: usize) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, true, name, None).unwrap();
        assert_eq!(path_ret.0.len(), expected);
    }

    #[test]
    fn test_optimal_edge_cases() {
        let expression = "a,ac,ab,ad,cd,bd,bc->";
        let size_dict = BTreeMap::from([('a', 20), ('b', 20), ('c', 20), ('d', 20)]);
        let edge_test4 = build_shapes(expression, Some(&size_dict), true).unwrap();
        let path_ret = contract_path(expression, &edge_test4, true, "greedy", "max-input").unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1], vec![0, 1, 2, 3, 4, 5]]);
        let path_ret = contract_path(expression, &edge_test4, true, "optimal", "max-input").unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1], vec![0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn test_greedy_edge_cases() {
        let expression = "abc,cfd,dbe,efa";
        let size_dict = BTreeMap::from([('a', 20), ('b', 20), ('c', 20), ('d', 20), ('e', 20), ('f', 20)]);
        let tensors = build_shapes(expression, Some(&size_dict), true).unwrap();
        let path_ret = contract_path(expression, &tensors, true, "greedy", "max-input").unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3]]);
        let path_ret = contract_path(expression, &tensors, true, "optimal", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1], vec![0, 2], vec![0, 1]]);
    }
}
