use crate::testing::build_shapes;
use opt_einsum_path::contract::contract_path;
use opt_einsum_path::typing::*;
use rstest::rstest;
use std::collections::BTreeMap;
mod testing;
use itertools::Itertools;

/*
- [x] test_size_by_dict
- [x] test_flop_cost
- [x] test_bad_path_option
- [~] test_explicit_path
- [x] test_path_optimal
- [x] test_path_greedy
- [x] test_memory_paths
- [x] test_path_edge_cases
- [x] test_path_scalar_cases
- [x] test_optimal_edge_cases
- [x] test_greedy_edge_cases
- [ ] test_dp_edge_cases_dimension_1
- [ ] test_dp_edge_cases_all_singlet_indices
- [ ] test_custom_dp_can_optimize_for_outer_products
- [ ] test_custom_dp_can_optimize_for_size
- [ ] test_custom_dp_can_set_cost_cap
- [ ] test_custom_dp_can_set_minimize
- [ ] test_dp_errors_when_no_contractions_found
- [ ] test_can_optimize_outer_products
- [ ] test_large_path
- [ ] test_custom_random_greedy
- [ ] test_custom_branchbound
- [ ] test_branchbound_validation
- [ ] test_parallel_random_greedy
- [ ] test_custom_path_optimizer
- [ ] test_custom_random_optimizer
- [ ] test_optimizer_registration
- [ ] test_path_with_assumed_shapes
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "Unknown optimization kind: optimall")]
    fn test_bad_path_option() {
        contract_path("a,b,c", &[vec![1], vec![2], vec![3]], "optimall", None).unwrap();
    }

    #[rstest]
    #[case("optimal", "abd,ac,bdc->", Some(5000 as _), vec![vec![0, 2], vec![0, 1]])]
    #[case("optimal", "abd,ac,bdc->", Some(0 as _)   , vec![vec![0, 1, 2]])         ]
    fn test_path_optimal(
        #[case] name: &str,
        #[case] expr: &str,
        #[case] memory_limit: Option<SizeType>,
        #[case] expected: PathType,
    ) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, name, memory_limit).unwrap();
        assert_eq!(path_ret.0, expected);
    }

    #[rstest]
    #[case("greedy", "abd,ac,bdc->", Some(5000 as _), vec![vec![0, 2], vec![0, 1]])]
    #[case("greedy", "abd,ac,bdc->", Some(0 as _)   , vec![vec![0, 1, 2]])         ]
    fn test_path_greedy(
        #[case] name: &str,
        #[case] expr: &str,
        #[case] memory_limit: Option<SizeType>,
        #[case] expected: PathType,
    ) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, name, memory_limit).unwrap();
        assert_eq!(path_ret.0, expected);
    }

    #[test]
    fn test_memory_paths() {
        let expression = "abc,bdef,fghj,cem,mhk,ljk->adgl";
        let views = build_shapes(expression, None, true).unwrap();

        // Test tiny memory limit
        let path_ret = contract_path(expression, &views, "optimal", Some(5 as _)).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3, 4, 5]]);

        let path_ret = contract_path(expression, &views, "greedy", Some(5 as _)).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3, 4, 5]]);

        // Check the possibilities, greedy is capped
        let path_ret = contract_path(expression, &views, "greedy", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 3], vec![0, 4], vec![0, 2], vec![0, 2], vec![0, 1]]);

        let path_ret = contract_path(expression, &views, "greedy", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 3], vec![0, 4], vec![0, 2], vec![0, 2], vec![0, 1]]);
    }

    #[rstest]
    #[case("greedy"    , "eb,cb,fb->cef"     , vec![vec![0, 2], vec![0, 1]])            ]
    #[case("branch-all", "eb,cb,fb->cef"     , vec![vec![0, 2], vec![0, 1]])            ]
    #[case("branch-2"  , "eb,cb,fb->cef"     , vec![vec![0, 2], vec![0, 1]])            ]
    #[case("optimal"   , "eb,cb,fb->cef"     , vec![vec![0, 2], vec![0, 1]])            ]
    #[case("dp"        , "eb,cb,fb->cef"     , vec![vec![1, 2], vec![0, 1]])            ]
    #[case("greedy"    , "dd,fb,be,cdb->cef" , vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("branch-all", "dd,fb,be,cdb->cef" , vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("branch-2"  , "dd,fb,be,cdb->cef" , vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("optimal"   , "dd,fb,be,cdb->cef" , vec![vec![0, 3], vec![0, 1], vec![0, 1]])]
    #[case("dp"        , "dd,fb,be,cdb->cef" , vec![vec![0, 3], vec![0, 2], vec![0, 1]])]
    #[case("greedy"    , "bca,cdb,dbf,afc->" , vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("branch-all", "bca,cdb,dbf,afc->" , vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("branch-2"  , "bca,cdb,dbf,afc->" , vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("optimal"   , "bca,cdb,dbf,afc->" , vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("dp"        , "bca,cdb,dbf,afc->" , vec![vec![1, 2], vec![1, 2], vec![0, 1]])]
    #[case("greedy"    , "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 1], vec![0, 1]])]
    #[case("branch-all", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("branch-2"  , "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    #[case("optimal"   , "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    // #[case("dp", "dcc,fce,ea,dbf->ab", vec![vec![1, 2], vec![0, 2], vec![0, 1]])]
    fn test_path_edge_cases(#[case] name: &str, #[case] expr: &str, #[case] expected: PathType) {
        let views = build_shapes(expr, None, true).unwrap();
        let path_ret = contract_path(expr, &views, name, None).unwrap();
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
        let path_ret = contract_path(expr, &views, name, None).unwrap();
        assert_eq!(path_ret.0.len(), expected);
    }

    #[test]
    fn test_optimal_edge_cases() {
        let expression = "a,ac,ab,ad,cd,bd,bc->";
        let size_dict = BTreeMap::from([('a', 20), ('b', 20), ('c', 20), ('d', 20)]);
        let edge_test4 = build_shapes(expression, Some(&size_dict), true).unwrap();
        let path_ret = contract_path(expression, &edge_test4, "greedy", "max-input").unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1], vec![0, 1, 2, 3, 4, 5]]);
        let path_ret = contract_path(expression, &edge_test4, "optimal", "max-input").unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1], vec![0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn test_greedy_edge_cases() {
        let expression = "abc,cfd,dbe,efa";
        let size_dict = BTreeMap::from([('a', 20), ('b', 20), ('c', 20), ('d', 20), ('e', 20), ('f', 20)]);
        let tensors = build_shapes(expression, Some(&size_dict), true).unwrap();
        let path_ret = contract_path(expression, &tensors, "greedy", "max-input").unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1, 2, 3]]);
        let path_ret = contract_path(expression, &tensors, "optimal", None).unwrap();
        assert_eq!(path_ret.0, vec![vec![0, 1], vec![0, 2], vec![0, 1]]);
    }

    #[test]
    fn test_dp_edge_cases_dimension_1() {
        let expression = "nlp,nlq,pl->n";
        let shapes = vec![
            vec![1, 1, 1], // nlp
            vec![1, 1, 1], // nlq
            vec![1, 1],    // pl
        ];
        let (_, info) = contract_path(expression, &shapes, "dp", None).unwrap();
        assert_eq!(info.scale_list.iter().max(), Some(&3));
    }

    #[test]
    fn test_dp_edge_cases_all_singlet_indices() {
        let expression = "a,bcd,efg->";
        let shapes = vec![
            vec![2],       // a
            vec![2, 2, 2], // bcd
            vec![2, 2, 2], // efg
        ];
        let (_, info) = contract_path(expression, &shapes, "dp", None).unwrap();
        println!("{info}");
        assert_eq!(info.scale_list.iter().max(), Some(&3));
    }

    #[test]
    fn test_custom_dp_can_optimize_for_outer_products() {
        use opt_einsum_path::paths::dp::DynamicProgramming;

        let expression = "a,b,abc->c";
        let shapes = vec![
            vec![2],       // a
            vec![2],       // b
            vec![2, 2, 3], // abc
        ];
        let optimizer1 = DynamicProgramming { search_outer: false, ..Default::default() };
        let optimizer2 = DynamicProgramming { search_outer: true, ..Default::default() };
        let (_, info1) = contract_path(expression, &shapes, optimizer1, None).unwrap();
        let (_, info2) = contract_path(expression, &shapes, optimizer2, None).unwrap();
        assert!(info2.opt_cost < info1.opt_cost);
        assert_eq!(info1.opt_cost, 36 as _);
        assert_eq!(info2.opt_cost, 28 as _);
    }

    #[test]
    fn test_custom_dp_can_optimize_for_size() {
        let expression = "qgcf,sotr,klb,jlretia,hpn,nseha,jgoqm,ipkb,cdfm,d->";
        let shapes = vec![
            vec![5, 2, 9, 4],
            vec![4, 9, 5, 9],
            vec![5, 4, 2],
            vec![5, 4, 9, 7, 5, 3, 6],
            vec![5, 2, 8],
            vec![8, 4, 7, 5, 6],
            vec![5, 2, 9, 5, 8],
            vec![3, 2, 5, 2],
            vec![9, 3, 4, 8],
            vec![3],
        ];

        let (_, info1) = contract_path(expression, &shapes, "dp-flops", None).unwrap();
        let (_, info2) = contract_path(expression, &shapes, "dp-size", None).unwrap();
        assert!(info1.opt_cost < info2.opt_cost);
        assert!(info1.largest_intermediate > info2.largest_intermediate);

        assert_eq!(info1.opt_cost, 663054 as _);
        assert_eq!(info2.opt_cost, 1114440 as _);
        assert_eq!(info1.largest_intermediate, 18900 as _);
        assert_eq!(info2.largest_intermediate, 2016 as _);
        assert_eq!(info1.path, vec![
            vec![4, 5],
            vec![2, 5],
            vec![2, 7],
            vec![5, 6],
            vec![1, 5],
            vec![1, 4],
            vec![0, 3],
            vec![0, 2],
            vec![0, 1]
        ]);
        assert_eq!(info2.path, vec![
            vec![2, 7],
            vec![3, 8],
            vec![3, 7],
            vec![2, 6],
            vec![1, 5],
            vec![1, 4],
            vec![1, 3],
            vec![1, 2],
            vec![0, 1]
        ]);
    }

    #[test]
    fn test_custom_dp_can_set_cost_cap() {
        use opt_einsum_path::paths::dp::DynamicProgramming;

        let expression = "ad,cfb,fdc,abge,eg->";
        let shapes = vec![vec![8, 8], vec![6, 9, 5], vec![9, 8, 6], vec![8, 5, 6, 4], vec![4, 6]];

        let opt1 = DynamicProgramming { cost_cap: true.into(), ..Default::default() };
        let opt2 = DynamicProgramming { cost_cap: false.into(), ..Default::default() };
        let opt3 = DynamicProgramming { cost_cap: Some(100 as _).into(), ..Default::default() };
        let (_, info1) = contract_path(expression, &shapes, opt1, None).unwrap();
        let (_, info2) = contract_path(expression, &shapes, opt2, None).unwrap();
        let (_, info3) = contract_path(expression, &shapes, opt3, None).unwrap();
        assert_eq!(info1.opt_cost, info2.opt_cost);
        assert_eq!(info1.opt_cost, info3.opt_cost);
        assert_eq!(info1.path, vec![vec![1, 2], vec![0, 3], vec![0, 2], vec![0, 1]]);
        assert_eq!(info1.path, info2.path);
        assert_eq!(info1.path, info3.path);
    }

    #[rstest]
    #[case("dp-flops"    ,  663054, 18900, vec![(4, 5), (2, 5), (2, 7), (5, 6), (1, 5), (1, 4), (0, 3), (0, 2), (0, 1)])]
    #[case("dp-size"     , 1114440,  2016, vec![(2, 7), (3, 8), (3, 7), (2, 6), (1, 5), (1, 4), (1, 3), (1, 2), (0, 1)])]
    #[case("dp-write"    ,  983790,  2016, vec![(0, 8), (3, 4), (1, 4), (5, 6), (1, 5), (0, 4), (0, 3), (1, 2), (0, 1)])]
    #[case("dp-combo"    ,  973518,  2016, vec![(4, 5), (2, 5), (6, 7), (2, 6), (1, 5), (1, 4), (0, 3), (0, 2), (0, 1)])]
    #[case("dp-limit"    ,  983832,  2016, vec![(2, 7), (3, 4), (0, 4), (3, 6), (2, 5), (0, 4), (0, 3), (1, 2), (0, 1)])]
    #[case("dp-combo-256",  983790,  2016, vec![(0, 8), (3, 4), (1, 4), (5, 6), (1, 5), (0, 4), (0, 3), (1, 2), (0, 1)])]
    #[case("dp-limit-256",  983832,  2016, vec![(2, 7), (3, 4), (0, 4), (3, 6), (2, 5), (0, 4), (0, 3), (1, 2), (0, 1)])]
    fn test_custom_dp_can_set_minimize(
        #[case] minimize: &str,
        #[case] cost: usize,
        #[case] width: usize,
        #[case] path: Vec<(usize, usize)>,
    ) {
        let expression = "qgcf,sotr,klb,jlretia,hpn,nseha,jgoqm,ipkb,cdfm,d->";
        let shapes = vec![
            vec![5, 2, 9, 4],
            vec![4, 9, 5, 9],
            vec![5, 4, 2],
            vec![5, 4, 9, 7, 5, 3, 6],
            vec![5, 2, 8],
            vec![8, 4, 7, 5, 6],
            vec![5, 2, 9, 5, 8],
            vec![3, 2, 5, 2],
            vec![9, 3, 4, 8],
            vec![3],
        ];
        let (_, info) = contract_path(expression, &shapes, minimize, None).unwrap();
        let path: PathType = path.iter().map(|(i, j)| vec![*i, *j]).collect();
        assert_eq!(info.opt_cost, cost as _);
        assert_eq!(info.largest_intermediate, width as _);
        assert_eq!(info.path, path);
    }

    #[test]
    fn test_dp_errors_when_no_contractions_found() {
        let expression = "jk,igelb,ho,nfcbd,ca,gk,hef,nal,omj,dim->";
        let shapes = vec![
            vec![3, 4],
            vec![8, 6, 4, 8, 5],
            vec![6, 9],
            vec![4, 9, 6, 5, 8],
            vec![6, 8],
            vec![6, 4],
            vec![6, 4, 9],
            vec![4, 8, 8],
            vec![9, 4, 3],
            vec![8, 8, 4],
        ];
        let (_, info) = contract_path(expression, &shapes, "dp-size", None).unwrap();
        let min_cost = info.largest_intermediate;

        assert!(contract_path(expression, &shapes, "dp", min_cost).is_ok());
        assert!(contract_path(expression, &shapes, "dp", min_cost - 1.0).is_err());
    }

    #[test]
    fn test_can_optimize_outer_products() {
        let expression = "ab,cd,ef,fg";
        let shapes = vec![vec![10, 10], vec![10, 10], vec![10, 10], vec![10, 2]];
        let optimizers = ["branch-2", "branch-all", "optimal", "dp", "greedy"];
        for &opt in &optimizers {
            let (path, _) = contract_path(expression, &shapes, opt, None).unwrap();
            assert_eq!(path, vec![vec![2, 3], vec![0, 2], vec![0, 1]]);
        }
    }

    #[test]
    fn test_large_path() {
        for num_symbols in [2, 3, 26, 26 + 26, 256 - 140, 300] {
            let symbols: String = (0..num_symbols).map(opt_einsum_path::parser::get_symbol).collect();
            let dimension_dict: BTreeMap<char, usize> = symbols.chars().zip([2, 3, 4].into_iter().cycle()).collect();
            let expression: String =
                symbols.chars().collect_vec().windows(2).map(|w| w.iter().collect::<String>()).collect_vec().join(",");
            let tensors = build_shapes(&expression, Some(&dimension_dict), true).unwrap();
            let _ = contract_path(&expression, &tensors, "greedy", None).unwrap();
        }
    }
}
