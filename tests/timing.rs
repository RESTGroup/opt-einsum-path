use opt_einsum_path::contract::contract_path;
use opt_einsum_path::parser::get_symbol;
use rstest::rstest;

#[rstest]
#[ignore = "This test is computationally expensive and should be run separately."]
#[case("dp", 15)]
#[ignore = "This test is computationally expensive and should be run separately."]
#[case("dp", 50)]
#[ignore = "This test is computationally expensive and should be run separately."]
#[case("optimal", 5)]
#[ignore = "This test is computationally expensive and should be run separately."]
#[case("branch-1", 100)]
#[ignore = "This test is computationally expensive and should be run separately."]
#[case("greedy", 200)]
#[ignore = "This test is computationally expensive and should be run separately."]
#[case("random-greedy-128", 200)]
fn test_large(#[case] optimizer: &str, #[case] n: u32) {
    println!("Testing large path optimization with n = {n} and opt = {optimizer}");

    // set the upper left/right, middle and lower left/right indices
    // --O--
    //   |
    // --O--
    let mut einsum_str = "ab,ac,".to_string();
    einsum_str += &(1..n - 1)
        .map(|i| {
            let j = 3 * i;
            format!(
                "{}{}{},{}{}{},",
                get_symbol(j),
                get_symbol(j - 1),
                get_symbol(j + 2),
                get_symbol(j),
                get_symbol(j - 2),
                get_symbol(j + 1),
            )
        })
        .collect::<String>();
    // finish with last site
    // --O
    //   |
    // --O
    einsum_str += &format!(
        "{}{},{}{}",
        get_symbol(3 * (n - 1)),
        get_symbol(3 * (n - 1) - 1),
        get_symbol(3 * (n - 1)),
        get_symbol(3 * (n - 1) - 2)
    );
    println!("Einsum string: {einsum_str}");

    let len_ops = einsum_str.split(',').count();
    let shapes = vec![vec![3, 10], vec![3, 10]]
        .into_iter()
        .chain(std::iter::repeat_n(vec![3, 10, 10], len_ops - 4))
        .chain(vec![vec![3, 10], vec![3, 10]])
        .collect::<Vec<_>>();

    let time = std::time::Instant::now();
    let (path, info) = contract_path(&einsum_str, &shapes, true, optimizer, None).unwrap();
    let elapsed = time.elapsed();

    println!("Optimal path: {path:?}");
    println!("Optimal cost: {:?}", info.opt_cost);
    println!("Elapsed time: {elapsed:?}");
}
