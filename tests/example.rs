#[test]
fn quick_example() {
    use opt_einsum_path::contract_path;
    let expression = "μi,νa,μνκλ,κj,λb->iajb";
    let nao = 1024; // μνκλ
    let nocc = 128; // ij
    let nvir = 896; // ab
    let g = vec![nao, nao, nao, nao]; // G_μνκλ
    let c = vec![nao, nocc]; // C_μi, C_κj
    let d = vec![nao, nvir]; // D_λb, D_νa
    let shapes = [c.clone(), d.clone(), g.clone(), c.clone(), d.clone()];
    let path_result = contract_path(
        expression, // Expression to contract | &str
        &shapes,    // Shapes of the tensors  | &[Vec<usize>]
        "optimal",  // Optimization kind      | impl PathOptimizer
        None,       // Memory limit           | impl Into<SizeLimitType>
    );
    let (path, path_info) = path_result.unwrap();

    println!("Path: {path:?}");
    // Path: [[0, 2], [1, 3], [0, 2], [0, 1]]

    println!("{path_info}");
    //   Complete contraction:  μi,νa,μνκλ,κj,λb->iajb
    //          Naive scaling:  8
    //      Optimized scaling:  5
    //       Naive FLOP count:  7.231e22
    //   Optimized FLOP count:  3.744e14
    //    Theoretical speedup:  1.931e8
    //   Largest intermediate:  1.374e11 elements
    // --------------------------------------------------------------------------------
    // scaling        BLAS                current                             remaining
    // --------------------------------------------------------------------------------
    //    5           GEMM          μi,μνκλ->νκλi                   νa,κj,λb,νκλi->iajb
    //    5           TDOT          κj,νκλi->νλij                      νa,λb,νλij->iajb
    //    5           GEMM          νa,νλij->λija                         λb,λija->iajb
    //    5           GEMM          λb,λija->iajb                            iajb->iajb
    assert_eq!(path, vec![vec![0, 2], vec![1, 3], vec![0, 2], vec![0, 1]]);
}
