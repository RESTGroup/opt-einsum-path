use opt_einsum_path::blas::can_blas;
use rstest::rstest;
use std::collections::BTreeSet;

#[rstest]
// DOT operations
#[case((vec!["k"  , "k"  ], ""  , "k"  ), Some("DOT")         )] // DDOT
#[case((vec!["ijk", "ijk"], ""  , "ijk"), Some("DOT")         )] // DDOT
// GEMM operations - No Transpose
#[case((vec!["ij" , "jk" ], "ik", "j"  ), Some("GEMM")        )] // GEMM N N
#[case((vec!["ijl", "jlk"], "ik", "jl" ), Some("GEMM")        )] // GEMM N N Tensor
#[case((vec!["ij" , "kj" ], "ik", "j"  ), Some("GEMM")        )] // GEMM N T
#[case((vec!["ijl", "kjl"], "ik", "jl" ), Some("GEMM")        )] // GEMM N T Tensor
// GEMM operations - Transpose Left
#[case((vec!["ji" , "jk" ], "ik", "j"  ), Some("GEMM")        )] // GEMM T N
#[case((vec!["jli", "jlk"], "ik", "jl" ), Some("GEMM")        )] // GEMM T N Tensor
#[case((vec!["ji" , "kj" ], "ik", "j"  ), Some("GEMM")        )] // GEMM T T
#[case((vec!["jli", "kjl"], "ik", "jl" ), Some("GEMM")        )] // GEMM T T Tensor
// GEMM with final transpose
#[case((vec!["ij" , "jk" ], "ki", "j"  ), Some("GEMM")        )]
#[case((vec!["ijl", "jlk"], "ki", "jl" ), Some("GEMM")        )]
// Tensor Dot operations
#[case((vec!["ilj", "jlk"], "ik", "jl" ), Some("TDOT")        )] // FT GEMM N N Tensor
#[case((vec!["ijl", "ljk"], "ik", "jl" ), Some("TDOT")        )] // ST GEMM N N Tensor
// Special cases
#[case((vec!["ijk", "ikj"], ""  , "ijk"), Some("DOT/EINSUM")  )] // Transpose DOT
#[case((vec!["i"  , "j"  ], "ij", ""   ), Some("OUTER/EINSUM"))] // Outer
#[case((vec!["ijk", "ik" ], "j" , "ik" ), Some("GEMV/EINSUM") )] // Matrix-vector
// Invalid cases
#[case((vec!["ijj", "jk" ], "ik", "j"  ), None                )] // Double index
#[case((vec!["ijk", "j"  ], "ij", ""   ), None                )] // Index sum 1
#[case((vec!["ij" , "ij" ], "ij", ""   ), None                )] // Index sum 2
fn test_can_blas_(#[case] inp: (Vec<&str>, &str, &str), #[case] benchmark: Option<&'static str>) {
    let (inputs, result, idx_removed) = inp;
    let idx_removed: BTreeSet<char> = idx_removed.chars().collect();
    assert_eq!(
        can_blas(&inputs, result, &idx_removed, None),
        benchmark,
        "Failed for inputs: {inputs:?}, result: {result}, idx_removed: {idx_removed:?}"
    );
}
