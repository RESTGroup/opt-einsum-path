# Einsum Path Contraction Optimization for Tensor Contraction

This crate performs **path optimization for tensor contraction** (not performing the actual tensor contraction computation).

This is a re-implementation (with minor modifications) to [opt_einsum](https://github.com/dgasmith/opt_einsum). The [opt_einsum user document](https://dgasmith.github.io/opt_einsum) is also a good resource for users not familiar with tensor contraction path. 

| Resources | Badges |
|--|--|
| Crate | [![Crate](https://img.shields.io/crates/v/opt-einsum-path.svg)](https://crates.io/crates/opt-einsum-path) |
| API Document | [![API Documentation](https://docs.rs/opt-einsum-path/badge.svg)](https://docs.rs/opt-einsum-path) |

## Quick Example

For contraction with multiple tensors:

$$
E_{iajb} = \sum_{\mu \nu \kappa \lambda} C_{\mu i} D_{\nu a} G_{\mu \nu \kappa \lambda} C_{\kappa j} D_{\lambda b}
$$

Computation cost of this contraction, if performs naively, scales as $O(N^8)$. But for certain optimized tensor contraction path, it scales as $O(N^5)$:

```rust
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

println!("Path: {path:?}"); // Vec<Vec<usize>>
// Path: [[0, 2], [1, 3], [0, 2], [0, 1]]

println!("{path_info}"); // PathInfo
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
```

## Supported Optimizers

- `optimal`: Slow, but will always give the best contraction path (by depth-first search).
- `greedy`: Fast and generally good, but will give sub-optimal contraction path occationally.
- `dp` (dynamic programming): Robust, generally gives optimal contraction path. Can be configured to optimize different objectives (`flops`, `size`, `write`, `combo`, `limit`). Defaults to `flops` objective.
- `branch` (branch-bound): Robust, gives more optimal contraction path with increasing branch-search level (`branch-1`, `branch-2`, `branch-all`).
- `random-greedy`: Greedy with some random optimization techniques. After multiple iterations, select the best contraction path. Can set number iterations like `random-greedy-128` for 128 iterations. Defaults to 32 iterations.

Default optimizer is similar (not exactly the same) to high-quality in original python package `opt_einsum` (`auto-hq`):
- `optimal` for [0, 6) tensors;
- `dp` for [6, 20) tensors;
- `random-greedy-128` for [20, inf) tensors.

## Cargo Features

- `par_rand` will allow parallel run (by rayon) in random-related optimizers. This is not activated by default.

## Citation and Relation to `opt_einsum`

This is originally developed for developing rust tensor toolkit [RSTSR](https://github.com/RESTGroup/rstsr) and electronic structure toolkit [REST](https://gitee.com/restgroup/rest). It is **formally NOT** a project related to [opt_einsum](https://github.com/dgasmith/opt_einsum).

The author thanks the original authors of `opt_einsum` and the algorithms implemented in NumPy. This really accelarates development of electronic structure algorithms.

We refer
- **Original implementation**: <https://github.com/dgasmith/opt_einsum>
- **Similar Project in Python** (cotengra): <https://github.com/jcmgray/cotengra>
- **Citation**: Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. *Journal of Open Source Software*, **2018**, *3* (26), 753. doi: <https://doi.org/10.21105/joss.00753>

## Future Development Plans

The following features are not on top priority. Only to be developed if requested by github issues (and not promised to accomplish):
- new optimizers/implementations,
- PyO3 export to python.

For developers, if you wish to develop a customized optimizer,
- the trait `PathOptimizer` can be implemented,
- then do some interface works in `src/paths/mod.rs`.

For electronic structure, this crate should be generally good enough, since in most cases we just handle contractions with no more than 10 tensors.

However, for other fields (many-body physics, tensor networks, quantum circuits, etc.), large number of tensors may involve. Though `greedy` or `dp` optimizers are usable for those cases, algorithms and implementation of this crate is not fully efficient (but may be faster than python's, especially `dp`). In this mean time, we recommend [cotengrust](https://github.com/jcmgray/cotengrust) for those specialized usages.

## Miscellaneous

This crate is licensed as Apache v2.

This crate contains code assisted by AI (deepseek) from language translation from original python package. These code have been checked manually.
