# Tensor Contraction Path Optimization

Computes the optimal contraction path for a tensor network operation using Einstein summation notation. This function analyzes the computational graph of a tensor contraction and finds the most efficient order of operations to minimize computational cost (FLOPs) and memory usage[^1].

[^1]: FLOPs is the usual target to be optimized. However, note that for optimizers `dp-size`, `dp-write`, `dp-combo`, `dp-limit`, they optimize different objectives.

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

## Parameters

### Parameter `subscripts: &str`

Einstein summation expression describing the tensor operation. The format follows standard einsum notation:
- Input tensors are separated by commas
- The output is specified after `->`, can be inferred if not specified
- Repeated indices are summed over
- Ellipsis (`...`) is supported

**Examples:**
- Matrix multiplication: `"ij, jk -> ik"` or identically `"ij, jk"`
- Tensor contraction: `"abcd, cefg -> abdefg"` or identically `"abcd, cefg"`
- Batch matrix multiplication with broadcasting: `"...ij, ...jk -> ...ik"` or identically ``"...ij, ...jk"``
- Tensor transpose: `"ijk -> ikj"`
- Tensor diagonal: `"iik -> ik"`
- Tensor reduce: `"ij, ji ->"` or identically `"ij, ji"`

### Parameter `operands: &[TensorShapeType]`

Slice of tensor shapes corresponding to each input tensor in the subscript expression. Each shape is a `Vec<usize>` representing the dimensions of the tensor.

**Example:**
- `let operands = vec![vec![2, 3], vec![3, 4]]; // For "ij,jk->ik"`

### Parameter `optimize: impl PathOptimizer`

Specifies the optimization algorithm to use. Accepts multiple types:

#### String identifiers:

- `"optimal"`: Exhaustive search (best but slow, good for ≤6 tensors)
- `"greedy"`: Fast heuristic (good balance of speed/quality)
- `"dp"` or `"dynamic-programming"`: Dynamic programming approach
- `"dp"`, `"dp-*"`: Dynamic programming with different minimization metrics:
    - `flops` (default): Computation cost based on additions and multiplications (FMA counted as 2 FLOPs)
    - `size`: Maximum size of intermediate tensor
    - `write`: Total size of intermediate tensors to be created
    - `combo-*`: Hybrid flops and write; by default `combo` or `combo-64` is 64:1 (mimic high DRAM-bandwidth servers), and this ratio can be set in string;
    - `limit-*`: Hybrid flops and size; by default `limit` or `limit-64` is 64:1, and this ratio can be set in string;
- `"branch-all"`, `"branch-2"`, `"branch-1"`: Branch and bound with different branching depths
- `"random-greedy"`, `"random-greedy-*"`: Randomized greedy with specified iterations, by default run 32 iterations; by setting to `"random-greedy-128"` will perform 128 iterations.
    - This can be paralleled by rayon. Use crate feature `par_rand` to activate parallel.
- `"auto"`: Automatic selection based on problem size
- `"auto-hq"`: High-quality automatic selection (default for most use cases)
- `"no-optimize"`: No optimization (naive contraction order)

Different optimizers may produce different results:

| Optimizer | Best For | Time Complexity | Quality |
|-----------|----------|-----------------|---------|
| `optimal` | Small problems (≤6 tensors) | Exponential | Best |
| `dp-*` | Medium problems | Faster exponential | Excellent |
| `branch-*` | Medium-large problems | Configurable | Very Good |
| `random-greedy-*` | Large problems | $O(k n^2)$, can parallel | Good |
| `greedy` | Very Large problems | $O(n^2)$ | Good |
| `auto-hq` | General purpose | Adaptive | Better |
| `auto` | General purpose | Adaptive | Faster |

**Note:** The actual performance depends on the specific tensor network structure. For critical applications, it's recommended to test multiple optimizers.

#### Boolean values:

- `true`: Equivalent to `"auto-hq"`
- `false`: Equivalent to `"no-optimize"`

#### List of path:

You can also directly specify contraction path.

The returned path is exactly the same to your input. No path optimization will be performed.

```rust
# use opt_einsum_path::contract_path;
let subscripts = "qgcf,sotr,klb,jlretia,hpn,nseha,jgoqm,ipkb,cdfm,d->";
# let shapes = vec![
#     vec![5, 2, 9, 4],
#     vec![4, 9, 5, 9],
#     vec![5, 4, 2],
#     vec![5, 4, 9, 7, 5, 3, 6],
#     vec![5, 2, 8],
#     vec![8, 4, 7, 5, 6],
#     vec![5, 2, 9, 5, 8],
#     vec![3, 2, 5, 2],
#     vec![9, 3, 4, 8],
#     vec![3],
# ];
let path_inp = [[0, 8], [3, 4], [1, 4], [5, 6], [1, 5], [0, 4], [0, 3], [1, 2], [0, 1]];
let (path, path_info) = contract_path(subscripts, &shapes, path_inp, None).unwrap();
assert_eq!(path, path_inp);
# println!("{path_info}");
```

#### Custom optimizer instances:

You can create and pass instances of specific optimizer structs for fine-grained control:

```rust
use opt_einsum_path::contract_path;
use opt_einsum_path::paths::dp::DynamicProgramming;

// Custom dynamic programming optimizer
let dp_optimizer = DynamicProgramming {
    minimize: "size".to_string(),
    search_outer: true,
    cost_cap: (12.0).into(),
    combo_factor: 2.0,
};

// Perform contraction with custom `dp_optimizer` with intermediate memory limit to 24 numbers.
let expr = "ij, jk -> ik";
let tensors = vec![vec![2, 3], vec![3, 4]];
let result = contract_path(expr, &tensors, dp_optimizer, 24.0);
```

### Parameter `memory_limit: impl Into<SizeLimitType>`

Constraints the maximum size of intermediate tensors during contraction.

Unit of memory limit is size of numbers. For example of 1 MB memory limit for `f64` tensor (8 bytes per number), the size of tensor must be lower than 1 MB / 8 bytes = 131072.

This parameter can be used with overrides.

- Specify size limit
    - `f64` type: e.g. `131072.0`.
    - `SizeLimitType::Size(limit)`
- No memory constraint
    - `None`
    - `&str` type: `"none"` or `"no-limit"`
    - `SizeLimitType::None`
- Limit to the size of the largest input tensor
    - `&str` type: `"max-input"`
    - `SizeLimitType::MaxInput`

**Note:** When a memory limit is specified, the optimizer may choose a suboptimal path that respects the memory constraint, potentially performing a single large contraction of all remaining tensors if no pairwise contractions are possible within the limit.

## Returns

`Result<(PathType, PathInfo), String>` where:

### Tuple first `PathType` (`Vec<Vec<usize>>`)

The optimal contraction path as a sequence of operations. Each inner vector contains the indices of tensors to contract at that step.

#### Example

Recall the quick example:

$$
E_{iajb} = \sum_{\mu \nu \kappa \lambda} C_{\mu i} D_{\nu a} G_{\mu \nu \kappa \lambda} C_{\kappa j} D_{\lambda b}
$$

```plain
                  Path:  [[0, 2], [1, 3], [0, 2], [0, 1]]
  Complete contraction:  μi,νa,μνκλ,κj,λb->iajb
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   5           GEMM          μi,μνκλ->νκλi                   νa,κj,λb,νκλi->iajb
   5           TDOT          κj,νκλi->νλij                      νa,λb,νλij->iajb
   5           GEMM          νa,νλij->λija                         λb,λija->iajb
   5           GEMM          λb,λija->iajb                            iajb->iajb
```

The contraction route can be interpreted as follows:

```plain
                                                        Result  Path
Slot      0      | 1      | 2      | 3      | 4
Contract  C_μi   | D_νa   | G_μνκλ | C_κj   | D_λb   -> E_iajb
Step 1    C_μi            * G_μνκλ                   -> T_νκλi  [0, 2]
Remain    D_νa   | C_κj   | D_λb   | T_νκλi
Step 2             C_κj            * T_νκλi          -> T_νλij  [1, 3]
Remain    D_νa   | D_λb   | T_νλij
Step 3    D_νa            * T_νλij                   -> T_λija  [0, 2]
Remain    D_λb   | T_λija
Step 4    D_λb   * T_λija                            -> E_iajb  [0, 1]
```

### Tuple second `PathInfo`

Detailed information about the contraction containing:
- Computational costs (naive vs optimized FLOPs)
- Memory requirements
- Contraction sequence details
- Theoretical speedup
- Intermediate tensor sizes

This variable is better printed as display (i.e. `{path_info}`), instead of debug (i.e. `{path_info:?}`).

## Performance Considerations

- For small tensor networks (≤6 tensors), `optimal` provides the best results
- For medium networks (7-20 tensors), `dp` or `branch-2` offer good quality
- For large networks (>20 tensors), `greedy` or `random-greedy-128` are recommended
- Memory constraints can significantly affect the optimal path choice
- The function performs only path optimization, not actual tensor computation

## See Also

- [`PathOptimizer`] trait for custom optimizers
- [`PathInfo`] for detailed contraction analysis
- Original [opt_einsum](https://github.com/dgasmith/opt_einsum) Python package for reference
