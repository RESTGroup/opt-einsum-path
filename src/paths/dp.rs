// src/paths/dp.rs
use crate::*;
use std::collections::VecDeque;

// Define our tree type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractionTree {
    Leaf(usize),
    Node(Vec<ContractionTree>),
}

impl From<usize> for ContractionTree {
    fn from(value: usize) -> Self {
        ContractionTree::Leaf(value)
    }
}

impl From<Vec<ContractionTree>> for ContractionTree {
    fn from(value: Vec<ContractionTree>) -> Self {
        ContractionTree::Node(value)
    }
}

impl From<Vec<usize>> for ContractionTree {
    fn from(value: Vec<usize>) -> Self {
        ContractionTree::Node(value.into_iter().map(ContractionTree::Leaf).collect())
    }
}

/// Converts a contraction tree to a contraction path.
///
/// A contraction tree can either be a leaf node containing an integer (representingno contraction)
/// or a node containing a sequence of subtrees to be contracted.Contractions are commutative
/// (order-independent) and solutions are not unique.
///
/// # Parameters
///
/// * `tree` - The contraction tree to convert, represented as a `ContractionTree` enum where leaves
///   are integers and nodes contain sequences of subtrees.
///
/// # Returns
///
/// A `PathType` (Vec<Vec<usize>>) representing the contraction path, where each inner Vec<usize>
/// represents a single contraction step with the indices of tensors to contract.
///
/// The conversion process works by:
/// 1. Processing leaf nodes (integers) first to determine their positions
/// 2. Building the contraction sequence by tracking elementary tensors and remaining contractions
/// 3. Maintaining proper index accounting throughout the conversion
///
/// Note: This implementation matches the behavior of Python's `_tree_to_sequence` function
/// from opt_einsum, producing equivalent output for equivalent input trees.
pub fn tree_to_sequence(tree: &ContractionTree) -> PathType {
    // Handle leaf case (equivalent to Python's int case)
    if let ContractionTree::Leaf(_) = tree {
        return Vec::new();
    }

    let mut c: VecDeque<&ContractionTree> = VecDeque::new(); // list of remaining contractions
    c.push_back(tree);

    let mut t: Vec<usize> = Vec::new(); // list of elementary tensors
    let mut s: VecDeque<Vec<usize>> = VecDeque::new(); // resulting contraction sequence

    while !c.is_empty() {
        let j = c.pop_back().unwrap();
        s.push_front(Vec::new());

        // First process the integer leaves
        if let ContractionTree::Node(children) = j {
            // Collect integer leaves first
            let mut int_children: Vec<usize> = children
                .iter()
                .filter_map(|child| match child {
                    ContractionTree::Leaf(i) => Some(*i),
                    _ => None,
                })
                .collect();

            // Sort them as in Python
            int_children.sort_unstable();

            for i in int_children {
                // Calculate the position as in Python: sum(1 for q in t if q < i)
                let pos = t.iter().filter(|&&q| q < i).count();
                s[0].push(pos);
                t.insert(pos, i);
            }

            // Then process the non-integer children (other nodes)
            for i_tup in children.iter().filter(|child| matches!(child, ContractionTree::Node(_))) {
                let pos = t.len() + c.len();
                s[0].push(pos);
                c.push_back(i_tup);
            }
        }
    }

    s.into_iter().collect()
}

/// Finds disconnected subgraphs in a list of input tensor dimensions.
///
/// Input tensors are considered connected if they share summation indices (indices not
/// present in the output). Disconnected subgraphs can be contracted independently
/// before forming outer products, which is useful for optimization.
///
/// # Parameters
///
/// * `inputs` - Slice of sets representing input tensor dimensions (lhs of einsum)
/// * `output` - Set representing output tensor dimensions (rhs of einsum)
///
/// # Returns
///
/// Vector of sets where each set contains indices of connected input tensors.
///
/// # Note
///
/// - Summation indices are determined as `(union of all inputs) \ output`
/// - The order of returned subgraphs is implementation-defined
/// - Within each subgraph, the order of tensor indices is sorted
pub fn find_disconnected_subgraphs(inputs: &[ArrayIndexType], output: &ArrayIndexType) -> Vec<BTreeSet<usize>> {
    let mut subgraphs = Vec::new();
    let mut unused_inputs: BTreeSet<usize> = (0..inputs.len()).collect();

    // Calculate all summation indices (union of all inputs minus output)
    let input_indices: ArrayIndexType = inputs.iter().flat_map(|set| set.iter()).cloned().collect();
    let i_sum = &input_indices - output;

    while !unused_inputs.is_empty() {
        let mut g = BTreeSet::new();
        let mut queue = VecDeque::new();

        // Start with any remaining input
        queue.push_back(*unused_inputs.iter().next().unwrap());
        unused_inputs.remove(&queue[0]);

        while !queue.is_empty() {
            let j = queue.pop_front().unwrap();
            g.insert(j);

            // Get summation indices for current input
            let i_tmp: ArrayIndexType = &i_sum & &inputs[j];

            // Find connected inputs
            let neighbors = unused_inputs.iter().filter(|&&k| !inputs[k].is_disjoint(&i_tmp)).cloned().collect_vec();

            for neighbor in neighbors {
                queue.push_back(neighbor);
                unused_inputs.remove(&neighbor);
            }
        }
        subgraphs.push(g);
    }
    subgraphs
}

/// Select elements of `seq` which are marked by the bitmap `s`.
///
/// # Parameters
///
/// * `s` - Bitmap where each bit represents whether to select the corresponding element
/// * `seq` - Sequence of items to select from
///
/// # Returns
///
/// An iterator yielding selected elements from `seq` where the corresponding bit in `s` is set.
pub fn bitmap_select<T>(s: usize, seq: &[T]) -> impl Iterator<Item = &T> {
    seq.iter().enumerate().filter(move |(i, _)| (s >> i) & 1 == 1).map(|(_, x)| x)
}

// Calculates the effective outer indices of the intermediate tensor
/// corresponding to the subgraph `s`.
///
/// # Parameters
///
/// * `g` - Bitmap representing all tensors in the current graph
/// * `all_tensors` - Bitmap representing all possible tensors
/// * `s` - Bitmap representing the subgraph to calculate legs for
/// * `inputs` - Slice of input tensor dimension sets
/// * `i1_cut_i2_wo_output` - Precomputed intersection of indices (i1 ∩ i2) \ output
/// * `i1_union_i2` - Precomputed union of indices (i1 ∪ i2)
///
/// # Returns
///
/// The effective outer indices of the intermediate tensor
pub fn dp_calc_legs(
    g: usize,
    all_tensors: usize,
    s: usize,
    inputs: &[&ArrayIndexType],
    i1_cut_i2_wo_output: &ArrayIndexType,
    i1_union_i2: &ArrayIndexType,
) -> ArrayIndexType {
    // set of remaining tensors (= g & (!s))
    let r = g & (all_tensors ^ s);

    // indices of remaining indices:
    let i_r = if r != 0 {
        bitmap_select(r, inputs).fold(ArrayIndexType::new(), |acc, s| &acc | s)
    } else {
        ArrayIndexType::new()
    };

    // contraction indices:
    let i_contract = i1_cut_i2_wo_output - &i_r;
    i1_union_i2 - &i_contract
}

#[derive(Debug, Clone)]
pub struct DpTerm {
    pub indices: ArrayIndexType,
    pub cost: SizeType,
    pub contract: ContractionTree,
}

pub struct DpCompareArgs<'a> {
    // parameters
    pub minimize: &'a str,
    pub combo_factor: SizeType,
    // inputs
    pub inputs: &'a [&'a ArrayIndexType],
    pub size_dict: &'a SizeDictType,
    pub all_tensors: usize,
    pub memory_limit: Option<SizeType>,
    pub cost_cap: SizeType,
    pub bitmap_g: usize,
}

impl<'a> DpCompareArgs<'a> {
    /// Performs the inner comparison of whether the two subgraphs (the bitmaps `s1` and `s2`)
    /// should be merged and added to the dynamic programming search. Will skip for a number of
    /// reasons:
    /// 1. If the number of operations to form `s = s1 | s2` including previous contractions is
    ///    above the cost-cap.
    /// 2. If we've already found a better way of making `s`.
    /// 3. If the intermediate tensor corresponding to `s` is going to break the memory limit.
    pub fn compare_flops(
        &self,
        xn: &mut BTreeMap<usize, DpTerm>,
        s1: usize,
        s2: usize,
        term1: &DpTerm,
        term2: &DpTerm,
        i1_cut_i2_wo_output: &ArrayIndexType,
    ) {
        let DpTerm { indices: i1, cost: cost1, contract: contract1 } = term1;
        let DpTerm { indices: i2, cost: cost2, contract: contract2 } = term2;
        let i1_union_i2 = i1 | i2;

        let cost = cost1 + cost2 + helpers::compute_size_by_dict(i1_union_i2.iter(), self.size_dict);
        if cost <= self.cost_cap {
            let s = s1 | s2;
            if xn.get(&s).is_none_or(|term| cost < term.cost) {
                let indices =
                    dp_calc_legs(self.bitmap_g, self.all_tensors, s, self.inputs, i1_cut_i2_wo_output, &i1_union_i2);
                let mem = helpers::compute_size_by_dict(indices.iter(), self.size_dict);
                if self.memory_limit.is_none_or(|limit| mem <= limit) {
                    let contract = vec![contract1.clone(), contract2.clone()].into();
                    xn.insert(s, DpTerm { indices, cost, contract });
                }
            }
        }
    }

    /// Like `compare_flops` but sieves the potential contraction based
    /// on the size of the intermediate tensor created, rather than the number of
    /// operations, and so calculates that first.
    pub fn compare_size(
        &self,
        xn: &mut BTreeMap<usize, DpTerm>,
        s1: usize,
        s2: usize,
        term1: &DpTerm,
        term2: &DpTerm,
        i1_cut_i2_wo_output: &ArrayIndexType,
    ) {
        let DpTerm { indices: i1, cost: cost1, contract: contract1 } = term1;
        let DpTerm { indices: i2, cost: cost2, contract: contract2 } = term2;
        let i1_union_i2 = i1 | i2;
        let s = s1 | s2;
        let indices = dp_calc_legs(self.bitmap_g, self.all_tensors, s, self.inputs, i1_cut_i2_wo_output, &i1_union_i2);

        let mem = helpers::compute_size_by_dict(indices.iter(), self.size_dict);
        let cost = (*cost1).max(*cost2).max(mem);
        if cost <= self.cost_cap
            && xn.get(&s).is_none_or(|term| cost < term.cost)
            && self.memory_limit.is_none_or(|limit| mem <= limit)
        {
            let contract = vec![contract1.clone(), contract2.clone()].into();
            xn.insert(s, DpTerm { indices, cost, contract });
        }
    }
    /// Like `compare_flops` but sieves the potential contraction based
    /// on the total size of memory created, rather than the number of
    /// operations, and so calculates that first.
    pub fn compare_write(
        &self,
        xn: &mut BTreeMap<usize, DpTerm>,
        s1: usize,
        s2: usize,
        term1: &DpTerm,
        term2: &DpTerm,
        i1_cut_i2_wo_output: &ArrayIndexType,
    ) {
        let DpTerm { indices: i1, cost: cost1, contract: contract1 } = term1;
        let DpTerm { indices: i2, cost: cost2, contract: contract2 } = term2;
        let i1_union_i2 = i1 | i2;
        let s = s1 | s2;
        let indices = dp_calc_legs(self.bitmap_g, self.all_tensors, s, self.inputs, i1_cut_i2_wo_output, &i1_union_i2);

        let mem = helpers::compute_size_by_dict(indices.iter(), self.size_dict);
        let cost = cost1 + cost2 + mem;

        if cost <= self.cost_cap
            && xn.get(&s).is_none_or(|term| cost < term.cost)
            && self.memory_limit.is_none_or(|limit| mem <= limit)
        {
            let contract = vec![contract1.clone(), contract2.clone()].into();
            xn.insert(s, DpTerm { indices, cost, contract });
        }
    }

    /// Like `compare_flops` but sieves the potential contraction based
    /// on some combination of both the flops and size.
    pub fn compare_combo(
        &self,
        xn: &mut BTreeMap<usize, DpTerm>,
        s1: usize,
        s2: usize,
        term1: &DpTerm,
        term2: &DpTerm,
        i1_cut_i2_wo_output: &ArrayIndexType,
    ) {
        let DpTerm { indices: i1, cost: cost1, contract: contract1 } = term1;
        let DpTerm { indices: i2, cost: cost2, contract: contract2 } = term2;
        let i1_union_i2 = i1 | i2;
        let s = s1 | s2;
        let indices = dp_calc_legs(self.bitmap_g, self.all_tensors, s, self.inputs, i1_cut_i2_wo_output, &i1_union_i2);

        let mem = helpers::compute_size_by_dict(indices.iter(), self.size_dict);
        let f = helpers::compute_size_by_dict(i1_union_i2.iter(), self.size_dict);

        // Hardcoded to sum: f + self.combo_factor * mem
        let combined = match self.minimize {
            "combo" => f + self.combo_factor * mem,
            "limit" => f.max(self.combo_factor * mem),
            _ => panic!("Unknown minimize type for combo mode: {}", self.minimize),
        };
        let cost = cost1 + cost2 + combined;

        if cost <= self.cost_cap
            && xn.get(&s).is_none_or(|term| cost < term.cost)
            && self.memory_limit.is_none_or(|limit| mem <= limit)
        {
            let contract = vec![contract1.clone(), contract2.clone()].into();
            xn.insert(s, DpTerm { indices, cost, contract });
        }
    }

    pub fn scale(&self) -> SizeType {
        match self.minimize {
            "flops" | "size" | "write" => SizeType::one(),
            "combo" | "limit" => SizeType::MAX,
            _ => panic!("Unknown minimize type: {}", self.minimize),
        }
    }

    pub fn compare(
        &self,
        xn: &mut BTreeMap<usize, DpTerm>,
        s1: usize,
        s2: usize,
        term1: &DpTerm,
        term2: &DpTerm,
        i1_cut_i2_wo_output: &ArrayIndexType,
    ) {
        let minimize_split = self.minimize.split('-').collect_vec();
        if minimize_split.is_empty() {
            panic!("Unknown minimize type: {}", self.minimize);
        }
        match minimize_split[0] {
            "flops" => self.compare_flops(xn, s1, s2, term1, term2, i1_cut_i2_wo_output),
            "size" => self.compare_size(xn, s1, s2, term1, term2, i1_cut_i2_wo_output),
            "write" => self.compare_write(xn, s1, s2, term1, term2, i1_cut_i2_wo_output),
            "combo" | "limit" => self.compare_combo(xn, s1, s2, term1, term2, i1_cut_i2_wo_output),
            _ => panic!("Unknown minimize type: {}", self.minimize),
        }
    }
}

/// Makes a simple left-to-right binary tree out of a sequence of terms.
///
/// # Arguments
/// * `seq` - Sequence of terms to nest
///
/// # Returns
/// A `ContractionTree` representing the left-nested binary tree
pub fn simple_tree_tuple(seq: &[ContractionTree]) -> ContractionTree {
    seq.iter().cloned().reduce(|left, right| ContractionTree::Node(vec![left, right])).unwrap()
}
use std::collections::{BTreeMap, BTreeSet};

/// Parses inputs for single term index operations (indices appearing on one tensor).
///
/// Returns:
/// - Parsed inputs with single indices removed
/// - Inputs that were reduced to scalars
/// - Contractions needed for the reductions
pub fn dp_parse_out_single_term_ops(
    inputs: &[&ArrayIndexType],
    all_inds: &[char],
    ind_counts: &SizeDictType,
) -> (Vec<ArrayIndexType>, Vec<ContractionTree>, Vec<ContractionTree>) {
    let i_single: BTreeSet<char> = all_inds.iter().filter(|&c| ind_counts.get(c) == Some(&1)).cloned().collect();

    let mut inputs_parsed = Vec::new();
    let mut inputs_done = Vec::new();
    let mut inputs_contractions = Vec::new();

    for (j, input) in inputs.iter().enumerate() {
        let i_reduced: ArrayIndexType = *input - &i_single;
        if i_reduced.is_empty() && !input.is_empty() {
            // Input reduced to scalar - remove
            inputs_done.push(vec![j].into());
        } else {
            // Add single contraction if indices were reduced
            inputs_contractions.push(if i_reduced.len() != input.len() { vec![j].into() } else { j.into() });
            inputs_parsed.push(i_reduced);
        }
    }

    (inputs_parsed, inputs_done, inputs_contractions)
}

#[derive(Debug, Clone)]
pub struct DynamicProgramming {
    pub minimize: String,
    pub search_outer: bool,
    pub cost_cap: SizeLimitType,
    pub combo_factor: SizeType,
}

impl Default for DynamicProgramming {
    fn default() -> Self {
        Self {
            minimize: "flops".into(),
            search_outer: false,
            cost_cap: None.into(),
            combo_factor: SizeType::from_usize(64).unwrap(),
        }
    }
}

impl DynamicProgramming {
    pub fn find_optimal_path(
        &self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        // Initialize cost function parameters
        let check_outer = match self.search_outer {
            true => |_: &ArrayIndexType| true,
            false => |x: &ArrayIndexType| !x.is_empty(),
        };

        // Count index occurrences
        let ind_counts: BTreeMap<char, usize> =
            inputs.iter().flat_map(|inds| inds.iter()).chain(output.iter()).fold(BTreeMap::new(), |mut counts, &c| {
                *counts.entry(c).or_default() += 1;
                counts
            });

        let all_inds: Vec<char> = ind_counts.keys().copied().collect();

        // Parse single-term operations
        let (inputs, inputs_done, inputs_contractions) = dp_parse_out_single_term_ops(inputs, &all_inds, &ind_counts);
        let inputs_ref = inputs.iter().collect_vec();

        if inputs.is_empty() {
            return tree_to_sequence(&simple_tree_tuple(&inputs_done));
        }

        // Initialize subgraph tracking
        let mut subgraph_contractions = inputs_done;
        let mut subgraph_sizes: Vec<SizeType> = vec![SizeType::one(); subgraph_contractions.len()];

        // Find disconnected subgraphs
        let subgraphs = if self.search_outer {
            vec![(0..inputs.len()).collect_vec()]
        } else {
            find_disconnected_subgraphs(&inputs, output).into_iter().map(|s| s.into_iter().collect()).collect()
        };

        let all_tensors = (1 << inputs.len()) - 1;

        for g in subgraphs {
            let bitmap_g = g.iter().fold(0, |acc, &j| acc | (1 << j));

            // Initialize DP table
            let mut x: Vec<BTreeMap<usize, DpTerm>> = vec![BTreeMap::new(); g.len() + 1];
            x[1] = g
                .iter()
                .map(|&j| {
                    (1 << j, DpTerm {
                        indices: inputs[j].clone(),
                        cost: SizeType::zero(),
                        contract: inputs_contractions[j].clone(),
                    })
                })
                .collect();

            // Initialize cost cap
            let subgraph_inds = bitmap_select(bitmap_g, &inputs).flat_map(|inds| inds.iter().copied()).collect();

            let mut cost_cap = match self.cost_cap {
                SizeLimitType::Size(cap) => cap,
                SizeLimitType::None => SizeType::MAX,
                SizeLimitType::MaxInput => helpers::compute_size_by_dict((&subgraph_inds & output).iter(), size_dict),
            };

            let cost_increment = if subgraph_inds.is_empty() {
                SizeType::from_usize(2).unwrap()
            } else {
                subgraph_inds
                    .iter()
                    .map(|c| size_dict[c] as SizeType)
                    .fold(SizeType::MAX, SizeType::min)
                    .max(SizeType::from_usize(2).unwrap())
            };

            let mut dp_comp_args = DpCompareArgs {
                inputs: &inputs_ref,
                size_dict,
                all_tensors,
                memory_limit,
                cost_cap,
                bitmap_g,
                combo_factor: self.combo_factor,
                minimize: &self.minimize,
            };
            let naive_scale = dp_comp_args.scale();
            let naive_cost = naive_scale
                * SizeType::from_usize(inputs.len()).unwrap()
                * SizeType::from_usize(size_dict.values().product()).unwrap();

            while x.last().unwrap().is_empty() {
                for n in 2..=g.len() {
                    let mut xn = x[n].clone();
                    for m in 1..=(n / 2) {
                        for (&s1, term1) in &x[m] {
                            for (&s2, term2) in &x[n - m] {
                                if (s1 & s2 == 0) && (m != n - m || s1 < s2) {
                                    let i1 = &term1.indices;
                                    let i2 = &term2.indices;
                                    let i1_cut_i2_wo_output = &(i1 & i2) - output;
                                    if check_outer(&i1_cut_i2_wo_output) {
                                        dp_comp_args.compare(&mut xn, s1, s2, term1, term2, &i1_cut_i2_wo_output);
                                    }
                                }
                            }
                        }
                    }
                    x[n] = xn;
                }

                // avoid overflow
                cost_cap = match cost_cap >= SizeType::MAX / cost_increment {
                    true => SizeType::MAX,
                    false => cost_cap * cost_increment,
                };
                dp_comp_args.cost_cap = cost_cap;

                if cost_cap > naive_cost && x.last().unwrap().is_empty() {
                    panic!("No contraction found for given memory_limit");
                }
            }

            let (_, term) = x.last().unwrap().iter().next().unwrap();
            subgraph_contractions.push(term.contract.clone());
            subgraph_sizes.push(helpers::compute_size_by_dict(term.indices.iter(), size_dict));
        }

        // Sort subgraphs by size
        let sorted_indices =
            (0..subgraph_sizes.len()).sorted_by(|&a, &b| subgraph_sizes[a].partial_cmp(&subgraph_sizes[b]).unwrap());
        let sorted_contractions = sorted_indices.map(|i| subgraph_contractions[i].clone()).collect_vec();

        tree_to_sequence(&simple_tree_tuple(&sorted_contractions))
    }
}

impl PathOptimizer for DynamicProgramming {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        self.find_optimal_path(inputs, output, size_dict, memory_limit)
    }
}

impl From<&str> for DynamicProgramming {
    fn from(s: &str) -> Self {
        let s = s.replace(['_', ' '], "-").to_lowercase();
        if s == "dp" || s == "dynamic-programming" {
            return DynamicProgramming::default();
        }
        if s.starts_with("dp-") {
            let minimize = s.strip_prefix("dp-").unwrap();
            // sanity of minimize
            if minimize.starts_with("combo") || minimize.starts_with("limit") {
                let minimize_split = minimize.split('-').collect_vec();
                if minimize_split.len() > 2 {
                    panic!("Unknown dynamic programming optimizer: {s}");
                }
                match minimize_split.len() {
                    1 => {
                        let minimize = minimize_split[0];
                        if minimize != "combo" && minimize != "limit" {
                            panic!("Unknown dynamic programming optimizer: {s}");
                        }
                        return DynamicProgramming { minimize: minimize.into(), ..Default::default() };
                    },
                    2 => {
                        let minimize = minimize_split[0];
                        if minimize != "combo" && minimize != "limit" {
                            panic!("Unknown dynamic programming optimizer: {s}");
                        }
                        let combo_factor = match minimize_split[1].parse::<SizeType>() {
                            Ok(factor) => factor,
                            Err(_) => panic!("Invalid combo factor in dynamic programming optimizer: {s}"),
                        };
                        return DynamicProgramming { minimize: minimize.into(), combo_factor, ..Default::default() };
                    },
                    _ => panic!("Unknown dynamic programming optimizer: {s}"),
                };
            } else if minimize == "flops" || minimize == "size" || minimize == "write" {
                return DynamicProgramming { minimize: minimize.into(), ..Default::default() };
            } else {
                panic!("Unknown dynamic programming optimizer: {s}");
            }
        }
        panic!("Unknown dynamic programming optimizer: {s}");
    }
}

#[test]
fn test_tree_to_sequence() {
    let tree: ContractionTree = ContractionTree::from(vec![
        ContractionTree::from(vec![1, 2]),
        vec![ContractionTree::from(0), ContractionTree::from(vec![4, 5, 3])].into(),
    ]);

    let path = tree_to_sequence(&tree);
    println!("{path:?}");
    assert_eq!(path, vec![vec![1, 2], vec![1, 2, 3], vec![0, 2], vec![0, 1]]);
}

#[test]
fn test_find_disconnected_subgraphs() {
    use crate::helpers::setify;
    // First test case
    let inputs1 = vec![setify("ab"), setify("c"), setify("ad")];
    let output1 = setify("bd");
    let result1 = find_disconnected_subgraphs(&inputs1, &output1);
    assert_eq!(result1, vec![setify([0, 2]), setify([1])]);

    // Second test case
    let inputs2 = vec![setify("ab"), setify("c"), setify("ad")];
    let output2 = setify("abd");
    let result2 = find_disconnected_subgraphs(&inputs2, &output2);
    assert_eq!(result2, vec![setify([0]), setify([1]), setify([2])]);
}

#[test]
fn test_bitmap_select() {
    use crate::helpers::setify;
    let seq = vec![setify("A"), setify("B"), setify("C"), setify("D"), setify("E")];

    // Test case from Python example
    let selected = bitmap_select(0b11010, &seq).collect_vec();
    assert_eq!(selected, vec![&setify("B"), &setify("D"), &setify("E")]);

    // Additional test cases
    assert_eq!(bitmap_select(0b00000, &seq).count(), 0);
    assert_eq!(bitmap_select(0b11111, &seq).count(), 5);
    assert_eq!(bitmap_select(0b00001, &seq).collect_vec(), vec![&setify("A")]);
}

#[test]
fn test_simple_tree_tuple() {
    let tree = simple_tree_tuple(&[1.into(), 2.into(), 3.into(), 4.into()]);
    assert_eq!(
        tree,
        ContractionTree::Node(vec![
            ContractionTree::Node(vec![ContractionTree::Node(vec![1.into(), 2.into()]), 3.into()]),
            4.into()
        ])
    );
}
