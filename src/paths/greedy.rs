use crate::*;

pub type GreedyChooseFn = Box<
    dyn FnMut(
        &mut BinaryHeap<GreedyContractionType>,
        &BTreeMap<ArrayIndexType, usize>,
    ) -> Option<GreedyContractionType>,
>;

/// Type representing the cost of a greedy contraction.
///
/// Please note that order of cost is not reversed (greater is better).
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GreedyCostType {
    /// The cost of the contraction.
    ///
    /// Cost is defined as the size of the resulting array after the contraction,
    /// minus the sizes of the two input arrays being contracted:
    /// `size(final) - size(input1) - size(input2)`.
    pub cost: SizeType,
    /// The ID of the first input array being contracted.
    pub id1: usize,
    /// The ID of the second input array being contracted.
    pub id2: usize,
}

impl Eq for GreedyCostType {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for GreedyCostType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Type representing a greedy contraction candidate.
///
/// Order of cost is reversed (less is better).
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct GreedyContractionType {
    /// The cost of the contraction, wrapped in `Reverse` to reverse the order.
    pub cost: Reverse<GreedyCostType>,
    /// The first input array indices being contracted.
    pub k1: ArrayIndexType,
    /// The second input array indices being contracted.
    pub k2: ArrayIndexType,
    /// The resulting array indices after the contraction.
    pub k12: ArrayIndexType,
}

/// Given k1 and k2 tensors, compute the resulting indices k12 and the cost of the contraction.
fn get_candidate(
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    remaining: &BTreeMap<ArrayIndexType, usize>,
    footprints: &BTreeMap<ArrayIndexType, SizeType>,
    dim_ref_counts: &BTreeMap<usize, BTreeSet<char>>,
    k1: &ArrayIndexType,
    k2: &ArrayIndexType,
    cost_fn: paths::CostFn,
) -> GreedyContractionType {
    let either = k1 | k2;
    let two = k1 & k2;
    let one = &either - &two;

    // k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    // indices in output must kept
    let part1 = either.intersection(output);
    // remaining indices kept if referenced by other tensors
    let part2 = two.intersection(&dim_ref_counts[&3]);
    let part3 = one.intersection(&dim_ref_counts[&2]);
    let k12: ArrayIndexType = part1.chain(part2).chain(part3).cloned().collect();

    let size12 = helpers::compute_size_by_dict(k12.iter(), size_dict);
    let footprint1 = footprints[k1];
    let footprint2 = footprints[k2];
    let cost = cost_fn(size12, footprint1, footprint2, 0, 0, 0);

    let id1 = remaining[k1];
    let id2 = remaining[k2];
    let (k1, id1, k2, id2) =
        if id1 > id2 { (k1.clone(), id1, k2.clone(), id2) } else { (k2.clone(), id2, k1.clone(), id1) };

    GreedyContractionType { cost: Reverse(GreedyCostType { cost, id1, id2 }), k1, k2, k12 }
}

/// Given k1 and its candidate k2s, push the best candidates to the queue.
fn push_candidate(
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    remaining: &BTreeMap<ArrayIndexType, usize>,
    footprints: &BTreeMap<ArrayIndexType, SizeType>,
    dim_ref_counts: &BTreeMap<usize, BTreeSet<char>>,
    k1: &ArrayIndexType,
    k2s: &[ArrayIndexType],
    queue: &mut BinaryHeap<GreedyContractionType>,
    push_all: bool,
    cost_fn: paths::CostFn,
) {
    let candidates: Vec<GreedyContractionType> = k2s
        .iter()
        .map(|k2| get_candidate(output, size_dict, remaining, footprints, dim_ref_counts, k1, k2, cost_fn))
        .collect();

    if push_all {
        candidates.into_iter().for_each(|c| queue.push(c));
    } else if let Some(max_cand) = candidates.into_iter().max() {
        queue.push(max_cand);
    }
}

/// Update the reference counts for dimensions in `dims` based on their presence in `dim_to_keys`.
///
/// Note on `dim_ref_counts`: This is a mapping of
/// - 0, 1, 2: the indices that appear in exactly that many remaining tensors (excluding output)
/// - 3: the indices that appear in 3 or more remaining tensors (excluding output)
fn update_ref_counts(
    dim_to_keys: &BTreeMap<char, BTreeSet<ArrayIndexType>>,
    dim_ref_counts: &mut BTreeMap<usize, BTreeSet<char>>,
    dims: &ArrayIndexType,
    output: &ArrayIndexType,
) {
    for dim in dims {
        if output.contains(dim) {
            continue;
        }
        let count = dim_to_keys.get(dim).map(|s| s.len()).unwrap_or(0);

        match count {
            0..=1 => {
                dim_ref_counts.get_mut(&2).unwrap().remove(dim);
                dim_ref_counts.get_mut(&3).unwrap().remove(dim);
            },
            2 => {
                dim_ref_counts.get_mut(&2).unwrap().insert(*dim);
                dim_ref_counts.get_mut(&3).unwrap().remove(dim);
            },
            3.. => {
                dim_ref_counts.get_mut(&2).unwrap().insert(*dim);
                dim_ref_counts.get_mut(&3).unwrap().insert(*dim);
            },
        }
    }
}

/// Default contraction chooser that simply takes the minimum cost option.
///
/// This function will pop candidates only when they are valid (both k1 and k2 must be present in
/// `remaining`).
pub fn simple_chooser(
    queue: &mut BinaryHeap<GreedyContractionType>,
    remaining: &BTreeMap<ArrayIndexType, usize>,
) -> Option<GreedyContractionType> {
    while let Some(cand) = queue.pop() {
        if remaining.contains_key(&cand.k1) && remaining.contains_key(&cand.k2) {
            return Some(cand);
        }
    }
    None
}

/// This is the core function for [`greedy`] but produces a path with static single assignment
/// ids rather than recycled linear ids. SSA ids are cheaper to work with and easier to reason
/// about.
pub fn ssa_greedy_optimize(
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    choose_fn: Option<&mut GreedyChooseFn>,
    cost_fn: Option<paths::CostFn>,
) -> PathType {
    if inputs.is_empty() {
        return vec![];
    }

    if inputs.len() == 1 {
        // Perform a single contraction to match output shape.
        return vec![vec![0]];
    }

    // set the function that chooses which contraction to take
    let push_all = choose_fn.is_none();
    let mut default_chooser: GreedyChooseFn = Box::new(simple_chooser);
    let choose_fn: &mut GreedyChooseFn = if let Some(choose_fn) = choose_fn { choose_fn } else { &mut default_chooser };

    // set the function that assigns a heuristic cost to a possible contraction
    let cost_fn = cost_fn.unwrap_or(paths::util::memory_removed(false));

    // A dim that is common to all tensors might as well be an output dim, since it cannot be contracted
    // until the final step. This avoids an expensive all-pairs comparison to search for possible
    // contractions at each step, leading to speedup in many practical problems where all tensors share
    // a common batch dimension.
    let common_dims = inputs.iter().skip(1).fold(inputs[0].clone(), |acc, s| &acc & s);
    let output = output | &common_dims;

    // Deduplicate shapes by eagerly computing Hadamard products.
    let mut remaining = BTreeMap::new(); // key -> ssa_id
    let mut ssa_ids = inputs.len();
    let mut ssa_path = Vec::new();

    for (ssa_id, &key) in inputs.iter().enumerate() {
        let key = key.clone();
        if let Some(&existing_id) = remaining.get(&key) {
            ssa_path.push(vec![existing_id, ssa_id]);
            remaining.insert(key, ssa_ids);
            ssa_ids += 1;
        } else {
            remaining.insert(key, ssa_id);
        }
    }

    // Keep track of possible contraction dims.
    let mut dim_to_keys: BTreeMap<char, BTreeSet<ArrayIndexType>> = BTreeMap::new();
    for key in remaining.keys() {
        for dim in key - &output {
            dim_to_keys.entry(dim).or_default().insert(key.clone());
        }
    }

    // Keep track of the number of tensors using each dim; when the dim is no longer used it can be
    // contracted. Since we specialize to binary ops, we only care about ref counts of >=2 or >=3.
    let mut dim_ref_counts = BTreeMap::from([(2, BTreeSet::new()), (3, BTreeSet::new())]);
    for (&dim, keys) in &dim_to_keys {
        if keys.len() >= 2 {
            dim_ref_counts.get_mut(&2).unwrap().insert(dim);
        }
        if keys.len() >= 3 {
            dim_ref_counts.get_mut(&3).unwrap().insert(dim);
        }
    }
    output.iter().for_each(|dim| {
        dim_ref_counts.get_mut(&2).unwrap().remove(dim);
        dim_ref_counts.get_mut(&3).unwrap().remove(dim);
    });

    // Compute separable part of the objective function for contractions.
    let mut footprints: BTreeMap<ArrayIndexType, SizeType> =
        remaining.keys().map(|k| (k.clone(), helpers::compute_size_by_dict(k.iter(), size_dict))).collect();

    // Find initial candidate contractions.
    let mut queue = BinaryHeap::new();
    for dim_keys in dim_to_keys.values() {
        let mut dim_keys_list = dim_keys.iter().cloned().collect_vec();
        dim_keys_list.sort_by_key(|k| remaining[k]);
        for i in 0..dim_keys_list.len().saturating_sub(1) {
            let k1 = &dim_keys_list[i];
            let k2s_guess = &dim_keys_list[i + 1..];
            push_candidate(
                &output,
                size_dict,
                &remaining,
                &footprints,
                &dim_ref_counts,
                k1,
                k2s_guess,
                &mut queue,
                push_all,
                cost_fn,
            );
        }
    }

    // Greedily contract pairs of tensors.
    while !queue.is_empty() {
        let Some(con) = choose_fn(&mut queue, &remaining) else {
            continue; // allow choose_fn to flag all candidates obsolete
        };
        let GreedyContractionType { k1, k2, k12, .. } = con;

        let ssa_id1 = remaining.remove(&k1).unwrap();
        let ssa_id2 = remaining.remove(&k2).unwrap();

        for dim in &k1 - &output {
            dim_to_keys.get_mut(&dim).unwrap().remove(&k1);
        }
        for dim in &k2 - &output {
            dim_to_keys.get_mut(&dim).unwrap().remove(&k2);
        }

        ssa_path.push(vec![ssa_id1, ssa_id2]);

        if remaining.contains_key(&k12) {
            ssa_path.push(vec![remaining[&k12], ssa_ids]);
            ssa_ids += 1;
        } else {
            for dim in &k12 - &output {
                dim_to_keys.get_mut(&dim).unwrap().insert(k12.clone());
            }
        }
        remaining.insert(k12.clone(), ssa_ids);
        ssa_ids += 1;

        let updated_dims = &(&k1 | &k2) - &output;
        update_ref_counts(&dim_to_keys, &mut dim_ref_counts, &updated_dims, &output);

        footprints.insert(k12.clone(), helpers::compute_size_by_dict(k12.iter(), size_dict));

        // Find new candidate contractions.
        let k1 = k12;
        let k2s: BTreeSet<ArrayIndexType> =
            (&k1 - &output).into_iter().flat_map(|dim| dim_to_keys[&dim].clone()).filter(|k| k != &k1).collect();

        if !k2s.is_empty() {
            push_candidate(
                &output,
                size_dict,
                &remaining,
                &footprints,
                &dim_ref_counts,
                &k1,
                &k2s.into_iter().collect_vec(),
                &mut queue,
                push_all,
                cost_fn,
            );
        }
    }

    // Greedily compute pairwise outer products.
    #[derive(Clone, Debug, PartialEq, PartialOrd)]
    struct FinalEntry {
        size: SizeType,
        ssa_id: usize,
        key: ArrayIndexType,
    }
    impl Eq for FinalEntry {}
    #[allow(clippy::derive_ord_xor_partial_ord)]
    impl Ord for FinalEntry {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    // Greedily compute pairwise outer products.
    let mut final_queue: BinaryHeap<Reverse<FinalEntry>> = remaining
        .into_iter()
        .map(|(key, ssa_id)| {
            let size = helpers::compute_size_by_dict((&key & &output).iter(), size_dict);
            Reverse(FinalEntry { size, ssa_id, key })
        })
        .collect();

    let Some(Reverse(FinalEntry { ssa_id: ssa_id1, key: k1, .. })) = final_queue.pop() else {
        return ssa_path;
    };

    let mut current_id = ssa_id1;
    let mut current_k = k1;

    while let Some(Reverse(FinalEntry { ssa_id: ssa_id2, key: k2, .. })) = final_queue.pop() {
        ssa_path.push(vec![current_id.min(ssa_id2), current_id.max(ssa_id2)]);
        let k12: ArrayIndexType = &(&current_k | &k2) & &output;
        let cost = helpers::compute_size_by_dict(k12.iter(), size_dict);
        let new_ssa_id = ssa_ids;
        ssa_ids += 1;

        final_queue.push(Reverse(FinalEntry { size: cost, ssa_id: new_ssa_id, key: k12.clone() }));
        let Reverse(FinalEntry { ssa_id: new_id, key: new_k, .. }) = final_queue.pop().unwrap();
        current_id = new_id;
        current_k = new_k;
    }

    ssa_path
}

/// Finds the path by a three stage greedy algorithm.
///
/// 1. Eagerly compute Hadamard products.
/// 2. Greedily compute contractions to maximize `removed_size`.
/// 3. Greedily compute outer products.
///
/// This algorithm scales quadratically with respect to the maximum number of elements sharing a
/// common dim.
///
/// # Parameters
///
/// - **inputs** - List of sets that represent the lhs side of the einsum subscript
/// - **output** - Set that represents the rhs side of the overall einsum subscript
/// - **size_dict** - Dictionary of index sizes
/// - **memory_limit** - The maximum number of elements in a temporary array
/// - **choose_fn** - A function that chooses which contraction to perform from the queue
/// - **cost_fn** - A function that assigns a potential contraction a cost.
///
/// # Returns
///
/// - **path** - The contraction order (a list of tuples of ints).
pub fn greedy(
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    memory_limit: Option<SizeType>,
    choose_fn: Option<&mut GreedyChooseFn>,
    cost_fn: Option<paths::CostFn>,
) -> Result<PathType, String> {
    if memory_limit.is_some() {
        let mut branch_optimizer = paths::branch_bound::BranchBound::from("branch-1");
        return branch_optimizer.optimize_path(inputs, output, size_dict, memory_limit);
    }

    let ssa_path = ssa_greedy_optimize(inputs, output, size_dict, choose_fn, cost_fn);
    Ok(paths::util::ssa_to_linear(&ssa_path))
}

#[derive(Default)]
pub struct Greedy {
    cost_fn: Option<paths::CostFn>,
    choose_fn: Option<GreedyChooseFn>,
}

impl std::fmt::Debug for Greedy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Greedy").field("cost_fn", &self.cost_fn).field("choose_fn", &self.choose_fn.is_some()).finish()
    }
}

impl PathOptimizer for Greedy {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        greedy(inputs, output, size_dict, memory_limit, self.choose_fn.as_mut(), self.cost_fn)
    }
}
