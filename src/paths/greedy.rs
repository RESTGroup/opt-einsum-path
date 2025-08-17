use crate::*;
use itertools::Itertools;
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct GreedyCostType {
    pub cost: SizeType,
    pub id1: usize,
    pub id2: usize,
}

impl Eq for GreedyCostType {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for GreedyCostType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct GreedyContractionType {
    cost: Reverse<GreedyCostType>,
    k1: ArrayIndexType,
    k2: ArrayIndexType,
    k12: ArrayIndexType,
}

fn get_candidate(
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    remaining: &BTreeMap<ArrayIndexType, usize>,
    footprints: &BTreeMap<ArrayIndexType, SizeType>,
    dim_ref_counts: &BTreeMap<usize, BTreeSet<char>>,
    k1: &ArrayIndexType,
    k2: &ArrayIndexType,
    cost_fn: fn(SizeType, SizeType, SizeType, &ArrayIndexType, &ArrayIndexType, &ArrayIndexType) -> SizeType,
) -> GreedyContractionType {
    let either = k1 | k2;
    let two = k1 & k2;
    let one = &either - &two;

    // k12 = (either & output) | (two & dim_ref_counts[3]) | (one & dim_ref_counts[2])
    let part1 = either.intersection(output);
    let part2 = two.intersection(&dim_ref_counts[&3]);
    let part3 = one.intersection(&dim_ref_counts[&2]);
    let k12: ArrayIndexType = part1.chain(part2).chain(part3).cloned().collect();

    let size12 = helpers::compute_size_by_dict(k12.iter(), size_dict);
    let footprint1 = footprints[k1];
    let footprint2 = footprints[k2];
    let cost = cost_fn(size12, footprint1, footprint2, &k12, k1, k2);

    let id1 = remaining[k1];
    let id2 = remaining[k2];
    let (k1, id1, k2, id2) =
        if id1 > id2 { (k1.clone(), id1, k2.clone(), id2) } else { (k2.clone(), id2, k1.clone(), id1) };

    GreedyContractionType { cost: Reverse(GreedyCostType { cost, id1, id2 }), k1, k2, k12 }
}

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
    cost_fn: fn(SizeType, SizeType, SizeType, &ArrayIndexType, &ArrayIndexType, &ArrayIndexType) -> SizeType,
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
                dim_ref_counts.entry(2).or_default().remove(dim);
                dim_ref_counts.entry(3).or_default().remove(dim);
            },
            2 => {
                dim_ref_counts.entry(2).or_default().insert(*dim);
                dim_ref_counts.entry(3).or_default().remove(dim);
            },
            _ => {
                dim_ref_counts.entry(2).or_default().insert(*dim);
                dim_ref_counts.entry(3).or_default().insert(*dim);
            },
        }
    }
}

fn simple_chooser(
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

fn ssa_greedy_optimize(
    inputs: &[ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    choose_fn: Option<
        fn(&mut BinaryHeap<GreedyContractionType>, &BTreeMap<ArrayIndexType, usize>) -> Option<GreedyContractionType>,
    >,
    cost_fn: fn(SizeType, SizeType, SizeType, &ArrayIndexType, &ArrayIndexType, &ArrayIndexType) -> SizeType,
) -> PathType {
    if inputs.len() == 1 {
        return vec![vec![0]];
    }

    let push_all = choose_fn.is_none();
    let choose_fn = choose_fn.unwrap_or(simple_chooser);

    let fs_inputs: Vec<ArrayIndexType> = inputs.to_vec();
    let common_dims: ArrayIndexType =
        fs_inputs.iter().skip(1).fold(fs_inputs[0].clone(), |acc, s| acc.intersection(s).cloned().collect());
    let output: ArrayIndexType = output.union(&common_dims).cloned().collect();

    let mut remaining = BTreeMap::new();
    let mut ssa_ids = fs_inputs.len();
    let mut ssa_path = Vec::new();

    for (ssa_id, key) in fs_inputs.into_iter().enumerate() {
        if let Some(&existing_id) = remaining.get(&key) {
            ssa_path.push(vec![existing_id, ssa_id]);
            remaining.insert(key, ssa_ids);
            ssa_ids += 1;
        } else {
            remaining.insert(key, ssa_id);
        }
    }

    let mut dim_to_keys: BTreeMap<char, BTreeSet<ArrayIndexType>> = BTreeMap::new();
    for key in remaining.keys() {
        key.difference(&output).for_each(|&dim| {
            dim_to_keys.entry(dim).or_default().insert(key.clone());
        });
    }

    let mut dim_ref_counts: BTreeMap<usize, BTreeSet<char>> =
        BTreeMap::from([(2, BTreeSet::new()), (3, BTreeSet::new())]);
    for (dim, keys) in &dim_to_keys {
        let count = keys.len();
        if count >= 2 {
            dim_ref_counts.get_mut(&2).unwrap().insert(*dim);
        }
        if count >= 3 {
            dim_ref_counts.get_mut(&3).unwrap().insert(*dim);
        }
    }
    output.iter().for_each(|dim| {
        dim_ref_counts.get_mut(&2).unwrap().remove(dim);
        dim_ref_counts.get_mut(&3).unwrap().remove(dim);
    });

    let mut footprints: BTreeMap<ArrayIndexType, SizeType> =
        remaining.keys().map(|k| (k.clone(), helpers::compute_size_by_dict(k.iter(), size_dict))).collect();

    let mut queue = BinaryHeap::new();
    for dim_keys in dim_to_keys.values() {
        let mut dim_keys_list: Vec<ArrayIndexType> = dim_keys.iter().cloned().collect();
        dim_keys_list.sort_by_key(|k| remaining[k]);
        for i in 0..dim_keys_list.len().saturating_sub(1) {
            let k1 = &dim_keys_list[i];
            let k2s_guess = &dim_keys_list[i + 1..];
            println!("push_candidate k1 {k1:?} k2s_guess {k2s_guess:?}");
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
    println!("queue: {queue:?}");

    while !queue.is_empty() {
        let Some(con) = choose_fn(&mut queue, &remaining) else {
            continue;
        };
        let GreedyContractionType { cost, k1, k2, k12 } = con;
        println!("Chose k1 {k1:?} k2 {k2:?} -> k12 {k12:?}, cost {cost:?}");

        let ssa_id2 = remaining.remove(&k2).unwrap();
        let ssa_id1 = remaining.remove(&k1).unwrap();

        k1.difference(&output).for_each(|&dim| {
            if let Some(keys) = dim_to_keys.get_mut(&dim) {
                keys.remove(&k1);
            }
        });
        k2.difference(&output).for_each(|&dim| {
            if let Some(keys) = dim_to_keys.get_mut(&dim) {
                keys.remove(&k2);
            }
        });

        ssa_path.push(vec![ssa_id1, ssa_id2]);
        println!("Pushed ssa path: {:?}", vec![ssa_id1, ssa_id2]);

        if remaining.contains_key(&k12) {
            let existing_id = remaining[&k12];
            ssa_path.push(vec![existing_id, ssa_ids]);
            ssa_ids += 1;
        } else {
            k12.difference(&output).for_each(|&dim| {
                dim_to_keys.entry(dim).or_default().insert(k12.clone());
            });
        }
        remaining.insert(k12.clone(), ssa_ids);
        ssa_ids += 1;

        let updated_dims: ArrayIndexType = &(&k1 | &k2) - &output;
        update_ref_counts(&dim_to_keys, &mut dim_ref_counts, &updated_dims, &output);

        footprints.insert(k12.clone(), helpers::compute_size_by_dict(k12.iter(), size_dict));

        let k1 = k12;
        let k2s: BTreeSet<ArrayIndexType> = k1
            .difference(&output)
            .flat_map(|dim| dim_to_keys.get(dim).cloned().unwrap_or_default())
            .filter(|k| k != &k1)
            .collect();

        if !k2s.is_empty() {
            println!("push_candidate k1 {k1:?} k2s {k2s:?}");
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

pub fn greedy(
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    memory_limit: Option<SizeType>,
    choose_fn: Option<
        fn(&mut BinaryHeap<GreedyContractionType>, &BTreeMap<ArrayIndexType, usize>) -> Option<GreedyContractionType>,
    >,
    cost_fn: fn(SizeType, SizeType, SizeType, &ArrayIndexType, &ArrayIndexType, &ArrayIndexType) -> SizeType,
) -> PathType {
    if memory_limit.is_some() {
        let mut branch_optimizer = paths::branch_bound::BranchBound::from("branch-1");
        return branch_optimizer.optimize_path(inputs, output, size_dict, memory_limit);
    }

    let inputs_owned: Vec<ArrayIndexType> = inputs.iter().cloned().cloned().collect();
    let ssa_path = ssa_greedy_optimize(&inputs_owned, output, size_dict, choose_fn, cost_fn);
    paths::util::ssa_to_linear(&ssa_path)
}

pub fn memory_removed(
    size12: SizeType,
    size1: SizeType,
    size2: SizeType,
    _k12: &ArrayIndexType,
    _k1: &ArrayIndexType,
    _k2: &ArrayIndexType,
) -> SizeType {
    size12 - size1 - size2
}

#[derive(Debug, Clone)]
pub struct Greedy {
    cost_fn: fn(SizeType, SizeType, SizeType, &ArrayIndexType, &ArrayIndexType, &ArrayIndexType) -> SizeType,
}

impl Default for Greedy {
    fn default() -> Self {
        Self { cost_fn: memory_removed }
    }
}

impl Greedy {
    pub fn new(
        cost_fn: fn(SizeType, SizeType, SizeType, &ArrayIndexType, &ArrayIndexType, &ArrayIndexType) -> SizeType,
    ) -> Self {
        Self { cost_fn }
    }
}

impl PathOptimizer for Greedy {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> PathType {
        greedy(inputs, output, size_dict, memory_limit, None, self.cost_fn)
    }
}
