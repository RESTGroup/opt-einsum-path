// src/paths/random.rs
use crate::paths::branch_bound::*;
use crate::paths::greedy::*;
use crate::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

/// A contraction 'chooser' that weights possible contractions using a
/// Boltzmann distribution. Explicitly, given costs `c_i` (with `c_0` the
/// smallest), the relative weights, `w_i`, are computed as:
///
/// w_i = exp( -(c_i - c_0) / temperature)
///
/// Additionally, if `rel_temperature` is set, scale `temperature` by
/// `abs(c_0)` to account for likely fluctuating cost magnitudes during the
/// course of a contraction.
///
/// # Parameters
///
/// * `queue` - The heapified list of candidate contractions.
/// * `remaining` - Mapping of remaining inputs' indices to the ssa id.
/// * `rng` - Random number generator.
/// * `nbranch` - How many potential paths to calculate probability for and choose from at each
///   step.
/// * `temperature` - When choosing a possible contraction, its relative probability will be
///   proportional to `exp(-cost / temperature)`. Thus the larger `temperature` is, the further
///   random paths will stray from the normal 'greedy' path. Conversely, if set to zero, only paths
///   with exactly the same cost as the best at each step will be explored.
/// * `rel_temperature` - Whether to normalize the `temperature` at each step to the scale of the
///   best cost. This is generally beneficial as the magnitude of costs can vary significantly
///   throughout a contraction.
///
/// # Returns
///
/// `Option<GreedyContractionType>` where Some contains the chosen contraction
pub fn thermal_chooser(
    queue: &mut BinaryHeap<GreedyContractionType>,
    remaining: &BTreeMap<ArrayIndexType, usize>,
    rng: &mut StdRng,
    nbranch: usize,
    temperature: f64,
    rel_temperature: bool,
) -> Option<GreedyContractionType> {
    let mut choices = Vec::new();
    let mut n = 0;

    // Extract up to nbranch valid choices from the queue
    while n < nbranch && !queue.is_empty() {
        if let Some(candidate) = queue.pop() {
            if remaining.contains_key(&candidate.k1) && remaining.contains_key(&candidate.k2) {
                choices.push(candidate);
                n += 1;
            }
        }
    }

    if choices.is_empty() {
        return None;
    }

    if choices.len() == 1 {
        return Some(choices.remove(0));
    }

    // Extract costs from choices
    let costs: Vec<f64> = choices.iter().map(|c| c.cost.0.cost.to_f64().unwrap()).collect();

    let cmin = costs.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));

    // Adjust by the overall scale to account for fluctuating absolute costs
    let effective_temperature = if rel_temperature { temperature * cmin.abs().max(1.0) } else { temperature };

    // Compute relative probability for each potential contraction
    let energies: Vec<f64> = if effective_temperature == 0.0 {
        costs.iter().map(|&c| if c == cmin { 1.0 } else { 0.0 }).collect()
    } else {
        costs.iter().map(|&c| (-(c - cmin) / effective_temperature).exp()).collect()
    };

    // Randomly choose a contraction based on energies
    let chosen_index = if energies.iter().sum::<f64>() > 0.0 {
        let mut cumulative = 0.0;
        let total: f64 = energies.iter().sum();
        let rand_val: f64 = rng.random_range(0.0..total);

        for (i, &energy) in energies.iter().enumerate() {
            cumulative += energy;
            if cumulative >= rand_val {
                return Some(choices.remove(i));
            }
        }
        0 // fallback
    } else {
        0
    };

    // Put the other choices back in the heap
    for (i, choice) in choices.clone().into_iter().enumerate() {
        if i != chosen_index {
            queue.push(choice);
        }
    }

    Some(choices.remove(chosen_index))
}

/// Compute the flops and max size of an ssa path.
pub fn ssa_path_compute_cost(
    ssa_path: &PathType,
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
) -> (SizeType, SizeType) {
    let mut inputs = inputs.iter().map(|x| (*x).clone()).collect_vec();
    let mut remaining: BTreeSet<usize> = (0..inputs.len()).collect();
    let mut total_cost = SizeType::zero();
    let mut max_size = SizeType::zero();

    for contraction in ssa_path {
        if contraction.len() < 2 {
            continue;
        }

        let i = contraction[0];
        let j = contraction[1];

        let inputs_ref = inputs.iter().collect_vec();
        let (k12, flops12) =
            paths::util::calc_k12_flops(&inputs_ref, output, &remaining.iter().cloned().collect_vec(), i, j, size_dict);

        let size12 = helpers::compute_size_by_dict(k12.iter(), size_dict);
        total_cost += flops12;
        max_size = max_size.max(size12);

        remaining.remove(&i);
        remaining.remove(&j);
        remaining.insert(inputs.len());
        inputs.push(k12);
    }

    (total_cost, max_size)
}

/// Configuration for random greedy optimization
#[derive(Debug, Clone)]
pub struct RandomGreedyConfig {
    pub max_repeats: usize,
    pub max_time: Option<Duration>,
    pub minimize: MinimizeStrategy,
    pub cost_fn: &'static str,
    pub temperature: f64,
    pub rel_temperature: bool,
    pub nbranch: usize,
}

impl Default for RandomGreedyConfig {
    fn default() -> Self {
        Self {
            max_repeats: 32,
            max_time: None,
            minimize: MinimizeStrategy::FlopsFirst,
            cost_fn: "memory-removed-jitter",
            temperature: 1.0,
            rel_temperature: true,
            nbranch: 8,
        }
    }
}

/// Random greedy path optimizer
#[derive(Debug, Clone)]
pub struct RandomGreedy {
    pub config: RandomGreedyConfig,
    pub best_flops: SizeType,
    pub best_size: SizeType,
    pub best_ssa_path: Option<PathType>,
    pub costs: Vec<SizeType>,
    pub sizes: Vec<SizeType>,
    pub repeats_start: usize,
}

impl Default for RandomGreedy {
    fn default() -> Self {
        Self {
            config: RandomGreedyConfig::default(),
            best_flops: SizeType::MAX,
            best_size: SizeType::MAX,
            best_ssa_path: None,
            costs: Vec::new(),
            sizes: Vec::new(),
            repeats_start: 0,
        }
    }
}

impl RandomGreedy {
    /// Create a new RandomGreedy optimizer with custom configuration
    pub fn new(config: RandomGreedyConfig) -> Self {
        Self { config, ..Default::default() }
    }

    /// Get the best path found so far
    pub fn path(&self) -> PathType {
        self.best_ssa_path.as_ref().map_or_else(Vec::new, |p| paths::util::ssa_to_linear(p))
    }

    /// Run a single trial of greedy optimization
    fn run_trial(
        config: &RandomGreedyConfig,
        r: usize,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
    ) -> (PathType, SizeType, SizeType) {
        let mut rng = StdRng::seed_from_u64(r as u64);
        // For the first trial, use standard greedy approach
        let nbranch = config.nbranch;
        let temperature = config.temperature;
        let rel_temperature = config.rel_temperature;
        let thermal_chooser_fn: GreedyChooseFn = Box::new({
            move |queue, remaining| thermal_chooser(queue, remaining, &mut rng, nbranch, temperature, rel_temperature)
        });
        let mut choose_fn = if r == 0 { Some(thermal_chooser_fn) } else { None };

        let cost_fn = match config.cost_fn {
            "memory-removed-jitter" => Some(paths::util::memory_removed(true)),
            _ => Some(paths::util::memory_removed(false)),
        };

        let ssa_path = paths::greedy::ssa_greedy_optimize(inputs, output, size_dict, choose_fn.as_mut(), cost_fn);

        let (cost, size) = ssa_path_compute_cost(&ssa_path, inputs, output, size_dict);

        (ssa_path, cost, size)
    }
}

impl PathOptimizer for RandomGreedy {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        output: &ArrayIndexType,
        size_dict: &SizeDictType,
        memory_limit: Option<SizeType>,
    ) -> Result<PathType, String> {
        // Handle memory limit by falling back to branch bound
        if memory_limit.is_some() {
            let mut branch_optimizer = paths::branch_bound::BranchBound::from("branch-1");
            return branch_optimizer.optimize_path(inputs, output, size_dict, memory_limit);
        }

        let start_time = Instant::now();
        let better_fn = paths::branch_bound::get_better_fn(self.config.minimize);

        let r_start = self.repeats_start + self.costs.len();
        let r_end = r_start + self.config.max_repeats;

        #[cfg(feature = "par_rand")]
        use rayon::prelude::*;
        #[cfg(feature = "par_rand")]
        let r_iter = (r_start..r_end).into_par_iter();
        #[cfg(not(feature = "par_rand"))]
        let r_iter = r_start..r_end;

        let trials: Vec<_> = r_iter
            .map(|r| {
                // Check if we've run out of time
                if self.config.max_time.is_some_and(|max_time| start_time.elapsed() > max_time) {
                    None
                } else {
                    Some(RandomGreedy::run_trial(&self.config, r, inputs, output, size_dict))
                }
            })
            .collect();

        for (ssa_path, cost, size) in trials.into_iter().flatten() {
            // Keep track of all costs and sizes
            self.costs.push(cost);
            self.sizes.push(size);

            // Check if we have found a new best
            let found_new_best = better_fn(
                cost.to_f64().unwrap(),
                size.to_f64().unwrap(),
                self.best_flops.to_f64().unwrap(),
                self.best_size.to_f64().unwrap(),
            );

            if found_new_best {
                self.best_flops = cost;
                self.best_size = size;
                self.best_ssa_path = Some(ssa_path);
            }
        }

        Ok(self.path())
    }
}

/// Convenience function for random greedy optimization
pub fn random_greedy(
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    memory_limit: Option<SizeType>,
    config: RandomGreedyConfig,
) -> Result<PathType, String> {
    let mut optimizer = RandomGreedy::new(config);
    optimizer.optimize_path(inputs, output, size_dict, memory_limit)
}

/// Pre-configured random greedy with 128 repeats
pub fn random_greedy_128(
    inputs: &[&ArrayIndexType],
    output: &ArrayIndexType,
    size_dict: &SizeDictType,
    memory_limit: Option<SizeType>,
) -> Result<PathType, String> {
    let config = RandomGreedyConfig { max_repeats: 128, ..RandomGreedyConfig::default() };
    random_greedy(inputs, output, size_dict, memory_limit, config)
}

impl From<&str> for RandomGreedy {
    fn from(s: &str) -> Self {
        let s = s.trim().replace(['_', ' '], "-").to_lowercase();
        assert!(s.starts_with("random-greedy"), "RandomGreedy must start with 'random-greedy'");
        let v = s.strip_prefix("random-greedy").unwrap();
        if v.is_empty() {
            RandomGreedy::default()
        } else {
            let max_repeats = v.replace("-", "").parse::<usize>().unwrap_or_else(|_| {
                panic!("Invalid RandomGreedy configuration: {s}. Expected format: 'random-greedy-<max_repeats>'")
            });
            let config = RandomGreedyConfig { max_repeats, ..RandomGreedyConfig::default() };
            RandomGreedy::new(config)
        }
    }
}
