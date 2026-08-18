//! Deterministic agent frameworks (master plan II.5): Physarum swarms,
//! batch-synchronous diffusion-limited aggregation, and dielectric-breakdown
//! streamer growth.
//!
//! Determinism scheme (recorded deviation from the spec's per-step SHA3):
//! every agent gets a u64 base seed drawn once from a SHA3-forked stream;
//! per-step randomness is a stateless splitmix64 mix of (base, step, salt).
//! This is order-free across threads like the spec's construction, at a
//! fraction of the cost.

use crate::sim::Sha3RandomByteStream;
use rayon::prelude::*;

/// Stateless splitmix64 mix.
#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Order-free per-agent random value in `[0, 1)`.
#[inline]
fn agent_unit(base: u64, step: u64, salt: u64) -> f64 {
    let mixed = splitmix64(base ^ step.wrapping_mul(0xA24B_AED4_963E_E407) ^ salt);
    (mixed >> 11) as f64 / (1u64 << 53) as f64
}

/// Draw one SHA3-derived u64 base seed per agent.
#[must_use]
pub fn agent_seeds(rng: &mut Sha3RandomByteStream, count: usize) -> Vec<u64> {
    (0..count).map(|_| rng.next_u64()).collect()
}

/// Clamped nearest-cell sample of a scalar grid.
#[inline]
fn sample_grid(grid: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let col = (x as usize).min(width - 1);
    let row = (y as usize).min(height - 1);
    grid[row * width + col]
}

/// Parameters for a Physarum swarm.
#[derive(Clone, Copy)]
pub struct PhysarumParams {
    /// Sensor distance in grid cells.
    pub sense_distance: f32,
    /// Sensor angular offset (radians).
    pub sense_angle: f32,
    /// Turn applied toward the strongest sensor (radians).
    pub turn_angle: f32,
    /// Move distance per tick (grid cells).
    pub step_length: f32,
    /// Trail deposit per agent per tick.
    pub deposit: f32,
    /// Trail retention per tick after diffusion (0.94 keeps 94%).
    pub decay: f32,
    /// Uniform heading jitter magnitude (radians).
    pub jitter: f32,
    /// Blend weight of the food grid against the trail when sensing.
    pub food_weight: f32,
}

/// A Physarum colony over a trail grid (structure-of-arrays).
pub struct PhysarumSwarm {
    x: Vec<f32>,
    y: Vec<f32>,
    heading: Vec<f32>,
    seeds: Vec<u64>,
    width: usize,
    height: usize,
}

impl PhysarumSwarm {
    /// Seed agents proportionally to a weight grid (cumulative sampling),
    /// with random headings.
    #[must_use]
    pub fn seed_weighted(
        count: usize,
        width: usize,
        height: usize,
        weights: &[f32],
        rng: &mut Sha3RandomByteStream,
    ) -> Self {
        debug_assert_eq!(weights.len(), width * height);
        let mut cumulative: Vec<f64> = Vec::with_capacity(weights.len());
        let mut total = 0.0f64;
        for &weight in weights {
            total += f64::from(weight.max(0.0));
            cumulative.push(total);
        }
        let mut x = Vec::with_capacity(count);
        let mut y = Vec::with_capacity(count);
        let mut heading = Vec::with_capacity(count);
        for _ in 0..count {
            let pick = rng.next_f64() * total.max(1e-12);
            let index = cumulative.partition_point(|&value| value < pick).min(weights.len() - 1);
            let col = index % width;
            let row = index / width;
            x.push(col as f32 + rng.next_f64() as f32);
            y.push(row as f32 + rng.next_f64() as f32);
            heading.push((rng.next_f64() * std::f64::consts::TAU) as f32);
        }
        let seeds = agent_seeds(rng, count);
        Self { x, y, heading, seeds, width, height }
    }

    /// Number of agents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Whether the swarm is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// One tick: sense (trail blended with food) -> rotate -> move in
    /// parallel, filling `deposits` (cleared first) for a serial
    /// [`Self::apply_deposits`] pass. Diffusion/decay is a separate call.
    pub fn tick(
        &mut self,
        trail: &[f32],
        food: &[f32],
        params: &PhysarumParams,
        step: u64,
        deposits: &mut Vec<(u32, f32)>,
    ) {
        let width = self.width;
        let height = self.height;
        let food_weight = params.food_weight;
        let sense = |x: f32, y: f32| -> f32 {
            let t = sample_grid(trail, width, height, x, y);
            let f = sample_grid(food, width, height, x, y);
            t * (1.0 - food_weight) + f * food_weight
        };

        let max_x = width as f32 - 1.001;
        let max_y = height as f32 - 1.001;
        deposits.clear();
        deposits.par_extend(
            self.x
                .par_iter_mut()
                .zip(self.y.par_iter_mut())
                .zip(self.heading.par_iter_mut())
                .zip(self.seeds.par_iter())
                .map(|(((x, y), heading), &seed)| {
                    let ahead = *heading;
                    let sample_at = |angle: f32| {
                        sense(
                            (*x + params.sense_distance * angle.cos()).clamp(0.0, max_x),
                            (*y + params.sense_distance * angle.sin()).clamp(0.0, max_y),
                        )
                    };
                    let forward = sample_at(ahead);
                    let left = sample_at(ahead - params.sense_angle);
                    let right = sample_at(ahead + params.sense_angle);

                    if forward >= left && forward >= right {
                        // keep heading
                    } else if left > right {
                        *heading -= params.turn_angle;
                    } else if right > left {
                        *heading += params.turn_angle;
                    } else {
                        let coin = agent_unit(seed, step, 1);
                        *heading += if coin < 0.5 { -params.turn_angle } else { params.turn_angle };
                    }
                    let jitter = (agent_unit(seed, step, 2) - 0.5) as f32 * 2.0 * params.jitter;
                    *heading += jitter;

                    let mut next_x = *x + params.step_length * heading.cos();
                    let mut next_y = *y + params.step_length * heading.sin();
                    if next_x < 0.0 || next_x > max_x || next_y < 0.0 || next_y > max_y {
                        // Bounce: re-aim into the interior deterministically.
                        next_x = next_x.clamp(0.0, max_x);
                        next_y = next_y.clamp(0.0, max_y);
                        *heading = (agent_unit(seed, step, 3) * std::f64::consts::TAU) as f32;
                    }
                    *x = next_x;
                    *y = next_y;
                    let index = (next_y as usize).min(height - 1) * width
                        + (next_x as usize).min(width - 1);
                    (index as u32, params.deposit)
                }),
        );
    }

    /// Apply a deposit list to the trail grid (serial, deterministic).
    pub fn apply_deposits(trail: &mut [f32], deposits: &[(u32, f32)]) {
        for &(index, amount) in deposits {
            trail[index as usize] += amount;
        }
    }
}

/// 3x3 mean diffusion followed by multiplicative decay (ping-pong buffers).
pub fn diffuse_decay(
    trail: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
    width: usize,
    height: usize,
    decay: f32,
) {
    scratch.resize(trail.len(), 0.0);
    scratch.par_chunks_mut(width).enumerate().for_each(|(row, out)| {
        for (col, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for dy in -1i64..=1 {
                let sample_row = (row as i64 + dy).clamp(0, height as i64 - 1) as usize;
                for dx in -1i64..=1 {
                    let sample_col = (col as i64 + dx).clamp(0, width as i64 - 1) as usize;
                    sum += trail[sample_row * width + sample_col];
                }
            }
            *slot = sum / 9.0 * decay;
        }
    });
    std::mem::swap(trail, scratch);
}

/// Parameters for diffusion-limited aggregation.
#[derive(Clone, Copy)]
pub struct DlaParams {
    /// Base sticking probability.
    pub stick_base: f32,
    /// Sticking probability span multiplied by the local field value.
    pub stick_span: f32,
    /// Six-fold directional bias amplitude in [0, 1].
    pub bias_strength: f32,
    /// Walker steps per synchronous batch.
    pub batch_steps: usize,
    /// Concurrent walkers per batch.
    pub concurrent: usize,
}

/// A DLA aggregate over a grid; `age[cell]` is the 1-based stick order.
pub struct DlaGrid {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// 0 = empty; otherwise the stick order (1-based).
    pub age: Vec<u32>,
    next_age: u32,
    bbox: (f32, f32, f32, f32),
}

/// One live walker.
#[derive(Clone, Copy)]
struct Walker {
    x: f32,
    y: f32,
    id: u64,
    /// Total steps walked (walkers retire after a lifetime budget).
    steps_walked: u32,
}

impl DlaGrid {
    /// Create an aggregate seeded with nucleation sites (age 1, 2, ...).
    #[must_use]
    pub fn new(width: usize, height: usize, nuclei: &[(usize, usize)]) -> Self {
        let mut age = vec![0u32; width * height];
        let mut next_age = 1u32;
        let mut bbox = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &(col, row) in nuclei {
            let index = row.min(height - 1) * width + col.min(width - 1);
            if age[index] == 0 {
                age[index] = next_age;
                next_age += 1;
                bbox.0 = bbox.0.min(col as f32);
                bbox.1 = bbox.1.min(row as f32);
                bbox.2 = bbox.2.max(col as f32);
                bbox.3 = bbox.3.max(row as f32);
            }
        }
        Self { width, height, age, next_age, bbox }
    }

    /// Number of stuck cells (including nuclei).
    #[must_use]
    pub fn stuck_count(&self) -> u32 {
        self.next_age - 1
    }

    /// Whether any 8-neighbor of (col, row) is stuck.
    #[inline]
    fn adjacent_stuck(&self, col: usize, row: usize) -> bool {
        for dy in -1i64..=1 {
            let sample_row = row as i64 + dy;
            if sample_row < 0 || sample_row >= self.height as i64 {
                continue;
            }
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let sample_col = col as i64 + dx;
                if sample_col < 0 || sample_col >= self.width as i64 {
                    continue;
                }
                if self.age[sample_row as usize * self.width + sample_col as usize] != 0 {
                    return true;
                }
            }
        }
        false
    }

    /// Spawn position on the birth ring around the cluster bounding box.
    fn spawn(&self, id: u64, batch: u64) -> Walker {
        let center_x = 0.5 * (self.bbox.0 + self.bbox.2);
        let center_y = 0.5 * (self.bbox.1 + self.bbox.3);
        let half_w = 0.5 * (self.bbox.2 - self.bbox.0);
        let half_h = 0.5 * (self.bbox.3 - self.bbox.1);
        let radius = (half_w.hypot(half_h) + 12.0).min(0.75 * (self.width.max(self.height)) as f32);
        let angle = agent_unit(id, batch, 11) * std::f64::consts::TAU;
        Walker {
            x: (center_x + radius * angle.cos() as f32).clamp(1.0, self.width as f32 - 2.0),
            y: (center_y + radius * angle.sin() as f32).clamp(1.0, self.height as f32 - 2.0),
            id,
            steps_walked: 0,
        }
    }

    /// Grow the aggregate with a total walker budget over a field grid and
    /// an orientation grid (radians; six-fold comb bias). Deterministic:
    /// batch-synchronous walks with stick proposals applied in walker order.
    /// Terminates when the budget drains, walkers exhaust their lifetime
    /// step budget, or the aggregate covers half the grid.
    pub fn grow(
        &mut self,
        walker_budget: usize,
        field: &[f32],
        orientation: &[f32],
        params: &DlaParams,
        seed_rng: &mut Sha3RandomByteStream,
    ) {
        let base_seed = seed_rng.next_u64();
        let mut spawned = 0usize;
        let mut batch_index = 0u64;
        let mut pool: Vec<Walker> = Vec::with_capacity(params.concurrent);
        // Lifetime budget: enough to cross the grid several times.
        let max_walk = ((self.width + self.height) * 6) as u32;
        let fill_limit = (self.width * self.height / 2) as u32;

        while (spawned < walker_budget || !pool.is_empty()) && self.stuck_count() < fill_limit {
            while pool.len() < params.concurrent && spawned < walker_budget {
                let id = base_seed ^ (spawned as u64).wrapping_mul(0xD134_2543_DE82_EF95);
                pool.push(self.spawn(id, batch_index));
                spawned += 1;
            }
            batch_index += 1;

            let kill_margin = {
                let half_w = 0.5 * (self.bbox.2 - self.bbox.0);
                let half_h = 0.5 * (self.bbox.3 - self.bbox.1);
                (half_w.hypot(half_h) + 12.0) * 2.0 + 16.0
            };
            let center = (0.5 * (self.bbox.0 + self.bbox.2), 0.5 * (self.bbox.1 + self.bbox.3));

            // Parallel walk against a frozen snapshot of the aggregate.
            // Outcome per walker: (state, stick proposal (cell, id), killed).
            type BatchOutcome = (Walker, Option<(u32, u64)>, bool);
            let grid = &*self;
            let results: Vec<BatchOutcome> = pool
                .par_iter()
                .map(|&walker| {
                    let mut current = walker;
                    for step in 0..params.batch_steps {
                        let salt = batch_index.wrapping_mul(1_000_003) + step as u64;
                        let angle = agent_unit(current.id, salt, 21) * std::f64::consts::TAU;
                        let next_x =
                            (current.x + angle.cos() as f32).clamp(1.0, grid.width as f32 - 2.0);
                        let next_y =
                            (current.y + angle.sin() as f32).clamp(1.0, grid.height as f32 - 2.0);
                        let col = next_x as usize;
                        let row = next_y as usize;
                        let index = row * grid.width + col;
                        if grid.age[index] == 0 && grid.adjacent_stuck(col, row) {
                            let local = field[index].clamp(0.0, 1.0);
                            let mut probability =
                                (params.stick_base + params.stick_span * local).clamp(0.0, 1.0);
                            if params.bias_strength > 0.0 {
                                let approach = (next_y - current.y).atan2(next_x - current.x);
                                let delta = f64::from(approach - orientation[index]);
                                let comb = 0.5 * (1.0 - (6.0 * delta).cos());
                                probability *= 1.0 - params.bias_strength * comb as f32;
                            }
                            let roll = agent_unit(current.id, salt, 22);
                            if roll < f64::from(probability) {
                                return (current, Some((index as u32, current.id)), false);
                            }
                        }
                        current.x = next_x;
                        current.y = next_y;
                        current.steps_walked += 1;
                        let dist = (current.x - center.0).hypot(current.y - center.1);
                        if dist > kill_margin || current.steps_walked >= max_walk {
                            return (current, None, true);
                        }
                    }
                    (current, None, false)
                })
                .collect();

            // Apply stick proposals in walker-id order (deterministic).
            let mut proposals: Vec<(u32, u64)> =
                results.iter().filter_map(|&(_, stick, _)| stick).collect();
            proposals.sort_by_key(|&(_, id)| id);
            for (index, _) in proposals {
                if self.age[index as usize] == 0 {
                    self.age[index as usize] = self.next_age;
                    self.next_age += 1;
                    let col = (index as usize % self.width) as f32;
                    let row = (index as usize / self.width) as f32;
                    self.bbox.0 = self.bbox.0.min(col);
                    self.bbox.1 = self.bbox.1.min(row);
                    self.bbox.2 = self.bbox.2.max(col);
                    self.bbox.3 = self.bbox.3.max(row);
                }
            }

            // Survivors continue; stuck and killed walkers leave the pool.
            pool = results
                .into_iter()
                .filter_map(|(walker, stick, killed)| {
                    (stick.is_none() && !killed).then_some(walker)
                })
                .collect();
        }
    }
}

/// A grown dielectric-breakdown streamer.
pub struct Streamer {
    /// Grown cells in growth order (col, row).
    pub cells: Vec<(u16, u16)>,
    /// Parent index into `cells` for each cell (self for the root).
    pub parents: Vec<u32>,
    /// Whether the streamer reached the target.
    pub reached: bool,
    /// Indices (into `cells`) of the main channel from root to arrival.
    pub main_channel: Vec<u32>,
}

/// Grow a dielectric-breakdown streamer from `start` toward `target` on a
/// potential grid: frontier cells are sampled with probability proportional
/// to their normalized potential drop raised to `eta`, with a fraction of
/// picks biased straight at the target.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn grow_streamer(
    potential: &[f32],
    width: usize,
    height: usize,
    start: (usize, usize),
    target: (usize, usize),
    eta: f64,
    target_bias: f64,
    max_cells: usize,
    rng: &mut Sha3RandomByteStream,
) -> Streamer {
    let index_of = |col: usize, row: usize| row * width + col;
    let mut in_tree = vec![false; width * height];
    let mut cells: Vec<(u16, u16)> = Vec::new();
    let mut parents: Vec<u32> = Vec::new();
    let mut cell_of_index: Vec<u32> = vec![u32::MAX; width * height];

    let mut frontier: Vec<(u16, u16, u32)> = Vec::new(); // (col, row, parent cell)
    let push_neighbors = |col: usize,
                          row: usize,
                          parent: u32,
                          in_tree: &[bool],
                          frontier: &mut Vec<(u16, u16, u32)>| {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let ncol = col as i64 + dx;
                let nrow = row as i64 + dy;
                if ncol < 0 || nrow < 0 || ncol >= width as i64 || nrow >= height as i64 {
                    continue;
                }
                if !in_tree[index_of(ncol as usize, nrow as usize)] {
                    frontier.push((ncol as u16, nrow as u16, parent));
                }
            }
        }
    };

    in_tree[index_of(start.0, start.1)] = true;
    cells.push((start.0 as u16, start.1 as u16));
    parents.push(0);
    cell_of_index[index_of(start.0, start.1)] = 0;
    push_neighbors(start.0, start.1, 0, &in_tree, &mut frontier);

    let mut arrival: Option<u32> = None;
    while cells.len() < max_cells && arrival.is_none() {
        // Drop frontier entries swallowed by the tree.
        frontier.retain(|&(col, row, _)| !in_tree[index_of(col as usize, row as usize)]);
        if frontier.is_empty() {
            break;
        }

        let chosen = if rng.next_f64() < target_bias {
            // Target bias: sample the frontier by inverse squared distance
            // to the target (dithered, so channels never run laser-straight).
            let weights: Vec<f64> = frontier
                .iter()
                .map(|&(col, row, _)| {
                    let dx = f64::from(col) - target.0 as f64;
                    let dy = f64::from(row) - target.1 as f64;
                    1.0 / (dx * dx + dy * dy + 4.0).powi(2)
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let mut pick = rng.next_f64() * total;
            let mut chosen = frontier.len() - 1;
            for (slot, weight) in weights.iter().enumerate() {
                pick -= weight;
                if pick <= 0.0 {
                    chosen = slot;
                    break;
                }
            }
            chosen
        } else {
            // Potential-drop weighted sample.
            let (mut lowest, mut highest) = (f32::INFINITY, f32::NEG_INFINITY);
            for &(col, row, _) in &frontier {
                let value = potential[index_of(col as usize, row as usize)];
                lowest = lowest.min(value);
                highest = highest.max(value);
            }
            let range = f64::from(highest - lowest).max(1e-12);
            let weights: Vec<f64> = frontier
                .iter()
                .map(|&(col, row, _)| {
                    let value = potential[index_of(col as usize, row as usize)];
                    ((f64::from(highest - value) / range) + 1e-4).powf(eta)
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let mut pick = rng.next_f64() * total;
            let mut chosen = frontier.len() - 1;
            for (slot, weight) in weights.iter().enumerate() {
                pick -= weight;
                if pick <= 0.0 {
                    chosen = slot;
                    break;
                }
            }
            chosen
        };

        let (col, row, parent) = frontier.swap_remove(chosen);
        let index = index_of(col as usize, row as usize);
        in_tree[index] = true;
        let cell_id = cells.len() as u32;
        cells.push((col, row));
        parents.push(parent);
        cell_of_index[index] = cell_id;
        push_neighbors(col as usize, row as usize, cell_id, &in_tree, &mut frontier);

        let dx = (i64::from(col) - target.0 as i64).abs();
        let dy = (i64::from(row) - target.1 as i64).abs();
        if dx <= 1 && dy <= 1 {
            arrival = Some(cell_id);
        }
    }

    let mut main_channel = Vec::new();
    if let Some(mut cursor) = arrival {
        loop {
            main_channel.push(cursor);
            let parent = parents[cursor as usize];
            if parent == cursor {
                break;
            }
            if cursor == 0 {
                break;
            }
            cursor = parent;
        }
        main_channel.reverse();
    }
    Streamer { cells, parents, reached: arrival.is_some(), main_channel }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> Sha3RandomByteStream {
        Sha3RandomByteStream::new(&[42u8; 32], 100.0, 300.0, 300.0, 1.0)
    }

    #[test]
    fn physarum_ticks_are_deterministic() {
        let (width, height) = (64usize, 48usize);
        let food = vec![0.5f32; width * height];
        let run = || {
            let mut rng = test_rng();
            let mut swarm = PhysarumSwarm::seed_weighted(2_000, width, height, &food, &mut rng);
            let mut trail = vec![0.0f32; width * height];
            let mut scratch = Vec::new();
            let mut deposits = Vec::new();
            let params = PhysarumParams {
                sense_distance: 5.0,
                sense_angle: 0.4,
                turn_angle: 0.4,
                step_length: 1.0,
                deposit: 0.1,
                decay: 0.92,
                jitter: 0.15,
                food_weight: 0.35,
            };
            for step in 0..50 {
                swarm.tick(&trail, &food, &params, step, &mut deposits);
                PhysarumSwarm::apply_deposits(&mut trail, &deposits);
                diffuse_decay(&mut trail, &mut scratch, width, height, params.decay);
            }
            trail
        };
        let first = run();
        let second = run();
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "physarum must be bit-deterministic");
        }
    }

    #[test]
    fn physarum_gathers_on_a_food_stripe() {
        let (width, height) = (96usize, 64usize);
        let mut food = vec![0.02f32; width * height];
        for col in 0..width {
            for row in 28..36 {
                food[row * width + col] = 1.0;
            }
        }
        let mut rng = test_rng();
        let uniform = vec![1.0f32; width * height];
        let mut swarm = PhysarumSwarm::seed_weighted(4_000, width, height, &uniform, &mut rng);
        let mut trail = vec![0.0f32; width * height];
        let mut scratch = Vec::new();
        let mut deposits = Vec::new();
        let params = PhysarumParams {
            sense_distance: 6.0,
            sense_angle: 0.4,
            turn_angle: 0.4,
            step_length: 1.0,
            deposit: 0.08,
            decay: 0.92,
            jitter: 0.1,
            food_weight: 0.6,
        };
        for step in 0..300 {
            swarm.tick(&trail, &food, &params, step, &mut deposits);
            PhysarumSwarm::apply_deposits(&mut trail, &deposits);
            diffuse_decay(&mut trail, &mut scratch, width, height, params.decay);
        }
        let stripe: f32 = (28..36)
            .flat_map(|row| (0..width).map(move |col| (row, col)))
            .map(|(row, col)| trail[row * width + col])
            .sum();
        let total: f32 = trail.iter().sum();
        let stripe_share = stripe / total.max(1e-9);
        let stripe_area_share = 8.0 / height as f32;
        assert!(
            stripe_share > stripe_area_share * 2.0,
            "agents should crowd the food stripe: share {stripe_share:.3} vs area {stripe_area_share:.3}"
        );
    }

    #[test]
    fn dla_grows_a_connected_cluster_deterministically() {
        let (width, height) = (128usize, 128usize);
        let field = vec![0.5f32; width * height];
        let orientation = vec![0.0f32; width * height];
        let run = || {
            let mut rng = test_rng();
            let mut grid = DlaGrid::new(width, height, &[(64, 64)]);
            grid.grow(
                4_000,
                &field,
                &orientation,
                &DlaParams {
                    stick_base: 0.6,
                    stick_span: 0.4,
                    bias_strength: 0.0,
                    batch_steps: 64,
                    concurrent: 512,
                },
                &mut rng,
            );
            grid
        };
        let first = run();
        let second = run();
        assert_eq!(first.age, second.age, "DLA must be deterministic");
        assert!(
            first.stuck_count() > 500,
            "cluster should grow substantially, got {}",
            first.stuck_count()
        );

        // Connectivity: every stuck cell (except the nucleus) touches another.
        for row in 0..height {
            for col in 0..width {
                if first.age[row * width + col] > 1 {
                    assert!(
                        first.adjacent_stuck(col, row),
                        "stuck cell ({col},{row}) must touch the cluster"
                    );
                }
            }
        }
    }

    #[test]
    fn streamer_reaches_its_target() {
        let (width, height) = (96usize, 64usize);
        // Potential well at the target draws the discharge across.
        let target = (80usize, 32usize);
        let potential: Vec<f32> = (0..width * height)
            .map(|index| {
                let col = (index % width) as f32;
                let row = (index / width) as f32;
                let dx = col - target.0 as f32;
                let dy = row - target.1 as f32;
                -200.0 / (dx * dx + dy * dy + 4.0).sqrt()
            })
            .collect();
        let mut rng = test_rng();
        let bolt =
            grow_streamer(&potential, width, height, (12, 32), target, 2.0, 0.15, 20_000, &mut rng);
        assert!(bolt.reached, "streamer must arrive at the target");
        assert!(!bolt.main_channel.is_empty());
        let first = bolt.cells[bolt.main_channel[0] as usize];
        assert_eq!((usize::from(first.0), usize::from(first.1)), (12, 32));
        let last = bolt.cells[*bolt.main_channel.last().expect("non-empty") as usize];
        let dx = (i64::from(last.0) - target.0 as i64).abs();
        let dy = (i64::from(last.1) - target.1 as i64).abs();
        assert!(dx <= 1 && dy <= 1, "channel must terminate at the target");
    }
}
