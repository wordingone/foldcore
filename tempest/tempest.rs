// 3D History-Dependent Cellular Automaton
// f: Each cell maintains 3-bit history, transition depends on history pattern

const DIM_X: usize = 32;
const DIM_Y: usize = 32;
const DIM_Z: usize = 32;
const TOTAL_CELLS: usize = DIM_X * DIM_Y * DIM_Z;
const CYCLES: usize = 10000;

// Each cell stores current value (bit 0) + 2 previous values (bits 1,2)
type Grid = Vec<u8>;

fn idx(x: usize, y: usize, z: usize) -> usize {
    ((z % DIM_Z) * DIM_Y + (y % DIM_Y)) * DIM_X + (x % DIM_X)
}

fn coords(i: usize) -> (usize, usize, usize) {
    let x = i % DIM_X;
    let y = (i / DIM_X) % DIM_Y;
    let z = i / (DIM_X * DIM_Y);
    (x, y, z)
}

// Count face neighbors that are currently 1
fn face_neighbors(grid: &Grid, x: usize, y: usize, z: usize) -> u8 {
    let offsets: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ];
    let mut count = 0u8;
    for &(dx, dy, dz) in &offsets {
        let nx = ((x as i32 + dx + DIM_X as i32) as usize) % DIM_X;
        let ny = ((y as i32 + dy + DIM_Y as i32) as usize) % DIM_Y;
        let nz = ((z as i32 + dz + DIM_Z as i32) as usize) % DIM_Z;
        if grid[idx(nx, ny, nz)] & 1 == 1 {
            count += 1;
        }
    }
    count
}

// Count neighbors with specific history pattern
fn neighbors_with_history(grid: &Grid, x: usize, y: usize, z: usize, pattern: u8) -> u8 {
    let offsets: [(i32, i32, i32); 6] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ];
    let mut count = 0u8;
    for &(dx, dy, dz) in &offsets {
        let nx = ((x as i32 + dx + DIM_X as i32) as usize) % DIM_X;
        let ny = ((y as i32 + dy + DIM_Y as i32) as usize) % DIM_Y;
        let nz = ((z as i32 + dz + DIM_Z as i32) as usize) % DIM_Z;
        if grid[idx(nx, ny, nz)] & 0b111 == pattern {
            count += 1;
        }
    }
    count
}

// History patterns:
// 000 = stable dead (dead for 3 cycles)
// 001 = just born (was dead, now alive)
// 010 = died once, was alive before
// 011 = alive, was alive, was dead before
// 100 = died, was dead, was alive before
// 101 = oscillating (alive-dead-alive)
// 110 = dying (alive-alive-dead)
// 111 = stable alive (alive for 3 cycles)

fn transition(grid: &Grid) -> Grid {
    let mut next = vec![0u8; TOTAL_CELLS];

    for i in 0..TOTAL_CELLS {
        let (x, y, z) = coords(i);
        let cell = grid[i];
        let current = cell & 1;
        let history = cell & 0b111;

        let face_count = face_neighbors(grid, x, y, z);

        // Count neighbors by history type
        let stable_alive = neighbors_with_history(grid, x, y, z, 0b111);
        let stable_dead = neighbors_with_history(grid, x, y, z, 0b000);
        let oscillating = neighbors_with_history(grid, x, y, z, 0b101);

        let new_value: u8;

        match history {
            0b000 => {
                // Stable dead: need strong stimulus to activate
                // Birth if 2-3 face neighbors AND at least 1 stable alive neighbor
                if face_count >= 2 && face_count <= 3 && stable_alive >= 1 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b001 => {
                // Just born: fragile, needs support
                // Survive if 2-4 face neighbors
                if face_count >= 2 && face_count <= 4 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b010 => {
                // Died once: can be reborn easily
                // Birth if 1-3 face neighbors
                if face_count >= 1 && face_count <= 3 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b011 => {
                // Recently alive: moderate stability
                // Survive if 1-4 face neighbors
                if face_count >= 1 && face_count <= 4 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b100 => {
                // Was alive before: can return
                // Birth if 2-4 face neighbors
                if face_count >= 2 && face_count <= 4 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b101 => {
                // Oscillating: tends to continue
                // Alive if odd number of oscillating neighbors
                if oscillating % 2 == 1 || face_count == 3 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b110 => {
                // Dying: hard to save
                // Survive only if 3 face neighbors AND no stable dead neighbors
                if face_count == 3 && stable_dead == 0 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            0b111 => {
                // Stable alive: resilient
                // Survive if 1-5 face neighbors
                if face_count >= 1 && face_count <= 5 {
                    new_value = 1;
                } else {
                    new_value = 0;
                }
            }
            _ => {
                new_value = 0;
            }
        }

        // Shift history left, insert new value
        // new_history = ((old_history << 1) | new_value) & 0b111
        let new_history = ((history << 1) | new_value) & 0b111;
        next[i] = new_history;
    }

    next
}

fn count_ones(grid: &Grid) -> usize {
    grid.iter().map(|&b| (b & 1) as usize).sum()
}

fn count_history_pattern(grid: &Grid, pattern: u8) -> usize {
    grid.iter().filter(|&&b| (b & 0b111) == pattern).count()
}

fn entropy(grid: &Grid) -> f64 {
    let ones = count_ones(grid);
    let zeros = TOTAL_CELLS - ones;
    if ones == 0 || zeros == 0 {
        return 0.0;
    }
    let p1 = ones as f64 / TOTAL_CELLS as f64;
    let p0 = zeros as f64 / TOTAL_CELLS as f64;
    -(p1 * p1.log2() + p0 * p0.log2())
}

fn history_entropy(grid: &Grid) -> f64 {
    let mut counts = [0usize; 8];
    for &b in grid {
        counts[(b & 0b111) as usize] += 1;
    }
    let mut h = 0.0;
    for count in &counts {
        if *count > 0 {
            let p = *count as f64 / TOTAL_CELLS as f64;
            h -= p * p.log2();
        }
    }
    h
}

fn layer_distribution(grid: &Grid) -> Vec<usize> {
    let mut layers = vec![0usize; DIM_Z];
    for z in 0..DIM_Z {
        for y in 0..DIM_Y {
            for x in 0..DIM_X {
                layers[z] += (grid[idx(x, y, z)] & 1) as usize;
            }
        }
    }
    layers
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn first_64_cells(grid: &Grid) -> String {
    grid.iter()
        .take(64)
        .map(|&b| format!("{}", b & 1))
        .collect()
}

fn count_changes(old: &Grid, new: &Grid) -> usize {
    old.iter().zip(new.iter()).filter(|(&a, &b)| (a & 1) != (b & 1)).count()
}

fn main() {
    let seed: u64 = 0xA1570203DEFED3D;
    let mut rng = seed;

    // Initialize with ~40% density, random history (all start as just-current state)
    let mut grid: Grid = (0..TOTAL_CELLS)
        .map(|_| if xorshift(&mut rng) % 100 < 40 { 0b001 } else { 0b000 })
        .collect();

    // Run a few setup cycles to populate history
    for _ in 0..3 {
        grid = transition(&grid);
    }

    let checkpoints = [0, 10, 50, 100, 500, 1000, 2000, 5000, 10000];

    println!("# 3D History-Dependent Cellular Automaton");
    println!();
    println!("## Parameters");
    println!();
    println!("| Parameter | Value |");
    println!("|-----------|-------|");
    println!("| M | {} cells ({} bits state + 2 bits history each) |", TOTAL_CELLS, TOTAL_CELLS);
    println!("| C | {} cycles |", CYCLES);
    println!("| Seed | 0xA1570203DEFED3D |");
    println!("| Dimensions | {}x{}x{} |", DIM_X, DIM_Y, DIM_Z);
    println!("| f | History-dependent with 8 behavioral modes |");
    println!("| Topology | 3D torus (wraparound on all axes) |");
    println!("| Initial density | ~40% |");
    println!("| D | 0 (closed system) |");
    println!();
    println!("## The Physics (f)");
    println!();
    println!("3D history-dependent cellular automaton on 32x32x32 torus:");
    println!();
    println!("```");
    println!("Each cell stores 3-bit history: [t-2, t-1, t]");
    println!("8 history patterns -> 8 behavioral modes:");
    println!();
    println!("000 (stable dead): Birth if 2-3 face neighbors AND 1+ stable alive");
    println!("001 (just born): Survive if 2-4 face neighbors");
    println!("010 (died once): Birth if 1-3 face neighbors");
    println!("011 (recently alive): Survive if 1-4 face neighbors");
    println!("100 (was alive before): Birth if 2-4 face neighbors");
    println!("101 (oscillating): Alive if odd oscillating neighbors OR 3 face");
    println!("110 (dying): Survive if 3 face AND no stable dead");
    println!("111 (stable alive): Survive if 1-5 face neighbors");
    println!("```");
    println!();
    println!("## Observations");

    let mut cycle = 0;
    let mut prev_ones = count_ones(&grid);
    let mut prev_entropy = entropy(&grid);
    let mut prev_grid = grid.clone();

    for &checkpoint in &checkpoints {
        while cycle < checkpoint {
            let new_grid = transition(&grid);
            prev_grid = grid;
            grid = new_grid;
            cycle += 1;
        }

        let ones = count_ones(&grid);
        let ent = entropy(&grid);
        let hist_ent = history_entropy(&grid);
        let layers = layer_distribution(&grid);
        let changes = if checkpoint > 0 { count_changes(&prev_grid, &grid) } else { 0 };

        let suffix = if checkpoint == 0 { " (Initial)" }
                     else if checkpoint == CYCLES { " (Final)" }
                     else { "" };

        println!();
        println!("### Cycle {}{}", checkpoint, suffix);
        println!();
        println!("| Metric | Value |");
        println!("|--------|-------|");
        println!("| 1s | {} |", ones);
        println!("| 0s | {} |", TOTAL_CELLS - ones);
        println!("| Density | {:.4} |", ones as f64 / TOTAL_CELLS as f64);
        println!("| Binary entropy | {:.4} |", ent);
        println!("| History entropy (8 patterns) | {:.4} / 3.0 |", hist_ent);
        println!("| First 64 cells | {} |", first_64_cells(&grid));

        // History pattern distribution
        println!("| Pattern 000 (stable dead) | {} ({:.2}%) |",
            count_history_pattern(&grid, 0b000),
            count_history_pattern(&grid, 0b000) as f64 / TOTAL_CELLS as f64 * 100.0);
        println!("| Pattern 111 (stable alive) | {} ({:.2}%) |",
            count_history_pattern(&grid, 0b111),
            count_history_pattern(&grid, 0b111) as f64 / TOTAL_CELLS as f64 * 100.0);
        println!("| Pattern 101 (oscillating) | {} ({:.2}%) |",
            count_history_pattern(&grid, 0b101),
            count_history_pattern(&grid, 0b101) as f64 / TOTAL_CELLS as f64 * 100.0);

        let min_layer = layers.iter().min().unwrap_or(&0);
        let max_layer = layers.iter().max().unwrap_or(&0);
        let mean_layer: f64 = layers.iter().sum::<usize>() as f64 / DIM_Z as f64;
        let layer_size = DIM_X * DIM_Y;
        println!("| Layer min/mean/max 1s | {}/{:.1}/{} |", min_layer, mean_layer, max_layer);
        println!("| Layer density range | {:.2}% - {:.2}% |",
            *min_layer as f64 / layer_size as f64 * 100.0,
            *max_layer as f64 / layer_size as f64 * 100.0);

        if checkpoint > 0 {
            println!("| Cells changed last cycle | {} ({:.2}%) |", changes, changes as f64 / TOTAL_CELLS as f64 * 100.0);
            println!();
            println!("Changes from previous checkpoint:");
            println!("- 1s: {} -> {} (delta: {})", prev_ones, ones, ones as i64 - prev_ones as i64);
            println!("- Binary entropy: {:.4} -> {:.4}", prev_entropy, ent);
        }

        prev_ones = ones;
        prev_entropy = ent;
        prev_grid = grid.clone();
    }

    println!();
    println!("## Final State Analysis");
    println!();

    let final_layers = layer_distribution(&grid);
    let layer_size = DIM_X * DIM_Y;
    println!("Layer-by-layer 1s count (z=0 to z=31):");
    println!();
    for (z, count) in final_layers.iter().enumerate() {
        println!("z={:2}: {} 1s ({:.2}%)", z, count, *count as f64 / layer_size as f64 * 100.0);
    }

    println!();
    println!("History pattern distribution:");
    println!();
    let patterns = ["000 (stable dead)", "001 (just born)", "010 (died once)",
                   "011 (recently alive)", "100 (was alive before)", "101 (oscillating)",
                   "110 (dying)", "111 (stable alive)"];
    for (i, name) in patterns.iter().enumerate() {
        let count = count_history_pattern(&grid, i as u8);
        println!("{}: {} ({:.2}%)", name, count, count as f64 / TOTAL_CELLS as f64 * 100.0);
    }
}
