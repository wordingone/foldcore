/**
 * Gradient Ablation — Step 14
 *
 * Runs the full test suite with the gradient term zeroed out:
 *   fold: u = si + alr * ri + 0 + memW * mi
 *
 * Tests:
 *   Part A — T1-T4 at DIM=64, 512, 8192  (FluxCoreTrue-style dual-memory, nograd)
 *   Part B — TEST 1 attractor genesis (4-dist A→B→C→D→A, DIM=64, DynamicAttractorField nograd)
 *   Part C — CSI real data (384-dim, nograd)
 *
 * Outputs everything to results/ablation_gradient.txt
 */

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = path.join(__dirname, '..');

// ═══════════════════════════════════════════════════════════════════
// UTILITIES (shared)
// ═══════════════════════════════════════════════════════════════════
let _rng = 42;
const rand = () => { _rng = (Math.imul(1664525, _rng) + 1013904223) >>> 0; return (_rng >>> 0) / 4294967296; };
const randN = () => { const u = rand() + 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const normalize = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; const n = Math.sqrt(s) + 1e-12; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
const randUnit = d => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = randN(); return normalize(v); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const l1 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; };
const clone = v => new Float64Array(v);
const avg = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
const std = a => { const m = avg(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length || 1)); };
const tail = (a, f) => a.slice(Math.max(0, Math.floor(a.length * (1 - f))));
const head = (a, f) => a.slice(0, Math.max(1, Math.floor(a.length * f)));

function makeOrtho(base, dim) {
  const r = randUnit(dim), p = dot(r, base), b = new Float64Array(dim);
  for (let i = 0; i < dim; i++) b[i] = r[i] - p * base[i];
  return normalize(b);
}
const makeStatic = (base, n) => () => { const o = new Float64Array(base.length); for (let i = 0; i < o.length; i++) o[i] = base[i] + n * randN(); return normalize(o); };
const makeRotating = (A, P, d, n) => t => { const c = Math.cos(t * d), s = Math.sin(t * d), o = new Float64Array(A.length); for (let i = 0; i < o.length; i++) o[i] = c * A[i] + s * P[i] + n * randN(); return normalize(o); };
const makeNoise = d => () => randUnit(d);

// ═══════════════════════════════════════════════════════════════════
// Logging
// ═══════════════════════════════════════════════════════════════════
const lines = [];
const log = s => { lines.push(s); process.stdout.write(s + '\n'); };

// ═══════════════════════════════════════════════════════════════════
// PART A — FluxCoreTrue (Dual-Memory) with gradient ZEROED
// ═══════════════════════════════════════════════════════════════════
class FluxCoreTrueNoGrad {
  constructor(dim) {
    this.dim = dim;
    this.self = randUnit(dim);
    this.mem1 = randUnit(dim);
    this.mem2 = randUnit(dim);
    this.velocity = new Float64Array(dim);
    this.prevSelf = clone(this.self);
  }

  fold(reality, baseLr, memLr, k, memW, velDecay, velGain) {
    const dim = this.dim, s = this.self, r = reality;
    let gd = 0;
    for (let i = 0; i < dim; i++) gd += Math.abs(s[i] - r[i]);
    gd /= dim;

    const alr = baseLr * (1 + k * gd);
    const sim1 = Math.abs(dot(this.mem1, r));
    const sim2 = Math.abs(dot(this.mem2, r));
    const activeMem = sim1 > sim2 ? this.mem1 : this.mem2;
    const activeIdx = sim1 > sim2 ? 1 : 2;

    for (let idx = 0; idx < dim; idx++) {
      const si = s[idx], ri = r[idx], mi = activeMem[idx];
      const d = Math.abs(si - ri);
      const left = s[(idx + dim - 1) % dim];
      const right = s[(idx + 1) % dim];
      const grad = (si - left) - (si - right);
      // ABLATED: gradient term zeroed
      s[idx] = si + alr * ri + 0 /* (alr * 0.5) * d * grad */ + memW * mi;
    }
    normalize(s);

    if (activeIdx === 1) {
      for (let i = 0; i < dim; i++) this.mem1[i] += memLr * r[i];
      normalize(this.mem1);
    } else {
      for (let i = 0; i < dim; i++) this.mem2[i] += memLr * r[i];
      normalize(this.mem2);
    }

    for (let i = 0; i < dim; i++) {
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    }
    this.prevSelf = clone(s);
    return gd;
  }

  output(velScale) {
    const dim = this.dim;
    const o = new Float64Array(dim);
    for (let i = 0; i < dim; i++) o[i] = this.self[i] + velScale * this.velocity[i];
    return normalize(o);
  }
}

// ── T1 — ANTICIPATION ──
function test1(dim, noise, delta, lr, warmup, measure) {
  log(`\n--- T1: Anticipation  DIM=${dim} delta=${delta}rad/tick lr=${lr} ---`);
  const lag = (1 - lr) / lr;
  const velScale = lag + 1;
  const A = randUnit(dim), P = makeOrtho(A, dim), gen = makeRotating(A, P, delta, noise);
  const fc = new FluxCoreTrueNoGrad(dim);

  for (let t = 0; t < warmup; t++) fc.fold(gen(t), lr, 0.005, 0, 0.05, 0.95, 0.05);

  const gains = [];
  for (let t = warmup; t < warmup + measure; t++) {
    const rNow = gen(t), rNext = gen(t + 1);
    fc.fold(rNow, lr, 0.005, 0, 0.05, 0.95, 0.05);
    const output = fc.output(velScale);
    gains.push(l1(fc.self, rNext) - l1(output, rNext));
  }

  const g = avg(tail(gains, 0.3));
  const pass = g > 0;
  log(`  lag=${lag.toFixed(1)} velScale=${velScale.toFixed(1)}`);
  log(`  avg gain (last 30%) = ${g.toFixed(8)}  -> ${pass ? 'PASS' : 'FAIL'}`);
  return { pass, gain: g };
}

// ── T2 — ACCELERATED REACQUISITION ──
function test2(dim, noise, lr, warmup, A1len, Blen, A2len, earlyW) {
  log(`\n--- T2: Accelerated Reacquisition  DIM=${dim} noise=${noise} ---`);
  const bA = randUnit(dim), bB = makeOrtho(bA, dim);
  const gA = makeStatic(bA, noise), gB = makeStatic(bB, noise), gN = makeNoise(dim);
  const fc = new FluxCoreTrueNoGrad(dim);

  for (let t = 0; t < warmup; t++) fc.fold(gN(), lr, 0.015, 20, 0.15, 0.95, 0.05);

  const ea1 = [];
  for (let t = 0; t < A1len; t++) {
    const r = gA();
    fc.fold(r, lr, 0.015, 20, 0.15, 0.95, 0.05);
    if (t < earlyW) ea1.push(l1(fc.self, r));
  }

  for (let t = 0; t < Blen; t++) fc.fold(gB(), lr, 0.015, 20, 0.15, 0.95, 0.05);

  const mem1_A = Math.abs(dot(fc.mem1, bA));
  const mem2_B = Math.abs(dot(fc.mem2, bB));
  log(`  mem1.A = ${mem1_A.toFixed(4)}, mem2.B = ${mem2_B.toFixed(4)}`);

  const ea2 = [];
  for (let t = 0; t < A2len; t++) {
    const r = gA();
    fc.fold(r, lr, 0.015, 20, 0.15, 0.95, 0.05);
    if (t < earlyW) ea2.push(l1(fc.self, r));
  }

  const m1 = avg(ea1), m2 = avg(ea2);
  const pass = m2 < m1;
  log(`  early A1 (${earlyW} ticks) = ${m1.toFixed(7)}`);
  log(`  early A2 (${earlyW} ticks) = ${m2.toFixed(7)}`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'}`);
  return { pass, earlyA1: m1, earlyA2: m2 };
}

// ── T3 — DISTRIBUTIONAL SHIFT ──
function test3(dim, noise, lr, warmup, stLen, abLen) {
  log(`\n--- T3: Distributional Shift  DIM=${dim} noise=${noise} ---`);
  const A = randUnit(dim), B = makeOrtho(A, dim);
  const gA = makeStatic(A, noise), gB = makeStatic(B, noise);
  const fc = new FluxCoreTrueNoGrad(dim);

  for (let t = 0; t < warmup; t++) fc.fold(gA(), lr, 0.01, 20, 0.1, 0.95, 0.05);

  const sL = [], spL = [];
  for (let t = 0; t < stLen; t++) {
    const r = gA();
    fc.fold(r, lr, 0.01, 20, 0.1, 0.95, 0.05);
    sL.push(l1(fc.self, r));
  }
  for (let t = 0; t < abLen; t++) {
    const r = gB();
    fc.fold(r, lr, 0.01, 20, 0.1, 0.95, 0.05);
    spL.push(l1(fc.self, r));
  }

  const sm = avg(sL), ss = std(sL), spk = Math.max(...spL.slice(0, 10));
  const sigma = (spk - sm) / (ss + 1e-12), late = avg(tail(spL, 0.4));
  const pass = sigma >= 2 && late < spk;
  log(`  stable mean=${sm.toFixed(7)} std=${ss.toFixed(7)}`);
  log(`  spike=${spk.toFixed(7)} = ${sigma.toFixed(1)}s  late=${late.toFixed(7)}`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'}`);
  return { pass, stableMean: sm, spike: spk, sigma, late };
}

// ── T4 — ADAPTIVE LR ──
function test4(dim, noise, lr, warmup, stLen, abLen) {
  log(`\n--- T4: Adaptive lr vs Fixed lr  DIM=${dim} ---`);
  const A = randUnit(dim), B = makeOrtho(A, dim);
  const gA = makeStatic(A, noise), gB = makeStatic(B, noise);

  const fcA = new FluxCoreTrueNoGrad(dim);
  const fcF = new FluxCoreTrueNoGrad(dim);
  fcF.mem1 = clone(fcA.mem1);
  fcF.mem2 = clone(fcA.mem2);
  fcF.self = clone(fcA.self);

  for (let t = 0; t < warmup + stLen; t++) {
    const r = gA();
    fcA.fold(r, lr, 0.01, 20, 0.1, 0.95, 0.05);
    fcF.fold(r, lr, 0.01, 0, 0.1, 0.95, 0.05);
  }

  const aL = [], fL = [];
  for (let t = 0; t < abLen; t++) {
    const r = gB();
    fcA.fold(r, lr, 0.01, 20, 0.1, 0.95, 0.05);
    aL.push(l1(fcA.self, r));
    fcF.fold(r, lr, 0.01, 0, 0.1, 0.95, 0.05);
    fL.push(l1(fcF.self, r));
  }

  const q1a = avg(head(aL, 0.25)), q1f = avg(head(fL, 0.25));
  const cA = avg(aL), cF = avg(fL);
  const pass = q1a < q1f || cA < cF;
  log(`  Q1 cumL1: adaptive=${q1a.toFixed(7)}  fixed=${q1f.toFixed(7)}  Q1-wins=${q1a < q1f}`);
  log(`  Full cumL1: adaptive=${cA.toFixed(7)}  fixed=${cF.toFixed(7)}  full-wins=${cA < cF}`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'}`);
  return { pass, q1a, q1f, cA, cF };
}

// ═══════════════════════════════════════════════════════════════════
// PART B — DynamicAttractorField + ActiveInferenceEngine (nograd)
//          TEST 1: Attractor Genesis (A→B→C→D→A, DIM=64)
// ═══════════════════════════════════════════════════════════════════
class DynamicAttractorField {
  constructor(dim, opts = {}) {
    this.dim = dim;
    this.memories = [];
    this.spawnThreshold = opts.spawnThreshold ?? 0.5;
    this.mergeThreshold = opts.mergeThreshold ?? 0.95;
    this.pruneThreshold = opts.pruneThreshold ?? 500;
    this.memoryMeta = [];
    this.tick = 0;
  }

  selectAndUpdate(reality, self, memLr) {
    this.tick++;
    let surprise = 0;
    for (let i = 0; i < this.dim; i++) surprise += Math.abs(self[i] - reality[i]);
    surprise /= this.dim;

    let bestIdx = -1, bestSim = -1;
    for (let i = 0; i < this.memories.length; i++) {
      const sim = Math.abs(dot(this.memories[i], reality));
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }

    if (bestSim < this.spawnThreshold) {
      const newMem = clone(reality);
      const newIdx = this.memories.length;
      this.memories.push(newMem);
      this.memoryMeta[newIdx] = { lastUsed: this.tick, useCount: 1, birthTick: this.tick };
      return { memory: newMem, idx: newIdx, surprise, action: 'spawn', totalMemories: this.memories.length };
    }

    const activeMem = this.memories[bestIdx];
    for (let i = 0; i < this.dim; i++) this.memories[bestIdx][i] += memLr * reality[i];
    normalize(this.memories[bestIdx]);
    this.memoryMeta[bestIdx] = {
      lastUsed: this.tick,
      useCount: (this.memoryMeta[bestIdx]?.useCount || 0) + 1,
      birthTick: this.memoryMeta[bestIdx]?.birthTick || this.tick
    };

    this._mergeConverged();
    this._pruneStale();
    return { memory: activeMem, idx: bestIdx, surprise, action: 'use', totalMemories: this.memories.length };
  }

  _mergeConverged() {
    for (let i = 0; i < this.memories.length; i++) {
      for (let j = i + 1; j < this.memories.length; j++) {
        if (Math.abs(dot(this.memories[i], this.memories[j])) > this.mergeThreshold) {
          const mi = this.memoryMeta[i], mj = this.memoryMeta[j];
          const total = mi.useCount + mj.useCount;
          for (let k = 0; k < this.dim; k++)
            this.memories[i][k] = (mi.useCount * this.memories[i][k] + mj.useCount * this.memories[j][k]) / total;
          normalize(this.memories[i]);
          this.memoryMeta[i] = { lastUsed: Math.max(mi.lastUsed, mj.lastUsed), useCount: total, birthTick: Math.min(mi.birthTick, mj.birthTick) };
          this.memories.splice(j, 1); this.memoryMeta.splice(j, 1); j--;
        }
      }
    }
  }

  _pruneStale() {
    for (let i = this.memories.length - 1; i >= 0; i--) {
      const meta = this.memoryMeta[i];
      if (meta && this.tick - meta.lastUsed > this.pruneThreshold) {
        this.memories.splice(i, 1); this.memoryMeta.splice(i, 1);
      }
    }
  }

  getStats() {
    return {
      count: this.memories.length,
      ages: this.memories.map((_, i) => this.memoryMeta[i] ? this.tick - this.memoryMeta[i].birthTick : 0),
      useCounts: this.memoryMeta.map(m => m?.useCount || 0)
    };
  }
}

class ActiveInferenceEngineNoGrad {
  constructor(dim, opts = {}) {
    this.dim = dim;
    this.self = randUnit(dim);
    this.velocity = new Float64Array(dim);
    this.prevSelf = clone(this.self);
    this.attractors = new DynamicAttractorField(dim, opts);
  }

  fold(reality, baseLr, memLr, k, memW, velDecay, velGain) {
    const dim = this.dim, s = this.self, r = reality;
    let surprise = 0;
    for (let i = 0; i < dim; i++) surprise += Math.abs(s[i] - r[i]);
    surprise /= dim;

    const result = this.attractors.selectAndUpdate(r, s, memLr);
    const activeMem = result.memory;
    const alr = baseLr * (1 + k * surprise);

    for (let idx = 0; idx < dim; idx++) {
      const si = s[idx], ri = r[idx], mi = activeMem[idx];
      const d = Math.abs(si - ri);
      const left = s[(idx + dim - 1) % dim], right = s[(idx + 1) % dim];
      const grad = (si - left) - (si - right);
      // ABLATED: gradient term zeroed
      s[idx] = si + alr * ri + 0 /* (alr * 0.5) * d * grad */ + memW * mi;
    }
    normalize(s);

    for (let i = 0; i < dim; i++) {
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    }
    this.prevSelf = clone(s);

    return { surprise, attractorAction: result.action, memoryCount: result.totalMemories };
  }
}

// ═══════════════════════════════════════════════════════════════════
// RUN EVERYTHING
// ═══════════════════════════════════════════════════════════════════

log('='.repeat(72));
log('GRADIENT ABLATION — Step 14');
log('Fold equation: u = si + alr*ri + 0 + memW*mi  (gradient term ZEROED)');
log('='.repeat(72));

// ── PART A: T1-T4 at three scales ──
log('\n' + '='.repeat(72));
log('PART A: T1-T4 (Dual-Memory, No Gradient)');
log('='.repeat(72));

const partAResults = [];
for (const [dim, noise, delta, lr, earlyW] of [
  [64, 0.0003, 0.012, 0.08, 50],
  [512, 0.0003, 0.005, 0.08, 50],
  [8192, 0.0002, 0.002, 0.08, 50],
]) {
  log(`\n${'~'.repeat(72)}\nSCALE: DIM=${dim}\n${'~'.repeat(72)}`);
  const r1 = test1(dim, noise, delta, lr, 1500, 3000);
  const r2 = test2(dim, noise, lr, 500, 6000, 2000, 6000, earlyW);
  const r3 = test3(dim, noise, lr, 500, 3000, 600);
  const r4 = test4(dim, noise, lr, 500, 3000, 400);
  partAResults.push({ dim, t1: r1, t2: r2, t3: r3, t4: r4 });
}

log(`\n${'='.repeat(72)}`);
log('PART A RESULTS (No Gradient)');
log('='.repeat(72));
for (const r of partAResults) {
  const f = [r.t1.pass, r.t2.pass, r.t3.pass, r.t4.pass].map((v, i) => v ? `T${i+1}:PASS` : `T${i+1}:FAIL`).join('  ');
  const passes = [r.t1.pass, r.t2.pass, r.t3.pass, r.t4.pass].filter(Boolean).length;
  log(`  DIM=${String(r.dim).padEnd(6)}  ${passes}/4  [${f}]`);
}

// ── PART B: Attractor genesis TEST 1 (A→B→C→D→A, DIM=64) ──
log('\n' + '='.repeat(72));
log('PART B: Attractor Genesis (A->B->C->D->A, DIM=64, No Gradient)');
log('='.repeat(72));

{
  const DIM = 64;
  const NOISE = 0.001;

  // Reset RNG
  _rng = 42;

  let base = randUnit(DIM);
  const distributions = [clone(base)];
  for (let i = 1; i < 4; i++) {
    base = makeOrtho(base, DIM);
    distributions.push(clone(base));
  }

  const engine = new ActiveInferenceEngineNoGrad(DIM);

  let currentDist = 0;
  log('\nRunning 2000 ticks with 4 distribution switches...');

  for (let tick = 0; tick < 2000; tick++) {
    if (tick % 500 === 0 && tick > 0) {
      currentDist = (currentDist + 1) % distributions.length;
      log(`  [Environment] Switched to distribution ${currentDist}`);
    }

    const dist = distributions[currentDist];
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = dist[i] + NOISE * randN();
    const reality = normalize(v);

    const result = engine.fold(reality, 0.08, 0.015, 20, 0.15, 0.95, 0.05);

    if (tick % 250 === 0) {
      const stats = engine.attractors.getStats();
      log(`  Tick ${tick.toString().padStart(4)}: ${stats.count} memories (ages: ${stats.ages.join(',')}), surprise=${result.surprise.toFixed(4)}, action=${result.attractorAction}`);
    }
  }

  const finalStats = engine.attractors.getStats();
  log(`\n  Final: ${finalStats.count} memories`);
}

// ── PART C: CSI Real Data ──
log('\n' + '='.repeat(72));
log('PART C: CSI Real Data (384-dim, No Gradient)');
log('='.repeat(72));

{
  // Reset RNG
  _rng = 42;

  const embedded = JSON.parse(readFileSync(path.join(BASE, 'data', 'csi_embedded.json'), 'utf8'));
  const centers = JSON.parse(readFileSync(path.join(BASE, 'data', 'csi_division_centers.json'), 'utf8'));
  const divNames = Object.keys(centers).sort();

  log(`\nLoaded ${embedded.length} records, ${divNames.length} division centers`);
  log(`Vector dimension: ${embedded[0].vec.length}`);

  const DIM = 384;
  const engine = new ActiveInferenceEngineNoGrad(DIM, { spawnThreshold: 0.5, pruneThreshold: 2000 });

  log('\nRunning FluxCore (no gradient) on CSI embeddings...\n');

  const surpriseWindow = [];
  let spawnCount = 0;

  for (let tick = 0; tick < embedded.length; tick++) {
    const rec = embedded[tick];
    const vec = new Float64Array(rec.vec);
    const result = engine.fold(vec, 0.08, 0.015, 20, 0.15, 0.95, 0.05);
    surpriseWindow.push(result.surprise);
    if (result.attractorAction === 'spawn') spawnCount++;

    if ((tick + 1) % 100 === 0 || tick === embedded.length - 1) {
      const avgSurp = surpriseWindow.slice(-100).reduce((s, x) => s + x, 0) / Math.min(100, surpriseWindow.length);
      log(`  Tick ${String(tick + 1).padStart(4)}: memories=${result.memoryCount}, surprise=${result.surprise.toFixed(4)}, avg(last100)=${avgSurp.toFixed(4)}, div=${embedded[tick].division}`);
    }
  }

  log(`\nTotal spawns: ${spawnCount}`);

  // Memory-division alignment
  log('\n' + '-'.repeat(72));
  log('MEMORY-DIVISION ALIGNMENT (No Gradient)');
  log('-'.repeat(72));

  const mems = engine.attractors.memories;
  const meta = engine.attractors.memoryMeta;

  log(`\nFinal memory count: ${mems.length}`);

  let passCount = 0;
  const alignments = [];

  for (let m = 0; m < mems.length; m++) {
    const mem = mems[m];
    const sims = divNames.map(div => ({
      div,
      sim: dot(mem, new Float64Array(centers[div]))
    })).sort((a, b) => b.sim - a.sim);

    const best = sims[0], second = sims[1];
    const specificity = best.sim - second.sim;
    if (best.sim > 0.3) passCount++;
    alignments.push({ best, second, specificity });
  }

  const meanSim = alignments.reduce((s, a) => s + a.best.sim, 0) / alignments.length;
  const meanSpec = alignments.reduce((s, a) => s + a.specificity, 0) / alignments.length;

  log(`  Pass (dot > 0.3)   : ${passCount}/${mems.length}`);
  log(`  Mean best sim      : ${meanSim.toFixed(4)}`);
  log(`  Mean specificity   : ${meanSpec.toFixed(4)}`);
  log(`  Total spawns       : ${spawnCount}`);

  // ═══════════════════════════════════════════════════════════════════
  // COMPARISON TABLE
  // ═══════════════════════════════════════════════════════════════════
  log('\n' + '='.repeat(72));
  log('COMPARISON TABLE: GRADIENT vs NO GRADIENT');
  log('='.repeat(72));

  // Baseline values from results files
  log('\n--- T1-T4 (Dual-Memory FluxCoreTrue) ---');
  log('');
  log('  | Test | DIM  | Baseline (grad) | Ablation (nograd) | Verdict  |');
  log('  |------|------|-----------------|-------------------|----------|');

  // Baseline T1-T4 from baseline_true.txt
  const baselineT1 = [
    { dim: 64, t1: { gain: 0.00758019, pass: true }, t2: { earlyA1: 0.0571563, earlyA2: 0.0093612, pass: true }, t3: { spike: 0.1127372, sigma: 1881.9, pass: true }, t4: { q1a: 0.0469616, q1f: 0.0705916, pass: true } },
    { dim: 512, t1: { gain: 0.00208123, pass: true }, t2: { earlyA1: 0.0249031, earlyA2: 0.0045047, pass: true }, t3: { spike: 0.0435604, sigma: 3153.6, pass: true }, t4: { q1a: 0.0190567, q1f: 0.0229611, pass: true } },
    { dim: 8192, t1: { gain: 0.00021733, pass: true }, t2: { earlyA1: 0.0073691, earlyA2: 0.0013329, pass: true }, t3: { spike: 0.0118526, sigma: 6044.0, pass: true }, t4: { q1a: 0.0056287, q1f: 0.0059615, pass: true } },
  ];

  for (let i = 0; i < 3; i++) {
    const b = baselineT1[i];
    const a = partAResults[i];
    const dim = b.dim;

    // T1
    const bGain = b.t1.gain;
    const aGain = a.t1.gain;
    const t1Verdict = a.t1.pass ? (aGain >= bGain ? 'SAME/BETTER' : 'WORSE-VAL') : 'REGRESSED';
    log(`  | T1   | ${String(dim).padEnd(4)} | gain=${bGain.toFixed(8)} | gain=${aGain.toFixed(8)} | ${t1Verdict.padEnd(8)} |`);

    // T2
    const t2Verdict = a.t2.pass ? 'PASS' : 'REGRESSED';
    log(`  | T2   | ${String(dim).padEnd(4)} | A2=${b.t2.earlyA2.toFixed(7)}    | A2=${a.t2.earlyA2.toFixed(7)}    | ${t2Verdict.padEnd(8)} |`);

    // T3
    const t3Verdict = a.t3.pass ? 'PASS' : 'REGRESSED';
    log(`  | T3   | ${String(dim).padEnd(4)} | sigma=${b.t3.sigma.toFixed(1).padEnd(8)} | sigma=${a.t3.sigma.toFixed(1).padEnd(8)} | ${t3Verdict.padEnd(8)} |`);

    // T4
    const t4Verdict = a.t4.pass ? 'PASS' : 'REGRESSED';
    log(`  | T4   | ${String(dim).padEnd(4)} | Q1=${b.t4.q1a.toFixed(7)}    | Q1=${a.t4.q1a.toFixed(7)}    | ${t4Verdict.padEnd(8)} |`);
  }

  log('\n--- CSI Real Data ---');
  log('');
  log('  | Metric           | Baseline (grad) | Ablation (nograd) |');
  log('  |------------------|-----------------|-------------------|');
  log(`  | Total memories   | 359             | ${mems.length.toString().padEnd(17)} |`);
  log(`  | Pass (dot>0.3)   | 357/359         | ${passCount}/${mems.length}`.padEnd(42) + '|');
  log(`  | Mean best sim    | 0.5710          | ${meanSim.toFixed(4).padEnd(17)} |`);
  log(`  | Mean specificity | 0.0584          | ${meanSpec.toFixed(4).padEnd(17)} |`);
  log(`  | Total spawns     | 359             | ${spawnCount.toString().padEnd(17)} |`);

  // Overall verdict
  const allPartAPassed = partAResults.every(r => r.t1.pass && r.t2.pass && r.t3.pass && r.t4.pass);
  const csiClose = Math.abs(meanSim - 0.5710) < 0.05;

  log('\n' + '='.repeat(72));
  log('VERDICT');
  log('='.repeat(72));
  log(`  T1-T4 all pass (nograd): ${allPartAPassed ? 'YES' : 'NO'}`);
  log(`  CSI within 0.05 of baseline: ${csiClose ? 'YES' : 'NO'}`);

  if (allPartAPassed && csiClose) {
    log('  => Gradient term is UNNECESSARY. Removing it has no measurable negative impact.');
  } else if (allPartAPassed && !csiClose) {
    log('  => Gradient term has MARGINAL effect on real data. T1-T4 unaffected.');
  } else {
    log('  => Gradient term is NECESSARY. Removing it degrades performance.');
  }

  log('='.repeat(72));
}

// Write output
writeFileSync(path.join(BASE, 'results', 'ablation_gradient.txt'), lines.join('\n') + '\n');
log('\nCaptured to results/ablation_gradient.txt');
