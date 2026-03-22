/**
 * FluxCore Step 32 — tanh fold autoregression experiment
 *
 * Hypothesis: replacing normalize(u) with element-wise tanh(u) removes
 * the sphere projection, allowing the fold to escape the fixed point
 * under autoregression (self-feeding).
 *
 * Fold equation:
 *   u[i] = s[i] + alr * r[i] + memW * m[i]
 *   s[i] = tanh(u[i])   (was: s = normalize(u))
 *
 * State lives in [-1,1]^d instead of on unit sphere S^(d-1).
 */

import { writeFileSync } from 'fs';

// ════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════════════════════
let _rng = 42;
const rand = () => { _rng = (Math.imul(1664525, _rng) + 1013904223) >>> 0; return (_rng >>> 0) / 4294967296; };
const randN = () => { const u = rand() + 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

const normalize = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; const n = Math.sqrt(s) + 1e-12; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
const tanhVec = v => { for (let i = 0; i < v.length; i++) v[i] = Math.tanh(v[i]); return v; };
const randUnit = d => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = randN(); return normalize(v); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };
const cosineSim = (a, b) => { const na = norm(a), nb = norm(b); if (na < 1e-12 || nb < 1e-12) return 0; return dot(a, b) / (na * nb); };
const clone = v => new Float64Array(v);
const avg = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;

function makeOrtho(base, dim) {
  const r = randUnit(dim), p = dot(r, base), b = new Float64Array(dim);
  for (let i = 0; i < dim; i++) b[i] = r[i] - p * base[i];
  return normalize(b);
}

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD — adapted for tanh (non-unit-norm vectors)
// ════════════════════════════════════════════════════════════════════════════
class DynamicAttractorField {
  constructor(dim) {
    this.dim = dim;
    this.memories = [];
    this.memoryMeta = [];
    this.tick = 0;
    this.simMu = 0.5;
    this.simSigma = 0.1;
    this.simEmaAlpha = 0.01;
    this.simVarEma = 0.01;
    this.simCount = 0;
    this.surpriseEma = 0.05;
    this.memoryContrib = [];
    this.surpriseNoiseFloor = Infinity;
  }

  selectAndUpdate(reality, self, memLr) {
    this.tick++;

    // Surprise: use cosine distance since vectors are not unit-norm
    const cs = cosineSim(self, reality);
    let surprise = 1 - Math.abs(cs);

    this.surpriseEma = 0.99 * this.surpriseEma + 0.01 * surprise;
    if (surprise < this.surpriseNoiseFloor) this.surpriseNoiseFloor = surprise;

    // Find best-matching memory (cosine similarity)
    let bestIdx = -1, bestSim = -1;
    for (let i = 0; i < this.memories.length; i++) {
      const sim = Math.abs(cosineSim(this.memories[i], reality));
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }

    // Update running similarity stats
    if (this.memories.length > 0) {
      this.simCount++;
      if (this.simCount <= 5) {
        this.simMu = ((this.simCount - 1) * this.simMu + bestSim) / this.simCount;
        const diff = bestSim - this.simMu;
        this.simVarEma = ((this.simCount - 1) * this.simVarEma + diff * diff) / this.simCount;
        this.simSigma = Math.sqrt(this.simVarEma);
      } else {
        this.simMu = (1 - this.simEmaAlpha) * this.simMu + this.simEmaAlpha * bestSim;
        const diff = bestSim - this.simMu;
        this.simVarEma = (1 - this.simEmaAlpha) * this.simVarEma + this.simEmaAlpha * diff * diff;
        this.simSigma = Math.sqrt(this.simVarEma);
      }
    }

    let effectiveSpawn = this.simMu - 2 * this.simSigma;
    effectiveSpawn = Math.max(0.05, Math.min(0.95, effectiveSpawn));
    let effectiveMerge = 1 - this.surpriseEma;
    effectiveMerge = Math.max(0.8, Math.min(0.995, effectiveMerge));

    // SPAWN
    if (bestSim < effectiveSpawn) {
      const newMem = clone(reality);
      const newIdx = this.memories.length;
      this.memories.push(newMem);
      this.memoryMeta[newIdx] = { lastUsed: this.tick, useCount: 1, birthTick: this.tick };
      this.memoryContrib[newIdx] = 0;
      return {
        memory: newMem, idx: newIdx, surprise, action: 'spawn',
        totalMemories: this.memories.length, effectiveSpawn, effectiveMerge
      };
    }

    // Use best memory
    const activeMem = this.memories[bestIdx];
    const contribution = bestSim * (1 - surprise);
    const contribAlpha = 0.05;
    this.memoryContrib[bestIdx] = (1 - contribAlpha) * (this.memoryContrib[bestIdx] || 0) + contribAlpha * contribution;

    // Update memory toward reality — use tanh instead of normalize
    for (let i = 0; i < this.dim; i++) this.memories[bestIdx][i] += memLr * reality[i];
    tanhVec(this.memories[bestIdx]);

    this.memoryMeta[bestIdx] = {
      lastUsed: this.tick,
      useCount: (this.memoryMeta[bestIdx]?.useCount || 0) + 1,
      birthTick: this.memoryMeta[bestIdx]?.birthTick || this.tick
    };

    this._mergeConverged(effectiveMerge);
    this._pruneMemories();

    return {
      memory: activeMem, idx: bestIdx, surprise, action: 'use',
      totalMemories: this.memories.length, effectiveSpawn, effectiveMerge
    };
  }

  _mergeConverged(threshold) {
    for (let i = 0; i < this.memories.length; i++) {
      for (let j = i + 1; j < this.memories.length; j++) {
        if (Math.abs(cosineSim(this.memories[i], this.memories[j])) > threshold) {
          const mi = this.memoryMeta[i] || { useCount: 1 };
          const mj = this.memoryMeta[j] || { useCount: 1 };
          const total = mi.useCount + mj.useCount;
          for (let k = 0; k < this.dim; k++)
            this.memories[i][k] = (mi.useCount * this.memories[i][k] + mj.useCount * this.memories[j][k]) / total;
          tanhVec(this.memories[i]);
          this.memoryMeta[i] = { lastUsed: Math.max(mi.lastUsed, mj.lastUsed), useCount: total, birthTick: Math.min(mi.birthTick, mj.birthTick) };
          this.memoryContrib[i] = (this.memoryContrib[i] || 0) + (this.memoryContrib[j] || 0);
          this.memories.splice(j, 1); this.memoryMeta.splice(j, 1); this.memoryContrib.splice(j, 1); j--;
        }
      }
    }
  }

  _pruneMemories() {
    const noiseFloor = this.surpriseNoiseFloor * 1.5;
    const staleLimit = 5000;
    for (let i = this.memories.length - 1; i >= 0; i--) {
      const meta = this.memoryMeta[i];
      if (!meta) continue;
      const age = this.tick - meta.birthTick;
      const idle = this.tick - meta.lastUsed;
      const contrib = this.memoryContrib[i] || 0;
      if (age < 50) continue;
      if ((contrib < noiseFloor && idle > 100) || idle > staleLimit) {
        this.memories.splice(i, 1); this.memoryMeta.splice(i, 1); this.memoryContrib.splice(i, 1);
      }
    }
  }

  getStats() {
    return {
      count: this.memories.length,
      simMu: this.simMu,
      simSigma: this.simSigma,
      surpriseEma: this.surpriseEma,
      noiseFloor: this.surpriseNoiseFloor
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE ENGINE — tanh fold
// ════════════════════════════════════════════════════════════════════════════
class ActiveInferenceEngine {
  constructor(dim) {
    this.dim = dim;
    this.self = randUnit(dim);
    this.velocity = new Float64Array(dim);
    this.prevSelf = clone(this.self);
    this.attractors = new DynamicAttractorField(dim);
    this.actionGain = 10.0;
    this.lastPredicted = null;
    this.activeMemForAction = null;
    this.predictionErrorMagnitude = 0;
  }

  fold(reality, baseLr, memLr, k, memW, velDecay, velGain) {
    const dim = this.dim, s = this.self, r = reality;

    // Surprise via cosine distance
    const cs = cosineSim(s, r);
    let surprise = 1 - Math.abs(cs);

    if (this.lastPredicted) {
      let pe = 1 - Math.abs(cosineSim(r, this.lastPredicted));
      this.predictionErrorMagnitude = pe;
    } else {
      this.predictionErrorMagnitude = surprise;
    }

    const attractorResult = this.attractors.selectAndUpdate(r, s, memLr);
    const activeMem = attractorResult.memory;
    this.activeMemForAction = activeMem;
    const alr = baseLr * (1 + k * surprise);

    // TANH FOLD: u[i] = s[i] + alr * r[i] + memW * m[i], then tanh
    for (let idx = 0; idx < dim; idx++) {
      s[idx] = s[idx] + alr * r[idx] + memW * activeMem[idx];
    }
    tanhVec(s);

    for (let i = 0; i < dim; i++) {
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    }
    this.prevSelf = clone(s);

    return {
      surprise,
      predictionErrorMagnitude: this.predictionErrorMagnitude,
      attractorAction: attractorResult.action,
      memoryCount: attractorResult.totalMemories,
      effectiveSpawn: attractorResult.effectiveSpawn,
      effectiveMerge: attractorResult.effectiveMerge
    };
  }

  act(velScale) {
    const output = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) output[i] = this.self[i] + velScale * this.velocity[i];
    tanhVec(output);
    this.lastPredicted = clone(output);
    if (this.activeMemForAction && this.predictionErrorMagnitude > 0.02) {
      const action = clone(this.activeMemForAction);
      const n = norm(action);
      if (n > 1e-12) for (let i = 0; i < this.dim; i++) action[i] /= n;
      for (let i = 0; i < this.dim; i++) action[i] *= this.predictionErrorMagnitude * this.actionGain;
      return action;
    }
    return new Float64Array(this.dim);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// OUTPUT
// ════════════════════════════════════════════════════════════════════════════
const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

const PARAMS = { baseLr: 0.08, memLr: 0.015, k: 20, memW: 0.15, velDecay: 0.95, velGain: 0.05, velScale: 12.5 };

log('='.repeat(78));
log('FluxCore Step 32 — tanh fold autoregression experiment');
log('Fold: u[i] = s[i] + alr*r[i] + memW*m[i]; s[i] = tanh(u[i])');
log('State space: [-1,1]^d (was unit sphere S^(d-1))');
log('='.repeat(78));

const results = {};

// ════════════════════════════════════════════════════════════════════════════
// TEST: AUTOREGRESSION — does fixed point break?
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('AUTOREGRESSION TEST');
  log('  100 ticks external seed (4-cluster, std=0.3), then 10,000 ticks self-feeding');
  log('-'.repeat(78));

  _rng = 42;
  const DIM = 64;

  // Generate 4-cluster distribution with std=0.3
  const centers = [];
  let base = randUnit(DIM);
  for (let i = 0; i < 4; i++) {
    centers.push(clone(base));
    base = makeOrtho(base, DIM);
  }

  const sampleCluster = () => {
    const ci = Math.floor(rand() * 4);
    const c = centers[ci];
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = c[i] + 0.3 * randN();
    return normalize(v);
  };

  const entity = new ActiveInferenceEngine(DIM);

  // Phase 1: 100 ticks external reality
  log('\n  Phase 1: External reality (100 ticks)');
  for (let t = 0; t < 100; t++) {
    const reality = sampleCluster();
    const res = entity.fold(reality, PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
    entity.act(PARAMS.velScale);
    if (t % 25 === 0) {
      log(`    Tick ${String(t).padStart(4)}: surprise=${res.surprise.toFixed(6)}, mem=${res.memoryCount}, action=${res.attractorAction}`);
    }
  }

  const statsAfterSeed = entity.attractors.getStats();
  log(`\n  After seeding: ${statsAfterSeed.count} memories`);

  // Phase 2: 10,000 ticks autoregression (reality = self state)
  log('\n  Phase 2: Autoregression (10,000 ticks, reality = self state)');
  log(`  ${'Tick'.padStart(6)}  ${'Surprise'.padStart(10)}  ${'Avg100'.padStart(10)}  ${'MemCount'.padStart(8)}  ${'|ds|'.padStart(12)}  Action`);

  const surpriseLog = [];
  const dsLog = [];
  let prevState = clone(entity.self);

  for (let t = 0; t < 10000; t++) {
    // Autoregression: feed self state as reality
    const reality = clone(entity.self);
    const res = entity.fold(reality, PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
    entity.act(PARAMS.velScale);

    surpriseLog.push(res.surprise);

    // |ds| = L2 norm of state change
    let ds = 0;
    for (let i = 0; i < DIM; i++) {
      const d = entity.self[i] - prevState[i];
      ds += d * d;
    }
    ds = Math.sqrt(ds);
    dsLog.push(ds);
    prevState = clone(entity.self);

    if (t % 500 === 0) {
      const startIdx = Math.max(0, surpriseLog.length - 100);
      const avg100 = avg(surpriseLog.slice(startIdx));
      log(`  ${String(t).padStart(6)}  ${res.surprise.toFixed(8).padStart(10)}  ${avg100.toFixed(8).padStart(10)}  ${String(res.memoryCount).padStart(8)}  ${ds.toExponential(4).padStart(12)}  ${res.attractorAction}`);
    }
  }

  // Classification
  const late = surpriseLog.slice(1000);
  const avgLate = avg(late);
  const maxLate = Math.max(...late);
  const minLate = Math.min(...late);
  const lateDs = dsLog.slice(1000);
  const avgDs = avg(lateDs);

  log('\n  --- Classification ---');
  log(`  Avg surprise (tick 1000+): ${avgLate.toExponential(6)}`);
  log(`  Max surprise (tick 1000+): ${maxLate.toExponential(6)}`);
  log(`  Min surprise (tick 1000+): ${minLate.toExponential(6)}`);
  log(`  Avg |ds| (tick 1000+):     ${avgDs.toExponential(6)}`);

  const fixedPointBroken = avgLate > 0.001;
  results['autoregression'] = fixedPointBroken ? 'BROKEN' : 'FIXED_POINT';
  log(`\n  Result: ${fixedPointBroken ? 'FIXED POINT BROKEN' : 'FIXED POINT (surprise → 0)'}`);

  if (fixedPointBroken) {
    // Characterize the behavior
    // Check for oscillation: look at autocorrelation of surprise at lag 1
    let sumProd = 0, sumSq = 0;
    const mean = avgLate;
    for (let i = 1; i < late.length; i++) {
      sumProd += (late[i] - mean) * (late[i - 1] - mean);
      sumSq += (late[i] - mean) * (late[i] - mean);
    }
    const autocorr = sumSq > 0 ? sumProd / sumSq : 0;

    // Check for growth: compare first and last quarter
    const q1 = avg(late.slice(0, Math.floor(late.length / 4)));
    const q4 = avg(late.slice(Math.floor(3 * late.length / 4)));
    const growthRatio = q4 / (q1 + 1e-15);

    // Check |ds| growth
    const dsQ1 = avg(lateDs.slice(0, Math.floor(lateDs.length / 4)));
    const dsQ4 = avg(lateDs.slice(Math.floor(3 * lateDs.length / 4)));

    // Variance of surprise
    let variance = 0;
    for (const s of late) variance += (s - avgLate) * (s - avgLate);
    variance /= late.length;
    const cv = Math.sqrt(variance) / (avgLate + 1e-15);

    log(`\n  Characterization:`);
    log(`    Autocorrelation (lag-1): ${autocorr.toFixed(4)}`);
    log(`    Surprise Q1 avg: ${q1.toExponential(4)}, Q4 avg: ${q4.toExponential(4)}, ratio: ${growthRatio.toFixed(4)}`);
    log(`    |ds| Q1 avg: ${dsQ1.toExponential(4)}, Q4 avg: ${dsQ4.toExponential(4)}`);
    log(`    Surprise CV: ${cv.toFixed(4)}`);

    if (growthRatio > 2) log(`    → GROWING (surprise increasing over time)`);
    else if (growthRatio < 0.5) log(`    → DECAYING BACK (surprise decreasing)`);
    else if (autocorr < -0.3) log(`    → OSCILLATION (negative autocorrelation)`);
    else if (cv > 1.0) log(`    → CHAOTIC (high coefficient of variation)`);
    else log(`    → SUSTAINED (stable non-zero surprise)`);
  }

  // State norm analysis
  const stateNorm = norm(entity.self);
  log(`\n  Final state norm: ${stateNorm.toFixed(6)} (was ~1.0 on sphere, now in [-1,1]^d)`);
  log(`  Final memories: ${entity.attractors.getStats().count}`);
}

// ════════════════════════════════════════════════════════════════════════════
// T1: Basic Attractor Genesis
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('T1: Basic Attractor Genesis (500 ticks, distribution A)');
  log('-'.repeat(78));

  _rng = 42;
  const DIM = 64;
  const base = randUnit(DIM);
  const NOISE = 0.001;
  const genA = () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = base[i] + NOISE * randN();
    return normalize(v);
  };

  const entity = new ActiveInferenceEngine(DIM);
  let lastSurprise = 0;
  for (let t = 0; t < 500; t++) {
    const res = entity.fold(genA(), PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
    entity.act(PARAMS.velScale);
    lastSurprise = res.surprise;
  }

  const memCount = entity.attractors.getStats().count;
  const pass = memCount >= 2 && memCount <= 8 && lastSurprise < 0.05;
  results['T1_genesis'] = pass;
  log(`  Final: ${memCount} memories, surprise=${lastSurprise.toFixed(6)}`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'} (expected 2-8 memories, surprise < 0.05)`);
}

// ════════════════════════════════════════════════════════════════════════════
// T2: Alternating A/B
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('T2: Alternating A/B (200 ticks each, 3 cycles)');
  log('-'.repeat(78));

  _rng = 123;
  const DIM = 64;
  const bA = randUnit(DIM);
  const bB = makeOrtho(bA, DIM);
  const NOISE = 0.001;
  const genA = () => { const v = new Float64Array(DIM); for (let i = 0; i < DIM; i++) v[i] = bA[i] + NOISE * randN(); return normalize(v); };
  const genB = () => { const v = new Float64Array(DIM); for (let i = 0; i < DIM; i++) v[i] = bB[i] + NOISE * randN(); return normalize(v); };

  const entity = new ActiveInferenceEngine(DIM);
  let surpriseA = [], surpriseB = [];

  for (let cycle = 0; cycle < 3; cycle++) {
    const aS = [];
    for (let t = 0; t < 200; t++) {
      const res = entity.fold(genA(), PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
      entity.act(PARAMS.velScale);
      if (t >= 150) aS.push(res.surprise);
    }
    surpriseA.push(avg(aS));

    const bS = [];
    for (let t = 0; t < 200; t++) {
      const res = entity.fold(genB(), PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
      entity.act(PARAMS.velScale);
      if (t >= 150) bS.push(res.surprise);
    }
    surpriseB.push(avg(bS));

    log(`  Cycle ${cycle + 1}: A_surp=${surpriseA[cycle].toFixed(6)}, B_surp=${surpriseB[cycle].toFixed(6)}`);
  }

  // After training on both, A should have lower surprise than B's first exposure
  const lastA = surpriseA[surpriseA.length - 1];
  const firstB = surpriseB[0];
  // Actually the test is: surprise for A < surprise for B on the same cycle
  // i.e., the familiar distribution should have lower surprise
  const pass = surpriseA[2] < surpriseB[0];
  results['T2_alternating'] = pass;
  log(`  A_cycle3 (${surpriseA[2].toFixed(6)}) < B_cycle1 (${surpriseB[0].toFixed(6)}): ${pass ? 'PASS' : 'FAIL'}`);
}

// ════════════════════════════════════════════════════════════════════════════
// T3: 10% Noise Robustness
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('T3: Noise Robustness (Gaussian noise std=0.05)');
  log('-'.repeat(78));

  _rng = 42;
  const DIM = 64;
  const base = randUnit(DIM);

  const genNoisy = () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = base[i] + 0.05 * randN();
    return normalize(v);
  };

  const entity = new ActiveInferenceEngine(DIM);
  const surprises = [];

  for (let t = 0; t < 1000; t++) {
    const res = entity.fold(genNoisy(), PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
    entity.act(PARAMS.velScale);
    if (t >= 500) surprises.push(res.surprise);
  }

  const avgSurp = avg(surprises);
  const pass = avgSurp < 0.1;
  results['T3_noise'] = pass;
  log(`  Avg surprise (last 500 ticks): ${avgSurp.toFixed(6)}`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'} (expected < 0.1)`);
}

// ════════════════════════════════════════════════════════════════════════════
// T4: A→B Switch (spike then recovery)
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('T4: A→B Switch (300 ticks A, then switch to B)');
  log('-'.repeat(78));

  _rng = 456;
  const DIM = 64;
  const bA = randUnit(DIM);
  const bB = makeOrtho(bA, DIM);
  const NOISE = 0.001;
  const genA = () => { const v = new Float64Array(DIM); for (let i = 0; i < DIM; i++) v[i] = bA[i] + NOISE * randN(); return normalize(v); };
  const genB = () => { const v = new Float64Array(DIM); for (let i = 0; i < DIM; i++) v[i] = bB[i] + NOISE * randN(); return normalize(v); };

  const entity = new ActiveInferenceEngine(DIM);

  // Phase A: 300 ticks
  let lastA = 0;
  for (let t = 0; t < 300; t++) {
    const res = entity.fold(genA(), PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
    entity.act(PARAMS.velScale);
    lastA = res.surprise;
  }

  // Switch to B
  let spikeDetected = false;
  let recoveryDetected = false;
  let maxSpike = 0;
  const bSurprises = [];

  for (let t = 0; t < 500; t++) {
    const res = entity.fold(genB(), PARAMS.baseLr, PARAMS.memLr, PARAMS.k, PARAMS.memW, PARAMS.velDecay, PARAMS.velGain);
    entity.act(PARAMS.velScale);
    bSurprises.push(res.surprise);

    if (t < 10 && res.surprise > lastA * 2) spikeDetected = true;
    if (t < 10 && res.surprise > maxSpike) maxSpike = res.surprise;
  }

  const recoveryAvg = avg(bSurprises.slice(-50));
  recoveryDetected = recoveryAvg < maxSpike * 0.5;

  const pass = spikeDetected && recoveryDetected;
  results['T4_switch'] = pass;
  log(`  Last A surprise: ${lastA.toFixed(6)}`);
  log(`  Spike on switch: ${maxSpike.toFixed(6)} (${spikeDetected ? 'detected' : 'NOT detected'})`);
  log(`  Recovery avg (last 50): ${recoveryAvg.toFixed(6)} (${recoveryDetected ? 'recovered' : 'NOT recovered'})`);
  log(`  -> ${pass ? 'PASS' : 'FAIL'}`);
}

// ════════════════════════════════════════════════════════════════════════════
// SUMMARY
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '='.repeat(78));
  log('SUMMARY — Step 32: tanh fold autoregression');
  log('='.repeat(78));

  log('\nFold equation:');
  log('  u[i] = s[i] + alr * r[i] + memW * m[i]');
  log('  s[i] = tanh(u[i])       ← was: s = normalize(u)');
  log('  State space: [-1,1]^d   ← was: unit sphere S^(d-1)');
  log('');

  log('Results:');
  log('-'.repeat(50));

  for (const [test, result] of Object.entries(results)) {
    const display = typeof result === 'boolean' ? (result ? 'PASS' : 'FAIL') : result;
    log(`  ${test.padEnd(25)} ${display}`);
  }

  const regressionTests = ['T1_genesis', 'T2_alternating', 'T3_noise', 'T4_switch'];
  const regressionPassed = regressionTests.filter(t => results[t] === true).length;
  log(`\n  Regression: ${regressionPassed}/${regressionTests.length} passed`);
  log(`  Autoregression: ${results['autoregression']}`);

  log('='.repeat(78));
}

log('\n');
writeFileSync('B:/M/avir/research/fluxcore/results/tanh_autoregression.txt', allLines.join('\n') + '\n');
log('Captured to results/tanh_autoregression.txt');
