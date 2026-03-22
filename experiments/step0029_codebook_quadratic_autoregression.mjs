/**
 * FluxCore Step 29 — Quadratic Cross-Dimensional Self-Interaction
 *
 * Adds quadratic term to canonical fold:
 *   u[i] = s[i] + alr * r[i] + memW * m[i] + qW * s[i] * s[(i+1) % d]
 *
 * Tests:
 *   1. Autoregression: does quadratic term break fixed-point collapse?
 *   2. Regression: T1-T4 perception checks still pass?
 */

import { writeFileSync } from 'fs';

// ════════════════════════════════════════════════════════════════════════════
// UTILITIES (copied from canonical)
// ════════════════════════════════════════════════════════════════════════════
let _rng = 42;
const rand = () => { _rng = (Math.imul(1664525, _rng) + 1013904223) >>> 0; return (_rng >>> 0) / 4294967296; };
const randN = () => { const u = rand() + 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const normalize = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; const n = Math.sqrt(s) + 1e-12; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
const randUnit = d => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = randN(); return normalize(v); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const clone = v => new Float64Array(v);
const avg = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
function makeOrtho(base, dim) {
  const r = randUnit(dim), p = dot(r, base), b = new Float64Array(dim);
  for (let i = 0; i < dim; i++) b[i] = r[i] - p * base[i];
  return normalize(b);
}

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD (identical to canonical)
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
    let surprise = 0;
    for (let i = 0; i < this.dim; i++) surprise += Math.abs(self[i] - reality[i]);
    surprise /= this.dim;
    this.surpriseEma = 0.99 * this.surpriseEma + 0.01 * surprise;
    if (surprise < this.surpriseNoiseFloor) this.surpriseNoiseFloor = surprise;

    let bestIdx = -1, bestSim = -1;
    for (let i = 0; i < this.memories.length; i++) {
      const sim = Math.abs(dot(this.memories[i], reality));
      if (sim > bestSim) { bestSim = sim; bestIdx = i; }
    }

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

    if (bestSim < effectiveSpawn) {
      const newMem = clone(reality);
      const newIdx = this.memories.length;
      this.memories.push(newMem);
      this.memoryMeta[newIdx] = { lastUsed: this.tick, useCount: 1, birthTick: this.tick };
      this.memoryContrib[newIdx] = 0;
      return { memory: newMem, idx: newIdx, surprise, action: 'spawn', totalMemories: this.memories.length, effectiveSpawn, effectiveMerge };
    }

    const activeMem = this.memories[bestIdx];
    const contribution = bestSim * (1 - surprise);
    const contribAlpha = 0.05;
    this.memoryContrib[bestIdx] = (1 - contribAlpha) * (this.memoryContrib[bestIdx] || 0) + contribAlpha * contribution;
    for (let i = 0; i < this.dim; i++) this.memories[bestIdx][i] += memLr * reality[i];
    normalize(this.memories[bestIdx]);
    this.memoryMeta[bestIdx] = {
      lastUsed: this.tick,
      useCount: (this.memoryMeta[bestIdx]?.useCount || 0) + 1,
      birthTick: this.memoryMeta[bestIdx]?.birthTick || this.tick
    };
    this._mergeConverged(effectiveMerge);
    this._pruneMemories();
    return { memory: activeMem, idx: bestIdx, surprise, action: 'use', totalMemories: this.memories.length, effectiveSpawn, effectiveMerge };
  }

  _mergeConverged(threshold) {
    for (let i = 0; i < this.memories.length; i++) {
      for (let j = i + 1; j < this.memories.length; j++) {
        if (Math.abs(dot(this.memories[i], this.memories[j])) > threshold) {
          const mi = this.memoryMeta[i] || { useCount: 1 };
          const mj = this.memoryMeta[j] || { useCount: 1 };
          const total = mi.useCount + mj.useCount;
          for (let k = 0; k < this.dim; k++)
            this.memories[i][k] = (mi.useCount * this.memories[i][k] + mj.useCount * this.memories[j][k]) / total;
          normalize(this.memories[i]);
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
// QUADRATIC ACTIVE INFERENCE ENGINE
// ════════════════════════════════════════════════════════════════════════════
class QuadraticInferenceEngine {
  constructor(dim, qW) {
    this.dim = dim;
    this.qW = qW;
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
    let surprise = 0;
    for (let i = 0; i < dim; i++) surprise += Math.abs(s[i] - r[i]);
    surprise /= dim;

    if (this.lastPredicted) {
      let pe = 0;
      for (let i = 0; i < dim; i++) pe += Math.abs(r[i] - this.lastPredicted[i]);
      this.predictionErrorMagnitude = pe / dim;
    } else {
      this.predictionErrorMagnitude = surprise;
    }

    const attractorResult = this.attractors.selectAndUpdate(r, s, memLr);
    const activeMem = attractorResult.memory;
    this.activeMemForAction = activeMem;
    const alr = baseLr * (1 + k * surprise);

    // QUADRATIC FOLD: canonical + cross-dimensional self-interaction
    // u[i] = s[i] + alr * r[i] + memW * m[i] + qW * s[i] * s[(i+1) % d]
    for (let idx = 0; idx < dim; idx++) {
      s[idx] = s[idx] + alr * r[idx] + memW * activeMem[idx] + this.qW * s[idx] * s[(idx + 1) % dim];
    }
    normalize(s);

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
    normalize(output);
    this.lastPredicted = clone(output);
    if (this.activeMemForAction && this.predictionErrorMagnitude > 0.02) {
      const action = clone(this.activeMemForAction);
      normalize(action);
      for (let i = 0; i < this.dim; i++) action[i] *= this.predictionErrorMagnitude * this.actionGain;
      return action;
    }
    return new Float64Array(this.dim);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Canonical engine (no quadratic term, for regression baseline)
// ════════════════════════════════════════════════════════════════════════════
class CanonicalInferenceEngine {
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
    let surprise = 0;
    for (let i = 0; i < dim; i++) surprise += Math.abs(s[i] - r[i]);
    surprise /= dim;
    if (this.lastPredicted) {
      let pe = 0;
      for (let i = 0; i < dim; i++) pe += Math.abs(r[i] - this.lastPredicted[i]);
      this.predictionErrorMagnitude = pe / dim;
    } else {
      this.predictionErrorMagnitude = surprise;
    }
    const attractorResult = this.attractors.selectAndUpdate(r, s, memLr);
    const activeMem = attractorResult.memory;
    this.activeMemForAction = activeMem;
    const alr = baseLr * (1 + k * surprise);
    for (let idx = 0; idx < dim; idx++) {
      s[idx] = s[idx] + alr * r[idx] + memW * activeMem[idx];
    }
    normalize(s);
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
    normalize(output);
    this.lastPredicted = clone(output);
    if (this.activeMemForAction && this.predictionErrorMagnitude > 0.02) {
      const action = clone(this.activeMemForAction);
      normalize(action);
      for (let i = 0; i < this.dim; i++) action[i] *= this.predictionErrorMagnitude * this.actionGain;
      return action;
    }
    return new Float64Array(this.dim);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// PARAMETERS
// ════════════════════════════════════════════════════════════════════════════
const DIM = 64;
const baseLr = 0.08, memLr = 0.015, k = 20, memW = 0.15;
const velDecay = 0.95, velGain = 0.05, velScale = 12.5;
const QW_VALUES = [0.1, 0.5, 1.0];
const SEED_TICKS = 100;
const AUTO_TICKS = 10000;
const NOISE_STD = 0.3;

// ════════════════════════════════════════════════════════════════════════════
// OUTPUT
// ════════════════════════════════════════════════════════════════════════════
const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

log('='.repeat(78));
log('FluxCore Step 29 — Quadratic Cross-Dimensional Self-Interaction');
log('Fold: u[i] = s[i] + alr*r[i] + memW*m[i] + qW*s[i]*s[(i+1)%d]');
log('='.repeat(78));
log('');
log('Parameters:');
log(`  DIM=${DIM}, baseLr=${baseLr}, memLr=${memLr}, k=${k}, memW=${memW}`);
log(`  velDecay=${velDecay}, velGain=${velGain}, velScale=${velScale}`);
log(`  qW values: ${QW_VALUES.join(', ')}`);
log(`  Seed ticks: ${SEED_TICKS}, Auto ticks: ${AUTO_TICKS}`);
log(`  Cluster noise std: ${NOISE_STD}`);

// ════════════════════════════════════════════════════════════════════════════
// DISTRIBUTION GENERATORS
// ════════════════════════════════════════════════════════════════════════════
function makeClusterDist(dim, nClusters, std) {
  const centers = [];
  for (let i = 0; i < nClusters; i++) centers.push(randUnit(dim));
  return () => {
    const c = centers[Math.floor(rand() * nClusters)];
    const v = new Float64Array(dim);
    for (let i = 0; i < dim; i++) v[i] = c[i] + std * randN();
    return normalize(v);
  };
}

function makeOrthoDist(baseDist) {
  // Generate a distribution orthogonal to the first cluster center
  // Just use fresh random clusters
  return makeClusterDist(DIM, 4, NOISE_STD);
}

// ════════════════════════════════════════════════════════════════════════════
// TEST 1: AUTOREGRESSION per qW
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('TEST 1: AUTOREGRESSION — Does quadratic term break fixed-point collapse?');
log('='.repeat(78));

const autoResults = {};
let bestQW = null; // first qW that breaks fixed point

for (const qW of QW_VALUES) {
  log('\n' + '-'.repeat(78));
  log(`qW = ${qW}`);
  log('-'.repeat(78));

  _rng = 42;
  const distA = makeClusterDist(DIM, 4, NOISE_STD);
  const engine = new QuadraticInferenceEngine(DIM, qW);

  // Phase 1: Seed with distribution A for 100 ticks
  log('\n  Seeding (100 ticks, distribution A)...');
  for (let t = 0; t < SEED_TICKS; t++) {
    const reality = distA();
    engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
  }

  const seedStats = engine.attractors.getStats();
  log(`  After seed: ${seedStats.count} memories, surprise=${seedStats.surpriseEma.toFixed(6)}`);

  // Phase 2: Autoregression — feed fold's own state as reality
  log('\n  Autoregressing (10,000 ticks, reality = self state)...');
  log(`  ${'Tick'.padStart(6)} ${'Surprise'.padStart(10)} ${'Avg100'.padStart(10)} ${'Mems'.padStart(5)} ${'|ds|'.padStart(12)}`);
  log('  ' + '-'.repeat(50));

  const surpriseLog = [];
  const stateSamples = []; // first 8 dims at ticks 100,200,...,2000

  let prevState = clone(engine.self);
  let fixedPointBroken = false;
  let dynamicsType = 'FIXED POINT';

  // Track for dynamics classification
  const surpriseAfter1000 = [];
  const dsLog = [];

  for (let t = 0; t < AUTO_TICKS; t++) {
    // Autoregression: reality = current state
    const reality = clone(engine.self);
    const result = engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);

    surpriseLog.push(result.surprise);

    // State change magnitude
    let ds = 0;
    for (let i = 0; i < DIM; i++) ds += (engine.self[i] - prevState[i]) ** 2;
    ds = Math.sqrt(ds);
    dsLog.push(ds);
    prevState = clone(engine.self);

    if (t >= 1000) surpriseAfter1000.push(result.surprise);

    // Sample state at ticks 100,200,...,2000
    if ((t + 1) % 100 === 0 && t + 1 <= 2000) {
      stateSamples.push({ tick: t + 1, dims: Array.from(engine.self.slice(0, 8)) });
    }

    // Log every 500 ticks
    if ((t + 1) % 500 === 0) {
      const avg100 = avg(surpriseLog.slice(-100));
      log(`  ${String(t + 1).padStart(6)} ${result.surprise.toFixed(6).padStart(10)} ${avg100.toFixed(6).padStart(10)} ${String(result.memoryCount).padStart(5)} ${ds.toExponential(4).padStart(12)}`);
    }
  }

  // Classify dynamics
  const avgSurprisePost1000 = avg(surpriseAfter1000);
  const avgDsPost1000 = avg(dsLog.slice(1000));

  if (avgSurprisePost1000 > 0.001) {
    fixedPointBroken = true;

    // Check for oscillation: look at sign changes in ds
    const dsLate = dsLog.slice(5000);
    let signChanges = 0;
    for (let i = 1; i < dsLate.length; i++) {
      if ((dsLate[i] - avgDsPost1000) * (dsLate[i - 1] - avgDsPost1000) < 0) signChanges++;
    }
    const oscRate = signChanges / dsLate.length;

    // Check for divergence: is ds growing?
    const dsFirst = avg(dsLog.slice(1000, 2000));
    const dsLast = avg(dsLog.slice(9000));
    const dsGrowth = dsLast / (dsFirst + 1e-15);

    if (dsGrowth > 10) {
      dynamicsType = 'DIVERGENCE';
    } else if (oscRate > 0.3) {
      dynamicsType = 'OSCILLATION';
    } else if (dsGrowth > 1.5) {
      dynamicsType = 'CHAOS';
    } else {
      dynamicsType = 'TRAJECTORY';
    }
  }

  log(`\n  Avg surprise after tick 1000: ${avgSurprisePost1000.toFixed(8)}`);
  log(`  Avg |ds| after tick 1000:     ${avgDsPost1000.toExponential(4)}`);
  log(`  Fixed point broken:           ${fixedPointBroken ? 'YES' : 'NO'}`);
  log(`  Dynamics classification:      ${dynamicsType}`);

  // State sample table
  log('\n  State samples (first 8 dims):');
  log(`  ${'Tick'.padStart(5)} ${Array.from({length: 8}, (_, i) => `d${i}`.padStart(9)).join(' ')}`);
  log('  ' + '-'.repeat(80));
  for (const s of stateSamples) {
    log(`  ${String(s.tick).padStart(5)} ${s.dims.map(d => d.toFixed(5).padStart(9)).join(' ')}`);
  }

  autoResults[qW] = {
    fixedPointBroken,
    dynamicsType,
    avgSurprisePost1000,
    avgDsPost1000,
    memCount: engine.attractors.getStats().count
  };

  if (fixedPointBroken && bestQW === null) bestQW = qW;
}

// ════════════════════════════════════════════════════════════════════════════
// TEST 2: REGRESSION CHECK
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('TEST 2: REGRESSION CHECK — Does quadratic fold still perceive correctly?');
log('='.repeat(78));

const regressionQW = bestQW !== null ? bestQW : 0.1;
log(`\n  Using qW = ${regressionQW} (${bestQW !== null ? 'first to break fixed point' : 'default, all collapsed'})`);

// Also run canonical for comparison
const regressionResults = {};

function runRegressionSuite(engineFactory, label) {
  log(`\n  --- ${label} ---`);
  const results = {};

  // T1: 100 ticks distribution A, surprise < 0.01 at end
  {
    _rng = 42;
    const distA = makeClusterDist(DIM, 4, NOISE_STD);
    const eng = engineFactory();
    let lastSurprise = 0;
    for (let t = 0; t < 100; t++) {
      const res = eng.fold(distA(), baseLr, memLr, k, memW, velDecay, velGain);
      eng.act(velScale);
      lastSurprise = res.surprise;
    }
    const pass = lastSurprise < 0.01;
    results.T1 = { pass, lastSurprise };
    log(`    T1 (basic convergence):  surprise=${lastSurprise.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
  }

  // T2: 100 ticks alternating A/B, check fold tracks each
  {
    _rng = 42;
    const distA = makeClusterDist(DIM, 4, NOISE_STD);
    _rng = 999;
    const distB = makeClusterDist(DIM, 4, NOISE_STD);
    const eng = engineFactory();
    let surpriseA = [], surpriseB = [];
    for (let t = 0; t < 100; t++) {
      const dist = t % 2 === 0 ? distA : distB;
      const res = eng.fold(dist(), baseLr, memLr, k, memW, velDecay, velGain);
      eng.act(velScale);
      if (t >= 80) {
        if (t % 2 === 0) surpriseA.push(res.surprise);
        else surpriseB.push(res.surprise);
      }
    }
    const avgA = avg(surpriseA), avgB = avg(surpriseB);
    // Both should be tracking — surprise should be moderate, not divergent
    const pass = avgA < 0.1 && avgB < 0.1;
    results.T2 = { pass, avgA, avgB };
    log(`    T2 (alternating A/B):    avgA=${avgA.toFixed(6)} avgB=${avgB.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
  }

  // T3: 100 ticks dist A with 10% random noise ticks
  {
    _rng = 42;
    const distA = makeClusterDist(DIM, 4, NOISE_STD);
    const eng = engineFactory();
    let lastSurprises = [];
    for (let t = 0; t < 100; t++) {
      const useNoise = rand() < 0.1;
      const reality = useNoise ? randUnit(DIM) : distA();
      const res = eng.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
      eng.act(velScale);
      if (t >= 80 && !useNoise) lastSurprises.push(res.surprise);
    }
    const avgS = avg(lastSurprises);
    const pass = avgS < 0.02;
    results.T3 = { pass, avgSurprise: avgS };
    log(`    T3 (10% noise robust):   avgSurprise=${avgS.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
  }

  // T4: 100 ticks dist A, switch to dist B at tick 50, check spike then recovery
  {
    _rng = 42;
    const distA = makeClusterDist(DIM, 4, NOISE_STD);
    _rng = 888;
    const distB = makeClusterDist(DIM, 4, NOISE_STD);
    const eng = engineFactory();
    let spikeDetected = false;
    let recoverySurprises = [];
    for (let t = 0; t < 100; t++) {
      const dist = t < 50 ? distA : distB;
      const res = eng.fold(dist(), baseLr, memLr, k, memW, velDecay, velGain);
      eng.act(velScale);
      if (t === 51) spikeDetected = res.surprise > 0.005;
      if (t >= 90) recoverySurprises.push(res.surprise);
    }
    const recoveryAvg = avg(recoverySurprises);
    const pass = spikeDetected && recoveryAvg < 0.02;
    results.T4 = { pass, spikeDetected, recoveryAvg };
    log(`    T4 (A->B switch):        spike=${spikeDetected} recovery=${recoveryAvg.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
  }

  // Attractor genesis: 200 ticks random distributions, count memories
  {
    _rng = 42;
    const eng = engineFactory();
    for (let t = 0; t < 200; t++) {
      // Random distribution each tick
      const reality = randUnit(DIM);
      eng.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
      eng.act(velScale);
    }
    const memCount = eng.attractors.getStats().count;
    results.genesis = { memCount };
    log(`    Attractor genesis:       ${memCount} memories formed in 200 random ticks`);
  }

  return results;
}

// Run canonical baseline
const canonicalResults = runRegressionSuite(
  () => new CanonicalInferenceEngine(DIM),
  `Canonical (qW=0)`
);

// Run quadratic
const quadResults = runRegressionSuite(
  () => new QuadraticInferenceEngine(DIM, regressionQW),
  `Quadratic (qW=${regressionQW})`
);

// Compare
log('\n  --- Regression Comparison ---');
log(`  ${'Test'.padEnd(8)} ${'Canonical'.padEnd(12)} ${'Quadratic'.padEnd(12)} ${'Degradation'.padEnd(12)}`);
log('  ' + '-'.repeat(50));

const tests = ['T1', 'T2', 'T3', 'T4'];
let anyDegradation = false;
for (const t of tests) {
  const cp = canonicalResults[t].pass ? 'PASS' : 'FAIL';
  const qp = quadResults[t].pass ? 'PASS' : 'FAIL';
  const deg = (canonicalResults[t].pass && !quadResults[t].pass) ? 'DEGRADED' : 'OK';
  if (deg === 'DEGRADED') anyDegradation = true;
  log(`  ${t.padEnd(8)} ${cp.padEnd(12)} ${qp.padEnd(12)} ${deg.padEnd(12)}`);
}

log(`\n  Genesis: canonical=${canonicalResults.genesis.memCount} mems, quadratic=${quadResults.genesis.memCount} mems`);

// ════════════════════════════════════════════════════════════════════════════
// OVERALL ASSESSMENT
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('OVERALL ASSESSMENT');
log('='.repeat(78));

log('\nAutoregression results per qW:');
log(`  ${'qW'.padEnd(6)} ${'Fixed Point'.padEnd(14)} ${'Dynamics'.padEnd(14)} ${'Avg Surprise'.padEnd(14)} ${'Avg |ds|'.padEnd(14)} ${'Mems'.padEnd(5)}`);
log('  ' + '-'.repeat(70));
for (const qW of QW_VALUES) {
  const r = autoResults[qW];
  log(`  ${String(qW).padEnd(6)} ${(r.fixedPointBroken ? 'BROKEN' : 'INTACT').padEnd(14)} ${r.dynamicsType.padEnd(14)} ${r.avgSurprisePost1000.toFixed(8).padEnd(14)} ${r.avgDsPost1000.toExponential(4).padEnd(14)} ${String(r.memCount).padEnd(5)}`);
}

log(`\nRegression (qW=${regressionQW}):`);
for (const t of tests) {
  log(`  ${t}: ${quadResults[t].pass ? 'PASS' : 'FAIL'}`);
}
log(`  Attractor genesis: ${quadResults.genesis.memCount} memories`);
log(`  Any degradation vs canonical: ${anyDegradation ? 'YES' : 'NO'}`);

const overallVerdict = bestQW !== null
  ? `QUADRATIC TERM BREAKS FIXED POINT at qW=${bestQW} (${autoResults[bestQW].dynamicsType}). Regression: ${anyDegradation ? 'DEGRADED' : 'INTACT'}.`
  : `QUADRATIC TERM DOES NOT BREAK FIXED POINT at any tested qW. All collapse to fixed point.`;

log(`\nVERDICT: ${overallVerdict}`);
log('='.repeat(78));

writeFileSync('B:/M/avir/research/fluxcore/results/quadratic_autoregression.txt', allLines.join('\n') + '\n');
log('\nCaptured to results/quadratic_autoregression.txt');
