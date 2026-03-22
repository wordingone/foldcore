/**
 * Step 30 — Tangent-Space Decomposition with Alpha > 1 Instability Experiment
 *
 * Tests whether amplifying the perpendicular component of the fold update
 * (alpha > 1) breaks the fixed-point attractor during autoregression.
 *
 * Modification: replaces s = normalize(u) with tangent-space decomposition:
 *   uParallel = dot(u, sPrev) * sPrev
 *   uPerp = u - uParallel
 *   uMixed = uParallel + alpha * uPerp
 *   s = normalize(uMixed)
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
// ACTIVE INFERENCE ENGINE — tangent-space decomposition variant
// ════════════════════════════════════════════════════════════════════════════
class ActiveInferenceEngine {
  constructor(dim, alpha) {
    this.dim = dim;
    this.alpha = alpha;
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

    // Save sPrev BEFORE update
    const sPrev = clone(s);

    // Compute u (canonical fold weights)
    const u = new Float64Array(dim);
    for (let idx = 0; idx < dim; idx++) {
      u[idx] = s[idx] + alr * r[idx] + memW * activeMem[idx];
    }

    // Check for NaN/Inf in u
    let hasNaN = false;
    for (let i = 0; i < dim; i++) {
      if (!isFinite(u[i])) { hasNaN = true; break; }
    }
    if (hasNaN) {
      return { surprise, predictionErrorMagnitude: this.predictionErrorMagnitude, attractorAction: attractorResult.action, memoryCount: attractorResult.totalMemories, diverged: true };
    }

    // TANGENT-SPACE DECOMPOSITION
    const uDotSPrev = dot(u, sPrev);
    const uParallel = new Float64Array(dim);
    const uPerp = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      uParallel[i] = uDotSPrev * sPrev[i];
      uPerp[i] = u[i] - uParallel[i];
    }
    const uMixed = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
      uMixed[i] = uParallel[i] + this.alpha * uPerp[i];
    }

    // Check for NaN/Inf after mixing
    for (let i = 0; i < dim; i++) {
      if (!isFinite(uMixed[i])) {
        return { surprise, predictionErrorMagnitude: this.predictionErrorMagnitude, attractorAction: attractorResult.action, memoryCount: attractorResult.totalMemories, diverged: true };
      }
    }

    // Normalize into state
    for (let i = 0; i < dim; i++) s[i] = uMixed[i];
    normalize(s);

    // Velocity update
    for (let i = 0; i < dim; i++) {
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    }
    this.prevSelf = clone(s);

    // Compute |ds|
    let ds = 0;
    for (let i = 0; i < dim; i++) ds += (s[i] - sPrev[i]) * (s[i] - sPrev[i]);
    ds = Math.sqrt(ds);

    return {
      surprise,
      predictionErrorMagnitude: this.predictionErrorMagnitude,
      attractorAction: attractorResult.action,
      memoryCount: attractorResult.totalMemories,
      effectiveSpawn: attractorResult.effectiveSpawn,
      effectiveMerge: attractorResult.effectiveMerge,
      ds,
      diverged: false
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
// EXPERIMENT PARAMETERS
// ════════════════════════════════════════════════════════════════════════════
const DIM = 64;
const baseLr = 0.08, memLr = 0.015, k = 20, memW = 0.15, velDecay = 0.95, velGain = 0.05, velScale = 12.5;
const ALPHAS = [1.05, 1.1, 1.2, 1.5];
const SEED_TICKS = 100;
const AUTO_TICKS = 10000;
const NOISE = 0.3;

const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

log('='.repeat(78));
log('Step 30 — Tangent-Space Alpha > 1 Instability Experiment');
log('='.repeat(78));
log('');
log('Parameters:');
log(`  DIM=${DIM}, baseLr=${baseLr}, memLr=${memLr}, k=${k}, memW=${memW}`);
log(`  velDecay=${velDecay}, velGain=${velGain}, velScale=${velScale}`);
log(`  Seed ticks: ${SEED_TICKS}, Autoregression ticks: ${AUTO_TICKS}`);
log(`  Alpha values: ${ALPHAS.join(', ')}`);
log('');
log('Tangent-space decomposition:');
log('  uParallel = dot(u, sPrev) * sPrev');
log('  uPerp = u - uParallel');
log('  uMixed = uParallel + alpha * uPerp');
log('  s = normalize(uMixed)');
log('');

// ════════════════════════════════════════════════════════════════════════════
// Generate seeding distribution (4 cluster centroids, std=0.3)
// ════════════════════════════════════════════════════════════════════════════
function generateClusterSample(centers, std, dim) {
  const idx = Math.floor(rand() * centers.length);
  const c = centers[idx];
  const v = new Float64Array(dim);
  for (let i = 0; i < dim; i++) v[i] = c[i] + std * randN();
  return normalize(v);
}

// ════════════════════════════════════════════════════════════════════════════
// TEST 1 — AUTOREGRESSION per alpha
// ════════════════════════════════════════════════════════════════════════════
const autoResults = {};
let firstNonExplodingBreaker = null;

for (const alpha of ALPHAS) {
  log('-'.repeat(78));
  log(`TEST 1: Autoregression — alpha=${alpha}`);
  log('-'.repeat(78));

  _rng = 42;

  // Generate 4 cluster centroids
  const centers = [];
  for (let i = 0; i < 4; i++) centers.push(randUnit(DIM));

  const engine = new ActiveInferenceEngine(DIM, alpha);

  // Seed phase: 100 ticks with distribution A
  log(`\n  Seeding: ${SEED_TICKS} ticks with 4-cluster distribution (std=${NOISE})...`);
  let seedSurprise = 0;
  for (let t = 0; t < SEED_TICKS; t++) {
    const sample = generateClusterSample(centers, NOISE, DIM);
    const res = engine.fold(sample, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
    seedSurprise += res.surprise;
  }
  const seedMemCount = engine.attractors.getStats().count;
  log(`  Seed done: avg surprise=${(seedSurprise / SEED_TICKS).toFixed(6)}, memories=${seedMemCount}`);

  // Autoregression phase: reality = fold's own output (state)
  log(`\n  Autoregressing: ${AUTO_TICKS} ticks (reality = state)...`);
  log(`  ${'Tick'.padStart(6)}  ${'Surprise'.padStart(10)}  ${'Avg100'.padStart(10)}  ${'MemCnt'.padStart(6)}  ${'|ds|'.padStart(12)}`);

  const surpriseHistory = [];
  const trajectoryLog = []; // first 8 dims at ticks 100,200,...,2000
  let diverged = false;
  let divergeTick = -1;
  let classification = 'STABLE';

  for (let t = 0; t < AUTO_TICKS; t++) {
    const autoTick = t + 1;
    // Reality = fold's own state
    const reality = clone(engine.self);
    const res = engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);

    if (res.diverged) {
      diverged = true;
      divergeTick = autoTick;
      classification = 'DIVERGENCE/EXPLOSION';
      log(`  *** DIVERGED at auto tick ${autoTick} — NaN/Inf detected ***`);
      break;
    }

    // Check for NaN in state
    let stateNaN = false;
    for (let i = 0; i < DIM; i++) {
      if (!isFinite(engine.self[i])) { stateNaN = true; break; }
    }
    if (stateNaN) {
      diverged = true;
      divergeTick = autoTick;
      classification = 'DIVERGENCE/EXPLOSION';
      log(`  *** STATE NaN at auto tick ${autoTick} ***`);
      break;
    }

    surpriseHistory.push(res.surprise);

    // Log every 500 ticks
    if (autoTick % 500 === 0) {
      const avg100 = avg(surpriseHistory.slice(-100));
      const ds = res.ds !== undefined ? res.ds : 0;
      log(`  ${String(autoTick).padStart(6)}  ${res.surprise.toFixed(8).padStart(10)}  ${avg100.toFixed(8).padStart(10)}  ${String(res.memoryCount).padStart(6)}  ${ds.toFixed(10).padStart(12)}`);
    }

    // Track trajectory: first 8 dims at ticks 100,200,...,2000
    if (autoTick >= 100 && autoTick <= 2000 && autoTick % 100 === 0) {
      const dims8 = [];
      for (let d = 0; d < 8; d++) dims8.push(engine.self[d]);
      trajectoryLog.push({ tick: autoTick, dims: dims8 });
    }
  }

  // Classify dynamics
  if (!diverged) {
    // Check if surprise stays > 0.001 after 1000 auto ticks
    const lateSuprises = surpriseHistory.slice(1000);
    const lateMean = avg(lateSuprises);
    const fixedPointBroken = lateMean > 0.001;

    if (fixedPointBroken) {
      // Characterize: oscillation, trajectory, chaos
      // Check variance of late surprises
      const lateVar = avg(lateSuprises.map(s => (s - lateMean) * (s - lateMean)));
      const lateStd = Math.sqrt(lateVar);
      const cv = lateMean > 0 ? lateStd / lateMean : 0;

      // Check for periodicity: autocorrelation at lag 1-50
      let maxAutoCorr = 0;
      let bestLag = 0;
      const late500 = surpriseHistory.slice(-500);
      const late500Mean = avg(late500);
      let var0 = 0;
      for (const s of late500) var0 += (s - late500Mean) * (s - late500Mean);
      var0 /= late500.length;
      for (let lag = 1; lag <= 50 && lag < late500.length; lag++) {
        let ac = 0;
        for (let i = 0; i < late500.length - lag; i++) {
          ac += (late500[i] - late500Mean) * (late500[i + lag] - late500Mean);
        }
        ac /= (late500.length - lag) * (var0 + 1e-20);
        if (ac > maxAutoCorr) { maxAutoCorr = ac; bestLag = lag; }
      }

      if (cv < 0.01) {
        classification = 'FIXED POINT BROKEN — steady trajectory (non-zero surprise)';
      } else if (maxAutoCorr > 0.5) {
        classification = `FIXED POINT BROKEN — oscillation (period~${bestLag}, cv=${cv.toFixed(4)})`;
      } else if (cv > 1.0) {
        classification = `FIXED POINT BROKEN — chaos (cv=${cv.toFixed(4)})`;
      } else {
        classification = `FIXED POINT BROKEN — irregular dynamics (cv=${cv.toFixed(4)}, maxAC=${maxAutoCorr.toFixed(4)})`;
      }
    } else {
      classification = 'STABLE — fixed point intact (surprise < 0.001)';
    }
  }

  log(`\n  Classification: ${classification}`);

  if (!diverged && surpriseHistory.length > 0) {
    const finalSurprise = surpriseHistory[surpriseHistory.length - 1];
    const finalAvg100 = avg(surpriseHistory.slice(-100));
    log(`  Final surprise: ${finalSurprise.toFixed(10)}`);
    log(`  Final avg100:   ${finalAvg100.toFixed(10)}`);
    log(`  Final memories: ${engine.attractors.getStats().count}`);
  }

  // Trajectory table
  if (trajectoryLog.length > 0) {
    log(`\n  State trajectory (first 8 dims):`);
    log(`  ${'Tick'.padStart(6)}  ${Array.from({length: 8}, (_, i) => `dim${i}`.padStart(9)).join('  ')}`);
    for (const entry of trajectoryLog) {
      const dimStr = entry.dims.map(d => d.toFixed(5).padStart(9)).join('  ');
      log(`  ${String(entry.tick).padStart(6)}  ${dimStr}`);
    }
  }

  autoResults[alpha] = {
    classification,
    diverged,
    divergeTick,
    finalSurprise: diverged ? NaN : (surpriseHistory.length > 0 ? surpriseHistory[surpriseHistory.length - 1] : NaN),
    finalAvg100: diverged ? NaN : avg(surpriseHistory.slice(-100)),
    finalMemCount: engine.attractors.getStats().count,
    fixedPointBroken: classification.includes('BROKEN') || diverged
  };

  // Track first non-exploding breaker for regression test
  if (!firstNonExplodingBreaker && autoResults[alpha].fixedPointBroken && !diverged) {
    firstNonExplodingBreaker = alpha;
  }

  log('');
}

// ════════════════════════════════════════════════════════════════════════════
// TEST 2 — Regression check (T1-T4)
// ════════════════════════════════════════════════════════════════════════════
const regressionAlpha = firstNonExplodingBreaker || 1.05;

log('='.repeat(78));
log(`TEST 2: Regression Check (T1-T4) — alpha=${regressionAlpha}`);
log('='.repeat(78));

const regressionResults = {};

// T1: Attractor Genesis
{
  log('\n  T1: Attractor Genesis (A->B->C->D->A, DIM=64)');
  _rng = 42;

  // Build environment manually (same as canonical DynamicEnvironment)
  const distributions = [];
  let base = randUnit(DIM);
  for (let i = 0; i < 4; i++) {
    distributions.push(clone(base));
    base = makeOrtho(base, DIM);
  }
  let currentDist = 0;
  let envTick = 0;

  const engine = new ActiveInferenceEngine(DIM, regressionAlpha);

  for (let tick = 0; tick < 2000; tick++) {
    if (envTick % 500 === 0 && envTick > 0) {
      currentDist = (currentDist + 1) % distributions.length;
    }
    envTick++;
    const dist = distributions[currentDist];
    const noise = 0.001;
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = dist[i] + noise * randN();
    const reality = normalize(v);

    engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
  }

  const memCount = engine.attractors.getStats().count;
  const pass = memCount >= 2 && memCount <= 8;
  regressionResults['T1_genesis'] = { pass, memCount };
  log(`    Final: ${memCount} memories -> ${pass ? 'PASS' : 'FAIL'} (expected 2-8)`);
}

// T2: Reacquisition (A->B->A)
{
  log('\n  T2: Reacquisition (A->B->A, DIM=64)');
  _rng = 123;
  const NOISE_R = 0.0003;
  const bA = randUnit(DIM), bB = makeOrtho(bA, DIM);
  const makeStatic = (base) => () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = base[i] + NOISE_R * randN();
    return normalize(v);
  };
  const gA = makeStatic(bA), gB = makeStatic(bB);
  const eng = new ActiveInferenceEngine(DIM, regressionAlpha);

  const ea1 = [];
  for (let t = 0; t < 1000; t++) {
    const res = eng.fold(gA(), baseLr, memLr, k, memW, velDecay, velGain);
    if (t < 50) ea1.push(res.surprise);
  }
  for (let t = 0; t < 500; t++) eng.fold(gB(), baseLr, memLr, k, memW, velDecay, velGain);
  const ea2 = [];
  for (let t = 0; t < 1000; t++) {
    const res = eng.fold(gA(), baseLr, memLr, k, memW, velDecay, velGain);
    if (t < 50) ea2.push(res.surprise);
  }

  const m1 = avg(ea1), m2 = avg(ea2);
  const pass = m2 < m1;
  regressionResults['T2_reacq'] = { pass, m1, m2 };
  log(`    A1=${m1.toFixed(6)} A2=${m2.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
}

// T3: Multi-dist (A->B->C->D->A)
{
  log('\n  T3: Multi-dist (A->B->C->D->A, DIM=64)');
  _rng = 456;
  const NOISE_R = 0.0003;
  const bases = [randUnit(DIM)];
  for (let i = 1; i < 4; i++) bases.push(makeOrtho(bases[i-1], DIM));
  const makeStatic = (base) => () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = base[i] + NOISE_R * randN();
    return normalize(v);
  };
  const gens = bases.map(b => makeStatic(b));
  const eng = new ActiveInferenceEngine(DIM, regressionAlpha);

  const ea1 = [];
  for (let t = 0; t < 1000; t++) {
    const res = eng.fold(gens[0](), baseLr, memLr, k, memW, velDecay, velGain);
    if (t < 50) ea1.push(res.surprise);
  }
  for (let p = 1; p < 4; p++) {
    for (let t = 0; t < 500; t++) eng.fold(gens[p](), baseLr, memLr, k, memW, velDecay, velGain);
  }
  const ea2 = [];
  for (let t = 0; t < 1000; t++) {
    const res = eng.fold(gens[0](), baseLr, memLr, k, memW, velDecay, velGain);
    if (t < 50) ea2.push(res.surprise);
  }

  const m1 = avg(ea1), m2 = avg(ea2);
  const pass = m2 < m1;
  regressionResults['T3_multi'] = { pass, m1, m2 };
  log(`    A1=${m1.toFixed(6)} A2=${m2.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
  log(`    Final memories: ${eng.attractors.getStats().count}`);
}

// T4: Attractor genesis check
{
  log('\n  T4: Attractor Genesis (standalone, DIM=64)');
  _rng = 42;
  const distributions = [];
  let base = randUnit(DIM);
  for (let i = 0; i < 4; i++) {
    distributions.push(clone(base));
    base = makeOrtho(base, DIM);
  }
  let currentDist = 0;
  let envTick = 0;
  const engine = new ActiveInferenceEngine(DIM, regressionAlpha);

  const surprises = [];
  for (let tick = 0; tick < 2000; tick++) {
    if (envTick % 500 === 0 && envTick > 0) {
      currentDist = (currentDist + 1) % distributions.length;
    }
    envTick++;
    const dist = distributions[currentDist];
    const noise = 0.001;
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = dist[i] + noise * randN();
    const reality = normalize(v);
    const res = engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
    surprises.push(res.surprise);
  }

  const finalSurprise = avg(surprises.slice(-100));
  const memCount = engine.attractors.getStats().count;
  const pass = memCount >= 2 && memCount <= 8;
  regressionResults['T4_genesis2'] = { pass, memCount, finalSurprise };
  log(`    Final: ${memCount} memories, avg100 surprise=${finalSurprise.toFixed(6)} -> ${pass ? 'PASS' : 'FAIL'}`);
}

// Check 10% degradation rule
log('\n  Degradation check (vs canonical baselines):');
log('  Canonical baselines: T1 genesis 2-8 mem, T2 reacq A2<A1, T3 multi A2<A1');
const regressionPass = Object.values(regressionResults).every(r => r.pass);
log(`  Regression: ${regressionPass ? 'ALL PASS — no degradation' : 'SOME FAILURES — degradation detected'}`);

// ════════════════════════════════════════════════════════════════════════════
// SUMMARY
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('SUMMARY');
log('='.repeat(78));

log('\nPer-alpha autoregression results:');
log(`  ${'Alpha'.padEnd(8)}  ${'Classification'.padEnd(55)}  ${'FinalSurp'.padEnd(14)}  ${'Mems'.padEnd(5)}`);
log('  ' + '-'.repeat(86));
for (const alpha of ALPHAS) {
  const r = autoResults[alpha];
  const surp = r.diverged ? 'DIVERGED' : r.finalSurprise.toFixed(10);
  log(`  ${String(alpha).padEnd(8)}  ${r.classification.padEnd(55).substring(0, 55)}  ${surp.padStart(14)}  ${String(r.finalMemCount).padStart(5)}`);
}

log('\nT1-T4 regression (alpha=' + regressionAlpha + '):');
for (const [test, r] of Object.entries(regressionResults)) {
  log(`  ${test.padEnd(15)} ${r.pass ? 'PASS' : 'FAIL'}${r.m1 !== undefined ? ` (A1=${r.m1.toFixed(6)}, A2=${r.m2.toFixed(6)})` : ` (${r.memCount} memories)`}`);
}

log('\nKey findings:');
const breakers = ALPHAS.filter(a => autoResults[a].fixedPointBroken);
const exploded = ALPHAS.filter(a => autoResults[a].diverged);
const collapsed = ALPHAS.filter(a => !autoResults[a].fixedPointBroken && !autoResults[a].diverged);
const brokenNotExploded = breakers.filter(a => !autoResults[a].diverged);

log(`  Fixed point BROKEN:   ${breakers.length > 0 ? breakers.join(', ') : 'none'}`);
log(`  EXPLODED (NaN/Inf):   ${exploded.length > 0 ? exploded.join(', ') : 'none'}`);
log(`  COLLAPSED (stable):   ${collapsed.length > 0 ? collapsed.join(', ') : 'none'}`);
log(`  Broken (not exploded):${brokenNotExploded.length > 0 ? ' ' + brokenNotExploded.join(', ') : ' none'}`);
log(`  Regression (alpha=${regressionAlpha}): ${regressionPass ? 'PASS' : 'FAIL'}`);

log('\n' + '='.repeat(78));

// Write results
writeFileSync('B:/M/avir/research/fluxcore/results/alpha_instability.txt', allLines.join('\n') + '\n');
log('Captured to results/alpha_instability.txt');
