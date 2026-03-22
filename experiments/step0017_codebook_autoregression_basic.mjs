/**
 * Step 17: Autoregressive Self-Feeding on the Canonical Fold
 *
 * Experiment 1: Seed with distribution A (100 ticks), then disconnect reality
 *   and feed fold output back as reality for 10,000 ticks. Classify dynamics.
 *
 * Experiment 2: Seed A (100 ticks), autoregress 2000, inject B (200 ticks),
 *   autoregress 2000 more. Measure surprise spike and trajectory change.
 */

import { writeFileSync } from 'fs';

// ════════════════════════════════════════════════════════════════════════════
// UTILITIES (from fluxcore_canonical.mjs)
// ════════════════════════════════════════════════════════════════════════════
let _rng = 42;
const rand = () => { _rng = (Math.imul(1664525, _rng) + 1013904223) >>> 0; return (_rng >>> 0) / 4294967296; };
const randN = () => { const u = rand() + 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const normalize = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; const n = Math.sqrt(s) + 1e-12; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
const randUnit = d => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = randN(); return normalize(v); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const clone = v => new Float64Array(v);

function makeOrtho(base, dim) {
  const r = randUnit(dim), p = dot(r, base), b = new Float64Array(dim);
  for (let i = 0; i < dim; i++) b[i] = r[i] - p * base[i];
  return normalize(b);
}

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD (from fluxcore_canonical.mjs)
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
      ages: this.memories.map((_, i) => this.memoryMeta[i] ? this.tick - this.memoryMeta[i].birthTick : 0),
      useCounts: this.memoryMeta.map(m => m?.useCount || 0),
      simMu: this.simMu,
      simSigma: this.simSigma,
      surpriseEma: this.surpriseEma,
      noiseFloor: this.surpriseNoiseFloor
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// ACTIVE INFERENCE ENGINE — canonical fold (from fluxcore_canonical.mjs)
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

    // CANONICAL FOLD: u[i] = s[i] + alr * r[i] + memW * m[i]
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
    return output;
  }

  getSelf() {
    return clone(this.self);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// OUTPUT
// ════════════════════════════════════════════════════════════════════════════
const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

// Hyperparams (same as canonical validation)
const baseLr = 0.08, memLr = 0.015, k = 20, memW = 0.15, velDecay = 0.95, velGain = 0.05, velScale = 12.5;
const DIM = 64;

log('='.repeat(78));
log('Step 17: Autoregressive Self-Feeding — Canonical Fold');
log('='.repeat(78));
log(`DIM=${DIM}, baseLr=${baseLr}, memLr=${memLr}, k=${k}, memW=${memW}`);
log(`velDecay=${velDecay}, velGain=${velGain}, velScale=${velScale}`);
log('');

// ════════════════════════════════════════════════════════════════════════════
// EXPERIMENT 1: Pure Autoregression
// ════════════════════════════════════════════════════════════════════════════
{
  log('-'.repeat(78));
  log('EXPERIMENT 1: Pure Autoregression');
  log('  Seed: 100 ticks with distribution A');
  log('  Autoregress: 10,000 ticks (101-10100), reality = fold output');
  log('-'.repeat(78));

  _rng = 42;
  const engine = new ActiveInferenceEngine(DIM);
  const distA = randUnit(DIM);
  const NOISE = 0.001;

  // Helper: generate noisy sample from distribution
  const sampleA = () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = distA[i] + NOISE * randN();
    return normalize(v);
  };

  // Arrays for analysis
  const allSurprises = [];
  const allMemCounts = [];
  const stateSnapshots = []; // every 100 ticks: first 8 dims of self

  log('\n--- Seed Phase (ticks 1-100) ---');
  for (let t = 1; t <= 100; t++) {
    const reality = sampleA();
    const result = engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
    allSurprises.push(result.surprise);
    allMemCounts.push(result.memoryCount);

    if (t % 25 === 0) {
      log(`  Tick ${String(t).padStart(5)}: surprise=${result.surprise.toFixed(6)}, mem=${result.memoryCount}, action=${result.attractorAction}`);
    }
  }

  // Capture state at tick 100 before autoregression
  const selfAtSeedEnd = engine.getSelf();
  stateSnapshots.push({ tick: 100, dims: Array.from(selfAtSeedEnd.slice(0, 8)) });

  log('\n--- Autoregression Phase (ticks 101-10100) ---');
  let lastOutput = engine.act(velScale); // get output at tick 100 as first autoregressive input

  // Track dynamics metrics
  let prevSelf = engine.getSelf();
  const selfChanges = []; // ||s[t] - s[t-1]|| per tick

  for (let t = 101; t <= 10100; t++) {
    // Feed the fold's own output as reality
    const result = engine.fold(lastOutput, baseLr, memLr, k, memW, velDecay, velGain);
    lastOutput = engine.act(velScale);

    allSurprises.push(result.surprise);
    allMemCounts.push(result.memoryCount);

    // Measure state change
    const curSelf = engine.getSelf();
    let change = 0;
    for (let i = 0; i < DIM; i++) change += Math.abs(curSelf[i] - prevSelf[i]);
    change /= DIM;
    selfChanges.push(change);
    prevSelf = curSelf;

    // Log every 100 ticks
    if ((t - 100) % 100 === 0) {
      stateSnapshots.push({ tick: t, dims: Array.from(curSelf.slice(0, 8)) });
    }

    if ((t - 100) % 500 === 0) {
      const recentSurp = allSurprises.slice(-100);
      const avgSurp = recentSurp.reduce((s, x) => s + x, 0) / recentSurp.length;
      const recentChange = selfChanges.slice(-100);
      const avgChange = recentChange.reduce((s, x) => s + x, 0) / recentChange.length;
      log(`  Tick ${String(t).padStart(5)}: surprise=${result.surprise.toFixed(6)}, avg100=${avgSurp.toFixed(6)}, mem=${result.memoryCount}, |ds|=${avgChange.toFixed(8)}`);
    }
  }

  // Analysis
  log('\n--- Experiment 1 Analysis ---');

  // Surprise trajectory
  const seedSurprises = allSurprises.slice(0, 100);
  const autoSurprises = allSurprises.slice(100);
  const seedAvg = seedSurprises.reduce((s, x) => s + x, 0) / seedSurprises.length;
  const autoFirst100 = autoSurprises.slice(0, 100);
  const autoLast100 = autoSurprises.slice(-100);
  const autoFirst100Avg = autoFirst100.reduce((s, x) => s + x, 0) / autoFirst100.length;
  const autoLast100Avg = autoLast100.reduce((s, x) => s + x, 0) / autoLast100.length;

  log(`  Seed avg surprise:      ${seedAvg.toFixed(6)}`);
  log(`  Auto first 100 avg:     ${autoFirst100Avg.toFixed(6)}`);
  log(`  Auto last 100 avg:      ${autoLast100Avg.toFixed(6)}`);
  log(`  Final surprise:         ${allSurprises[allSurprises.length - 1].toFixed(6)}`);

  // State change trajectory
  const changeFirst100 = selfChanges.slice(0, 100);
  const changeLast100 = selfChanges.slice(-100);
  const changeFirst100Avg = changeFirst100.reduce((s, x) => s + x, 0) / changeFirst100.length;
  const changeLast100Avg = changeLast100.reduce((s, x) => s + x, 0) / changeLast100.length;
  const changeFinal = selfChanges[selfChanges.length - 1];

  log(`  State change first 100: ${changeFirst100Avg.toFixed(8)}`);
  log(`  State change last 100:  ${changeLast100Avg.toFixed(8)}`);
  log(`  Final state change:     ${changeFinal.toFixed(8)}`);

  // Memory count
  const finalMem = allMemCounts[allMemCounts.length - 1];
  const maxMem = Math.max(...allMemCounts);
  log(`  Final memory count:     ${finalMem}`);
  log(`  Max memory count:       ${maxMem}`);

  // Check for oscillation: compute surprise variance in last 1000 ticks
  const last1000 = autoSurprises.slice(-1000);
  const last1000Avg = last1000.reduce((s, x) => s + x, 0) / last1000.length;
  const last1000Var = last1000.reduce((s, x) => s + (x - last1000Avg) ** 2, 0) / last1000.length;
  const last1000Std = Math.sqrt(last1000Var);
  log(`  Last 1000 surprise std: ${last1000Std.toFixed(8)}`);

  // Classify dynamics
  const FIXED_POINT_THRESHOLD = 1e-6;
  const OSCILLATION_THRESHOLD = 0.001;
  let classification;

  if (changeLast100Avg < FIXED_POINT_THRESHOLD && last1000Std < FIXED_POINT_THRESHOLD) {
    classification = 'FIXED POINT';
  } else if (last1000Std > FIXED_POINT_THRESHOLD && last1000Std < OSCILLATION_THRESHOLD && changeLast100Avg < OSCILLATION_THRESHOLD) {
    classification = 'STABLE OSCILLATION';
  } else if (finalMem > 100 || autoLast100Avg > seedAvg * 5) {
    classification = 'DIVERGENCE';
  } else {
    classification = 'STRUCTURED TRAJECTORY';
  }

  log(`\n  >>> DYNAMICS CLASSIFICATION: ${classification} <<<`);

  // State trajectory samples (first 8 dims every 100 ticks)
  log('\n--- State Trajectory (first 8 dims, every 100 ticks) ---');
  log(`  ${'Tick'.padStart(6)}  ${Array.from({length: 8}, (_, i) => `d${i}`.padStart(9)).join(' ')}`);
  for (const snap of stateSnapshots) {
    const dimStr = snap.dims.map(d => d.toFixed(5).padStart(9)).join(' ');
    log(`  ${String(snap.tick).padStart(6)}  ${dimStr}`);
  }

  // Full tick-by-tick surprise (sampled every 10 ticks for readability)
  log('\n--- Surprise Trajectory (every 10 ticks, full run) ---');
  for (let i = 0; i < allSurprises.length; i += 10) {
    const tick = i + 1;
    log(`  Tick ${String(tick).padStart(6)}: surprise=${allSurprises[i].toFixed(6)}, mem=${allMemCounts[i]}`);
  }

  const stats = engine.attractors.getStats();
  log(`\n  Final attractor stats: simMu=${stats.simMu.toFixed(6)}, simSigma=${stats.simSigma.toFixed(6)}, surpriseEma=${stats.surpriseEma.toFixed(6)}, noiseFloor=${stats.noiseFloor.toFixed(6)}`);
}

// ════════════════════════════════════════════════════════════════════════════
// EXPERIMENT 2: Autoregression + B Injection
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n\n' + '='.repeat(78));
  log('EXPERIMENT 2: Autoregression with B Injection');
  log('  Phase 1: Seed with A (100 ticks)');
  log('  Phase 2: Autoregress (ticks 101-2100, 2000 ticks)');
  log('  Phase 3: Inject B (ticks 2101-2300, 200 ticks)');
  log('  Phase 4: Autoregress again (ticks 2301-4300, 2000 ticks)');
  log('='.repeat(78));

  _rng = 777;  // different seed for experiment 2
  const engine = new ActiveInferenceEngine(DIM);
  const distA = randUnit(DIM);
  const distB = makeOrtho(distA, DIM);
  const NOISE = 0.001;

  const sampleA = () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = distA[i] + NOISE * randN();
    return normalize(v);
  };
  const sampleB = () => {
    const v = new Float64Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = distB[i] + NOISE * randN();
    return normalize(v);
  };

  const allSurprises = [];
  const allMemCounts = [];
  const phaseLabels = [];

  // Phase 1: Seed with A
  log('\n--- Phase 1: Seed with A (ticks 1-100) ---');
  for (let t = 1; t <= 100; t++) {
    const result = engine.fold(sampleA(), baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
    allSurprises.push(result.surprise);
    allMemCounts.push(result.memoryCount);
    phaseLabels.push('seed_A');

    if (t % 25 === 0) {
      log(`  Tick ${String(t).padStart(5)}: surprise=${result.surprise.toFixed(6)}, mem=${result.memoryCount}`);
    }
  }

  // Phase 2: Autoregress 2000 ticks
  log('\n--- Phase 2: Autoregression (ticks 101-2100) ---');
  let lastOutput = engine.act(velScale);
  const preB_surprises = [];

  for (let t = 101; t <= 2100; t++) {
    const result = engine.fold(lastOutput, baseLr, memLr, k, memW, velDecay, velGain);
    lastOutput = engine.act(velScale);
    allSurprises.push(result.surprise);
    allMemCounts.push(result.memoryCount);
    phaseLabels.push('auto_1');

    if (t >= 2050) preB_surprises.push(result.surprise);

    if ((t - 100) % 500 === 0) {
      log(`  Tick ${String(t).padStart(5)}: surprise=${result.surprise.toFixed(6)}, mem=${result.memoryCount}`);
    }
  }

  const preBAvg = preB_surprises.reduce((s, x) => s + x, 0) / preB_surprises.length;
  const memBeforeB = allMemCounts[allMemCounts.length - 1];
  const selfBeforeB = engine.getSelf();
  log(`  Pre-B surprise (last 50): ${preBAvg.toFixed(6)}`);
  log(`  Pre-B memories: ${memBeforeB}`);

  // Phase 3: Inject B
  log('\n--- Phase 3: Inject B (ticks 2101-2300) ---');
  const bInjection_surprises = [];
  let bFirstSurprise = null;

  for (let t = 2101; t <= 2300; t++) {
    const result = engine.fold(sampleB(), baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
    allSurprises.push(result.surprise);
    allMemCounts.push(result.memoryCount);
    phaseLabels.push('inject_B');
    bInjection_surprises.push(result.surprise);

    if (t === 2101) bFirstSurprise = result.surprise;

    if (t % 25 === 0 || t === 2101) {
      log(`  Tick ${String(t).padStart(5)}: surprise=${result.surprise.toFixed(6)}, mem=${result.memoryCount}, action=${result.attractorAction}`);
    }
  }

  const bAvg = bInjection_surprises.reduce((s, x) => s + x, 0) / bInjection_surprises.length;
  const bMax = Math.max(...bInjection_surprises);
  const memAfterB = allMemCounts[allMemCounts.length - 1];
  const selfAfterB = engine.getSelf();
  log(`  B injection avg surprise: ${bAvg.toFixed(6)}`);
  log(`  B injection max surprise: ${bMax.toFixed(6)}`);
  log(`  B first tick surprise:    ${bFirstSurprise.toFixed(6)}`);
  log(`  Memories after B:         ${memAfterB}`);

  // Self similarity before/after B
  const selfSimBeforeAfterB = dot(selfBeforeB, selfAfterB);
  log(`  Self similarity pre/post B: ${selfSimBeforeAfterB.toFixed(6)}`);

  // Phase 4: Autoregress again
  log('\n--- Phase 4: Autoregression post-B (ticks 2301-4300) ---');
  lastOutput = engine.act(velScale);
  const postB_surprises = [];

  for (let t = 2301; t <= 4300; t++) {
    const result = engine.fold(lastOutput, baseLr, memLr, k, memW, velDecay, velGain);
    lastOutput = engine.act(velScale);
    allSurprises.push(result.surprise);
    allMemCounts.push(result.memoryCount);
    phaseLabels.push('auto_2');

    if (t >= 4250) postB_surprises.push(result.surprise);

    if ((t - 2300) % 500 === 0) {
      log(`  Tick ${String(t).padStart(5)}: surprise=${result.surprise.toFixed(6)}, mem=${result.memoryCount}`);
    }
  }

  const postBAvg = postB_surprises.reduce((s, x) => s + x, 0) / postB_surprises.length;
  const memFinal = allMemCounts[allMemCounts.length - 1];
  const selfFinal = engine.getSelf();

  // Similarity of final state to pre-B state and to B distribution
  const simFinalToPreB = dot(selfFinal, selfBeforeB);
  const simFinalToA = dot(selfFinal, distA);
  const simFinalToB = dot(selfFinal, distB);

  log(`\n  Post-B surprise (last 50): ${postBAvg.toFixed(6)}`);
  log(`  Final memories: ${memFinal}`);

  // Analysis
  log('\n--- Experiment 2 Analysis ---');
  log(`  Pre-B avg surprise (last 50):  ${preBAvg.toFixed(6)}`);
  log(`  B injection first surprise:    ${bFirstSurprise.toFixed(6)}`);
  log(`  B injection avg surprise:      ${bAvg.toFixed(6)}`);
  log(`  B injection max surprise:      ${bMax.toFixed(6)}`);
  log(`  Post-B avg surprise (last 50): ${postBAvg.toFixed(6)}`);
  log('');
  log(`  Memories before B:  ${memBeforeB}`);
  log(`  Memories after B:   ${memAfterB}`);
  log(`  New memories from B: ${memAfterB - memBeforeB}`);
  log(`  Final memories:     ${memFinal}`);
  log('');
  log(`  Self similarity (pre-B to post-B self): ${selfSimBeforeAfterB.toFixed(6)}`);
  log(`  Self similarity (final to pre-B):       ${simFinalToPreB.toFixed(6)}`);
  log(`  Self similarity (final to dist A):      ${simFinalToA.toFixed(6)}`);
  log(`  Self similarity (final to dist B):      ${simFinalToB.toFixed(6)}`);

  const spiked = bFirstSurprise > preBAvg * 2;
  const trajectoryChanged = simFinalToPreB < 0.99;
  const newMems = memAfterB - memBeforeB;

  log('');
  log(`  >>> Surprise spike on B injection: ${spiked ? 'YES' : 'NO'} (${bFirstSurprise.toFixed(6)} vs pre-B ${preBAvg.toFixed(6)}, ratio=${(bFirstSurprise / (preBAvg + 1e-12)).toFixed(2)}x) <<<`);
  log(`  >>> Trajectory changed after B:    ${trajectoryChanged ? 'YES' : 'NO'} (final-to-preB sim=${simFinalToPreB.toFixed(6)}) <<<`);
  log(`  >>> New memories from B:           ${newMems} <<<`);

  // Full surprise trace every 10 ticks
  log('\n--- Full Surprise Trace (every 10 ticks) ---');
  for (let i = 0; i < allSurprises.length; i += 10) {
    const tick = i + 1;
    log(`  Tick ${String(tick).padStart(6)} [${phaseLabels[i].padEnd(8)}]: surprise=${allSurprises[i].toFixed(6)}, mem=${allMemCounts[i]}`);
  }

  const stats = engine.attractors.getStats();
  log(`\n  Final attractor stats: simMu=${stats.simMu.toFixed(6)}, simSigma=${stats.simSigma.toFixed(6)}, surpriseEma=${stats.surpriseEma.toFixed(6)}, noiseFloor=${stats.noiseFloor.toFixed(6)}`);
}

// Save results
log('\n' + '='.repeat(78));
log('END OF STEP 17 AUTOREGRESSION EXPERIMENTS');
log('='.repeat(78));

writeFileSync('B:/M/avir/research/fluxcore/results/autoregression_basic.txt', allLines.join('\n') + '\n');
log('\nCaptured to results/autoregression_basic.txt');
