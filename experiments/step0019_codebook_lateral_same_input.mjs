/**
 * Step 19: Lateral Fold Composition — Same Input
 *
 * Two CanonicalFold instances (Fold-A, Fold-B) at DIM=64, identical init.
 * Both receive: effective_reality = normalize(0.7 * external + 0.3 * peerOutput)
 * Run 4-dist test: A->B->C->D->A (500 ticks each), 2500 ticks total.
 *
 * Also run single-fold baseline for comparison.
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
// ACTIVE INFERENCE ENGINE — canonical fold
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

  getSelf() { return clone(this.self); }
}

// ════════════════════════════════════════════════════════════════════════════
// ENVIRONMENT: 4 distributions, switch every 500 ticks
// ════════════════════════════════════════════════════════════════════════════
class DynamicEnvironment {
  constructor(dim) {
    this.dim = dim;
    this.currentDist = 0;
    this.distributions = [];
    this.tick = 0;
    let base = randUnit(dim);
    for (let i = 0; i < 4; i++) {
      this.distributions.push(clone(base));
      base = makeOrtho(base, dim);
    }
  }
  sense() {
    if (this.tick % 500 === 0 && this.tick > 0) {
      this.currentDist = (this.currentDist + 1) % this.distributions.length;
    }
    this.tick++;
    const dist = this.distributions[this.currentDist];
    const noise = 0.001;
    const v = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) v[i] = dist[i] + noise * randN();
    return normalize(v);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// OUTPUT
// ════════════════════════════════════════════════════════════════════════════
const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

const baseLr = 0.08, memLr = 0.015, k = 20, memW = 0.15, velDecay = 0.95, velGain = 0.05, velScale = 12.5;
const DIM = 64;
const TOTAL_TICKS = 2500;
const PEER_WEIGHT = 0.3;
const EXT_WEIGHT = 0.7;

log('='.repeat(78));
log('Step 19: Lateral Fold Composition — Same Input');
log('='.repeat(78));
log(`DIM=${DIM}, baseLr=${baseLr}, memLr=${memLr}, k=${k}, memW=${memW}`);
log(`velDecay=${velDecay}, velGain=${velGain}, velScale=${velScale}`);
log(`Lateral mix: ${EXT_WEIGHT} * external + ${PEER_WEIGHT} * peerOutput`);
log(`4 distributions, switch every 500 ticks, total ${TOTAL_TICKS} ticks`);
log('');

// Helper to blend reality with peer output
function blendReality(external, peerOutput, dim) {
  const v = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    v[i] = EXT_WEIGHT * external[i] + PEER_WEIGHT * peerOutput[i];
  }
  return normalize(v);
}

// ════════════════════════════════════════════════════════════════════════════
// SINGLE-FOLD BASELINE (no lateral coupling)
// ════════════════════════════════════════════════════════════════════════════
let baselineSurprises, baselineMemCount, baselineMems;
{
  log('-'.repeat(78));
  log('BASELINE: Single Fold (no lateral coupling)');
  log('-'.repeat(78));

  _rng = 42;
  const env = new DynamicEnvironment(DIM);
  const engine = new ActiveInferenceEngine(DIM);

  const surprises = [];
  const memCounts = [];

  for (let t = 0; t < TOTAL_TICKS; t++) {
    const reality = env.sense();
    const result = engine.fold(reality, baseLr, memLr, k, memW, velDecay, velGain);
    engine.act(velScale);
    surprises.push(result.surprise);
    memCounts.push(result.memoryCount);

    if ((t + 1) % 100 === 0) {
      const avg100 = surprises.slice(-100).reduce((s, x) => s + x, 0) / 100;
      const distIdx = Math.floor(t / 500);
      const distLabel = ['A', 'B', 'C', 'D', 'A'][distIdx];
      log(`  Tick ${String(t + 1).padStart(5)} [dist ${distLabel}]: surprise=${result.surprise.toFixed(6)}, avg100=${avg100.toFixed(6)}, mem=${result.memoryCount}`);
    }
  }

  const stats = engine.attractors.getStats();
  const avgSurprise = surprises.reduce((s, x) => s + x, 0) / surprises.length;
  const last500Avg = surprises.slice(-500).reduce((s, x) => s + x, 0) / 500;

  log(`\n  Baseline overall avg surprise: ${avgSurprise.toFixed(6)}`);
  log(`  Baseline last 500 avg:         ${last500Avg.toFixed(6)}`);
  log(`  Baseline final memories:        ${stats.count}`);
  log(`  Baseline simMu=${stats.simMu.toFixed(6)}, simSigma=${stats.simSigma.toFixed(6)}`);

  baselineSurprises = surprises;
  baselineMemCount = stats.count;
  baselineMems = engine.attractors.memories.map(m => clone(m));
}

// ════════════════════════════════════════════════════════════════════════════
// LATERAL COMPOSITION: Two folds, same input, peer-coupled
// ════════════════════════════════════════════════════════════════════════════
{
  log('\n' + '-'.repeat(78));
  log('LATERAL: Two Folds (Fold-A, Fold-B), peer-coupled');
  log(`  effective_reality = normalize(${EXT_WEIGHT} * external + ${PEER_WEIGHT} * peerOutput)`);
  log('-'.repeat(78));

  // Reset RNG to same seed so environment generates identical distributions
  _rng = 42;
  const env = new DynamicEnvironment(DIM);

  // Both folds initialized with same RNG state after environment creation
  const foldA = new ActiveInferenceEngine(DIM);
  const foldB = new ActiveInferenceEngine(DIM);

  // Initial peer outputs (before any ticks) — just use their initial self vectors
  let outputA = foldA.act(velScale);
  let outputB = foldB.act(velScale);

  const surprisesA = [], surprisesB = [];
  const memCountsA = [], memCountsB = [];
  const selfSimilarities = [];  // dot(foldA.self, foldB.self) each tick

  log('\n--- Tick-by-tick (every 100 ticks) ---');
  log(`  ${'Tick'.padStart(5)} ${'Dist'.padStart(4)} | ${'surpA'.padStart(10)} ${'surpB'.padStart(10)} ${'avgA100'.padStart(10)} ${'avgB100'.padStart(10)} | ${'memA'.padStart(4)} ${'memB'.padStart(4)} | ${'selfSim'.padStart(10)}`);
  log('  ' + '-'.repeat(85));

  for (let t = 0; t < TOTAL_TICKS; t++) {
    const external = env.sense();

    // Blend reality with peer output from previous tick
    const realityA = blendReality(external, outputB, DIM);
    const realityB = blendReality(external, outputA, DIM);

    // Run both folds
    const resultA = foldA.fold(realityA, baseLr, memLr, k, memW, velDecay, velGain);
    const resultB = foldB.fold(realityB, baseLr, memLr, k, memW, velDecay, velGain);

    // Generate outputs for next tick's peer coupling
    outputA = foldA.act(velScale);
    outputB = foldB.act(velScale);

    surprisesA.push(resultA.surprise);
    surprisesB.push(resultB.surprise);
    memCountsA.push(resultA.memoryCount);
    memCountsB.push(resultB.memoryCount);

    const selfSim = dot(foldA.getSelf(), foldB.getSelf());
    selfSimilarities.push(selfSim);

    if ((t + 1) % 100 === 0) {
      const avg100A = surprisesA.slice(-100).reduce((s, x) => s + x, 0) / 100;
      const avg100B = surprisesB.slice(-100).reduce((s, x) => s + x, 0) / 100;
      const distIdx = Math.floor(t / 500);
      const distLabel = ['A', 'B', 'C', 'D', 'A'][distIdx];
      log(`  ${String(t + 1).padStart(5)} ${distLabel.padStart(4)} | ${resultA.surprise.toFixed(6).padStart(10)} ${resultB.surprise.toFixed(6).padStart(10)} ${avg100A.toFixed(6).padStart(10)} ${avg100B.toFixed(6).padStart(10)} | ${String(resultA.memoryCount).padStart(4)} ${String(resultB.memoryCount).padStart(4)} | ${selfSim.toFixed(6).padStart(10)}`);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // ANALYSIS
  // ═══════════════════════════════════════════════════════════════════════
  log('\n' + '='.repeat(78));
  log('ANALYSIS');
  log('='.repeat(78));

  // 1. Self similarity trajectory
  log('\n--- Self Similarity (Fold-A.self vs Fold-B.self) ---');
  const simFirst100 = selfSimilarities.slice(0, 100).reduce((s, x) => s + x, 0) / 100;
  const simLast100 = selfSimilarities.slice(-100).reduce((s, x) => s + x, 0) / 100;
  const simFinal = selfSimilarities[selfSimilarities.length - 1];
  const simMin = Math.min(...selfSimilarities);
  const simMax = Math.max(...selfSimilarities);
  log(`  First 100 avg:  ${simFirst100.toFixed(6)}`);
  log(`  Last 100 avg:   ${simLast100.toFixed(6)}`);
  log(`  Final:          ${simFinal.toFixed(6)}`);
  log(`  Min:            ${simMin.toFixed(6)}`);
  log(`  Max:            ${simMax.toFixed(6)}`);

  // Self similarity at distribution boundaries
  log('\n  Self similarity at distribution switches:');
  for (const t of [499, 500, 501, 999, 1000, 1001, 1499, 1500, 1501, 1999, 2000, 2001]) {
    if (t < selfSimilarities.length) {
      log(`    Tick ${String(t + 1).padStart(5)}: ${selfSimilarities[t].toFixed(6)}`);
    }
  }

  // 2. Surprise comparison
  log('\n--- Surprise Comparison ---');
  const avgSurpA = surprisesA.reduce((s, x) => s + x, 0) / surprisesA.length;
  const avgSurpB = surprisesB.reduce((s, x) => s + x, 0) / surprisesB.length;
  const avgSurpLateral = (avgSurpA + avgSurpB) / 2;
  const avgSurpBaseline = baselineSurprises.reduce((s, x) => s + x, 0) / baselineSurprises.length;
  const last500A = surprisesA.slice(-500).reduce((s, x) => s + x, 0) / 500;
  const last500B = surprisesB.slice(-500).reduce((s, x) => s + x, 0) / 500;
  const last500Baseline = baselineSurprises.slice(-500).reduce((s, x) => s + x, 0) / 500;

  log(`  Fold-A overall avg:     ${avgSurpA.toFixed(6)}`);
  log(`  Fold-B overall avg:     ${avgSurpB.toFixed(6)}`);
  log(`  Lateral combined avg:   ${avgSurpLateral.toFixed(6)}`);
  log(`  Baseline (single) avg:  ${avgSurpBaseline.toFixed(6)}`);
  log(`  Lateral improvement:    ${((1 - avgSurpLateral / avgSurpBaseline) * 100).toFixed(2)}%`);
  log('');
  log(`  Fold-A last 500 avg:    ${last500A.toFixed(6)}`);
  log(`  Fold-B last 500 avg:    ${last500B.toFixed(6)}`);
  log(`  Lateral last 500 avg:   ${((last500A + last500B) / 2).toFixed(6)}`);
  log(`  Baseline last 500 avg:  ${last500Baseline.toFixed(6)}`);

  // Per-distribution surprise comparison
  log('\n  Per-distribution surprise (avg over 500 ticks each):');
  const distLabels = ['A', 'B', 'C', 'D', 'A'];
  for (let d = 0; d < 5; d++) {
    const start = d * 500, end = (d + 1) * 500;
    const sliceA = surprisesA.slice(start, end);
    const sliceB = surprisesB.slice(start, end);
    const sliceBase = baselineSurprises.slice(start, end);
    const dAvgA = sliceA.reduce((s, x) => s + x, 0) / sliceA.length;
    const dAvgB = sliceB.reduce((s, x) => s + x, 0) / sliceB.length;
    const dAvgBase = sliceBase.reduce((s, x) => s + x, 0) / sliceBase.length;
    log(`    Dist ${distLabels[d]} (ticks ${start + 1}-${end}): lateral=(${dAvgA.toFixed(6)}, ${dAvgB.toFixed(6)}) avg=${((dAvgA + dAvgB) / 2).toFixed(6)}, baseline=${dAvgBase.toFixed(6)}`);
  }

  // 3. Memory counts
  log('\n--- Memory Counts ---');
  const statsA = foldA.attractors.getStats();
  const statsB = foldB.attractors.getStats();
  log(`  Fold-A final memories: ${statsA.count}`);
  log(`  Fold-B final memories: ${statsB.count}`);
  log(`  Baseline memories:     ${baselineMemCount}`);
  log(`  Combined lateral:      ${statsA.count + statsB.count}`);

  // 4. Memory pool cross-comparison
  log('\n--- Memory Pool Cross-Comparison ---');
  log(`  Dot products: Fold-A memories (rows) vs Fold-B memories (cols)`);
  const memsA = foldA.attractors.memories;
  const memsB = foldB.attractors.memories;

  // Header
  let header = '        ';
  for (let j = 0; j < memsB.length; j++) header += `  B-${j}    `;
  log(`  ${header}`);

  const crossDots = [];
  for (let i = 0; i < memsA.length; i++) {
    let row = `  A-${i}  `;
    for (let j = 0; j < memsB.length; j++) {
      const d = dot(memsA[i], memsB[j]);
      crossDots.push(d);
      row += `  ${d.toFixed(4)} `;
    }
    log(row);
  }

  // Cross-dot statistics
  const absCrossDots = crossDots.map(d => Math.abs(d));
  const maxCross = Math.max(...absCrossDots);
  const avgCross = absCrossDots.reduce((s, x) => s + x, 0) / absCrossDots.length;
  const highOverlap = absCrossDots.filter(d => d > 0.9).length;
  const medOverlap = absCrossDots.filter(d => d > 0.5).length;

  log(`\n  Cross-dot |abs| stats:`);
  log(`    Max:                ${maxCross.toFixed(6)}`);
  log(`    Avg:                ${avgCross.toFixed(6)}`);
  log(`    Pairs with |d|>0.9: ${highOverlap} / ${crossDots.length}`);
  log(`    Pairs with |d|>0.5: ${medOverlap} / ${crossDots.length}`);

  // 5. Intra-fold memory similarity
  log('\n--- Intra-Fold Memory Similarity ---');
  const intraDots = (mems, label) => {
    const dots = [];
    for (let i = 0; i < mems.length; i++) {
      for (let j = i + 1; j < mems.length; j++) {
        const d = Math.abs(dot(mems[i], mems[j]));
        dots.push(d);
      }
    }
    if (dots.length === 0) {
      log(`  ${label}: only 1 memory, no intra-pairs`);
      return;
    }
    const max = Math.max(...dots);
    const avg = dots.reduce((s, x) => s + x, 0) / dots.length;
    log(`  ${label}: ${dots.length} pairs, max=${max.toFixed(6)}, avg=${avg.toFixed(6)}`);
  };
  intraDots(memsA, 'Fold-A');
  intraDots(memsB, 'Fold-B');
  intraDots(baselineMems, 'Baseline');

  // 6. Memory alignment to environment distributions
  log('\n--- Memory Alignment to Environment Distributions ---');
  // Reconstruct environment distributions (same RNG seed)
  _rng = 42;
  const envCheck = new DynamicEnvironment(DIM);
  const envDists = envCheck.distributions.map(d => clone(d));

  const alignReport = (mems, label) => {
    log(`\n  ${label}:`);
    for (let m = 0; m < mems.length; m++) {
      const sims = envDists.map((d, i) => ({ dist: i, sim: dot(mems[m], d) })).sort((a, b) => Math.abs(b.sim) - Math.abs(a.sim));
      const best = sims[0];
      log(`    Mem ${m}: best=dist${best.dist} (sim=${best.sim.toFixed(6)}), others=[${sims.slice(1).map(s => `d${s.dist}:${s.sim.toFixed(4)}`).join(', ')}]`);
    }
  };
  alignReport(memsA, 'Fold-A');
  alignReport(memsB, 'Fold-B');
  alignReport(baselineMems, 'Baseline');

  // ═══════════════════════════════════════════════════════════════════════
  // VERDICT
  // ═══════════════════════════════════════════════════════════════════════
  log('\n' + '='.repeat(78));
  log('VERDICT: SPECIALIZATION vs REDUNDANCY');
  log('='.repeat(78));

  // Criteria:
  // - Redundant if most cross-dot |abs| > 0.9 (memories are copies)
  // - Specialized if cross-dots are low (memories are complementary)
  // - Mixed if some high, some low

  let verdict;
  if (highOverlap > crossDots.length * 0.7) {
    verdict = 'REDUNDANT';
  } else if (maxCross < 0.5) {
    verdict = 'SPECIALIZED';
  } else if (highOverlap > 0 && highOverlap <= crossDots.length * 0.5) {
    verdict = 'PARTIAL SPECIALIZATION';
  } else {
    verdict = 'MIXED';
  }

  const selfConverged = simFinal > 0.99;
  const surpriseImproved = avgSurpLateral < avgSurpBaseline;

  log(`\n  Self vectors converged:  ${selfConverged ? 'YES' : 'NO'} (final sim=${simFinal.toFixed(6)})`);
  log(`  Memory verdict:          ${verdict}`);
  log(`    High overlap (|d|>0.9): ${highOverlap}/${crossDots.length} pairs`);
  log(`    Med overlap (|d|>0.5):  ${medOverlap}/${crossDots.length} pairs`);
  log(`    Max cross-dot:          ${maxCross.toFixed(6)}`);
  log(`    Avg cross-dot:          ${avgCross.toFixed(6)}`);
  log(`  Surprise improved:       ${surpriseImproved ? 'YES' : 'NO'} (lateral=${avgSurpLateral.toFixed(6)} vs baseline=${avgSurpBaseline.toFixed(6)})`);
  log(`  Memory A count: ${statsA.count}, B count: ${statsB.count}, baseline: ${baselineMemCount}`);
  log('');
  log(`  >>> VERDICT: ${verdict} <<<`);
  log(`  >>> Self convergence: ${selfConverged ? 'CONVERGED (identical selves)' : 'DIVERGENT (distinct selves)'} <<<`);
  log(`  >>> Tracking quality: lateral avg=${avgSurpLateral.toFixed(6)} vs baseline=${avgSurpBaseline.toFixed(6)} (${surpriseImproved ? 'IMPROVED' : 'DEGRADED'} by ${Math.abs((1 - avgSurpLateral / avgSurpBaseline) * 100).toFixed(2)}%) <<<`);
}

log('\n' + '='.repeat(78));
log('END OF STEP 19 LATERAL COMPOSITION');
log('='.repeat(78));

writeFileSync('B:/M/avir/research/fluxcore/results/lateral_same_input.txt', allLines.join('\n') + '\n');
log('\nCaptured to results/lateral_same_input.txt');
