/**
 * Step 20: Lateral Fold Composition with DIFFERENT Input Streams
 *
 * Fold-A receives div01-div10 CSI embeddings
 * Fold-B receives div11-div20 CSI embeddings (next 10 divisions)
 * Each tick: cross-blend 0.2 of the other fold's previous output
 *   effective_reality_A = normalize(0.8 * streamA[t] + 0.2 * foldB.output_prev)
 *   effective_reality_B = normalize(0.8 * streamB[t] + 0.2 * foldA.output_prev)
 *
 * Key question: does information transfer through lateral composition?
 */

import { writeFileSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

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

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC ATTRACTOR FIELD (from canonical)
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
// ACTIVE INFERENCE ENGINE (canonical fold)
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

  getOutput() {
    // Return current self state as the fold's output for lateral blending
    return clone(this.self);
  }
}

// ════════════════════════════════════════════════════════════════════════════
// LOAD DATA & SPLIT STREAMS
// ════════════════════════════════════════════════════════════════════════════
const allLines = [];
const log = s => { allLines.push(s); process.stdout.write(s + '\n'); };

log('='.repeat(78));
log('Step 20: Lateral Fold Composition — DIFFERENT Input Streams');
log('='.repeat(78));

const embedded = JSON.parse(readFileSync(path.join(ROOT, 'data', 'csi_embedded.json'), 'utf8'));
const centers = JSON.parse(readFileSync(path.join(ROOT, 'data', 'csi_division_centers.json'), 'utf8'));
const allDivs = Object.keys(centers).sort();

log(`\nLoaded ${embedded.length} records, ${allDivs.length} divisions, dim=384`);

// Split: first ~10 divisions -> stream A, next ~10 -> stream B
// Divisions: 00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,21,22,23,25,26,27,28,31,32,33,34,35,40,41,43,44,46,48
// Group A: first 10 (00-09), Group B: next 10 (10-14,21-25)
const groupADivs = allDivs.slice(0, 10);  // ~first 10 divisions
const groupBDivs = allDivs.slice(10, 20); // ~next 10 divisions

log(`\nGroup A divisions (${groupADivs.length}): ${groupADivs.join(', ')}`);
log(`Group B divisions (${groupBDivs.length}): ${groupBDivs.join(', ')}`);

const groupASet = new Set(groupADivs);
const groupBSet = new Set(groupBDivs);

const streamA = embedded.filter(r => groupASet.has(r.division));
const streamB = embedded.filter(r => groupBSet.has(r.division));

log(`Stream A: ${streamA.length} records`);
log(`Stream B: ${streamB.length} records`);

const DIM = 384;
const TICKS = 5000;
const BLEND_WEIGHT = 0.2;
const SELF_WEIGHT = 0.8;

// Fold parameters
const baseLr = 0.08, memLr = 0.015, k = 20, memW = 0.15, velDecay = 0.95, velGain = 0.05, velScale = 12.5;

// ════════════════════════════════════════════════════════════════════════════
// Helper: measure memory alignment against division centers
// ════════════════════════════════════════════════════════════════════════════
function measureAlignment(engine, divList, label) {
  const mems = engine.attractors.memories;
  const results = [];
  for (let m = 0; m < mems.length; m++) {
    const mem = mems[m];
    const sims = divList.map(div => ({
      div,
      sim: dot(mem, new Float64Array(centers[div]))
    })).sort((a, b) => b.sim - a.sim);
    results.push({
      mem: m,
      best: sims[0],
      second: sims.length > 1 ? sims[1] : { div: '-', sim: 0 },
      specificity: sims.length > 1 ? sims[0].sim - sims[1].sim : sims[0].sim
    });
  }
  return results;
}

function meanAlignment(alignments) {
  if (!alignments.length) return { meanBestSim: 0, passCount: 0, total: 0 };
  const bestSims = alignments.map(a => a.best.sim);
  return {
    meanBestSim: avg(bestSims),
    passCount: bestSims.filter(s => s > 0.3).length,
    total: alignments.length
  };
}

// ════════════════════════════════════════════════════════════════════════════
// PHASE 1: ISOLATED BASELINES (no lateral blending)
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('PHASE 1: ISOLATED BASELINES (no lateral blending)');
log('='.repeat(78));

// Isolated Fold-A
_rng = 42;
const isoA = new ActiveInferenceEngine(DIM);
for (let t = 0; t < TICKS; t++) {
  const rec = streamA[t % streamA.length];
  const vec = new Float64Array(rec.vec);
  isoA.fold(vec, baseLr, memLr, k, memW, velDecay, velGain);
  isoA.act(velScale);
}

const isoAOwnAlign = measureAlignment(isoA, groupADivs, 'isoA-own');
const isoACrossAlign = measureAlignment(isoA, groupBDivs, 'isoA-cross');
const isoAOwn = meanAlignment(isoAOwnAlign);
const isoACross = meanAlignment(isoACrossAlign);

log(`\nIsolated Fold-A: ${isoA.attractors.memories.length} memories`);
log(`  Own-stream alignment (divA):   mean=${isoAOwn.meanBestSim.toFixed(4)}, pass=${isoAOwn.passCount}/${isoAOwn.total}`);
log(`  Cross-stream alignment (divB): mean=${isoACross.meanBestSim.toFixed(4)}, pass=${isoACross.passCount}/${isoACross.total}`);

// Isolated Fold-B
_rng = 137;
const isoB = new ActiveInferenceEngine(DIM);
for (let t = 0; t < TICKS; t++) {
  const rec = streamB[t % streamB.length];
  const vec = new Float64Array(rec.vec);
  isoB.fold(vec, baseLr, memLr, k, memW, velDecay, velGain);
  isoB.act(velScale);
}

const isoBOwnAlign = measureAlignment(isoB, groupBDivs, 'isoB-own');
const isoBCrossAlign = measureAlignment(isoB, groupADivs, 'isoB-cross');
const isoBOwn = meanAlignment(isoBOwnAlign);
const isoBCross = meanAlignment(isoBCrossAlign);

log(`\nIsolated Fold-B: ${isoB.attractors.memories.length} memories`);
log(`  Own-stream alignment (divB):   mean=${isoBOwn.meanBestSim.toFixed(4)}, pass=${isoBOwn.passCount}/${isoBOwn.total}`);
log(`  Cross-stream alignment (divA): mean=${isoBCross.meanBestSim.toFixed(4)}, pass=${isoBCross.passCount}/${isoBCross.total}`);

// ════════════════════════════════════════════════════════════════════════════
// PHASE 2: LATERAL COMPOSITION (0.8 own + 0.2 other's previous output)
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('PHASE 2: LATERAL COMPOSITION (0.8 own + 0.2 other prev output)');
log('='.repeat(78));

_rng = 42;
const foldA = new ActiveInferenceEngine(DIM);
_rng = 137;
const foldB = new ActiveInferenceEngine(DIM);

// Previous outputs (initialized to random — will get overwritten quickly)
let prevOutputA = clone(foldA.self);
let prevOutputB = clone(foldB.self);

const surpriseA = [], surpriseB = [];
let spawnCountA = 0, spawnCountB = 0;

for (let t = 0; t < TICKS; t++) {
  const recA = streamA[t % streamA.length];
  const recB = streamB[t % streamB.length];
  const rawA = new Float64Array(recA.vec);
  const rawB = new Float64Array(recB.vec);

  // Blend: effective_reality = normalize(0.8 * own_stream + 0.2 * other_prev_output)
  const effectiveA = new Float64Array(DIM);
  const effectiveB = new Float64Array(DIM);
  for (let i = 0; i < DIM; i++) {
    effectiveA[i] = SELF_WEIGHT * rawA[i] + BLEND_WEIGHT * prevOutputB[i];
    effectiveB[i] = SELF_WEIGHT * rawB[i] + BLEND_WEIGHT * prevOutputA[i];
  }
  normalize(effectiveA);
  normalize(effectiveB);

  // Fold each
  const resA = foldA.fold(effectiveA, baseLr, memLr, k, memW, velDecay, velGain);
  const resB = foldB.fold(effectiveB, baseLr, memLr, k, memW, velDecay, velGain);

  surpriseA.push(resA.surprise);
  surpriseB.push(resB.surprise);
  if (resA.attractorAction === 'spawn') spawnCountA++;
  if (resB.attractorAction === 'spawn') spawnCountB++;

  // Act and capture outputs for next tick's lateral blend
  foldA.act(velScale);
  foldB.act(velScale);
  prevOutputA = foldA.getOutput();
  prevOutputB = foldB.getOutput();

  if ((t + 1) % 1000 === 0) {
    const avgSA = avg(surpriseA.slice(-200));
    const avgSB = avg(surpriseB.slice(-200));
    log(`  Tick ${String(t + 1).padStart(5)}: A(mem=${resA.memoryCount}, surp=${avgSA.toFixed(4)}) B(mem=${resB.memoryCount}, surp=${avgSB.toFixed(4)})`);
  }
}

log(`\nLateral Fold-A: ${foldA.attractors.memories.length} memories (${spawnCountA} spawns)`);
log(`Lateral Fold-B: ${foldB.attractors.memories.length} memories (${spawnCountB} spawns)`);

// ════════════════════════════════════════════════════════════════════════════
// PHASE 3: ALIGNMENT ANALYSIS
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('PHASE 3: ALIGNMENT ANALYSIS');
log('='.repeat(78));

// Lateral Fold-A alignment
const latAOwnAlign = measureAlignment(foldA, groupADivs, 'latA-own');
const latACrossAlign = measureAlignment(foldA, groupBDivs, 'latA-cross');
const latAOwn = meanAlignment(latAOwnAlign);
const latACross = meanAlignment(latACrossAlign);

log(`\nLateral Fold-A: ${foldA.attractors.memories.length} memories`);
log(`  Own-stream alignment (divA):   mean=${latAOwn.meanBestSim.toFixed(4)}, pass=${latAOwn.passCount}/${latAOwn.total}`);
log(`  Cross-stream alignment (divB): mean=${latACross.meanBestSim.toFixed(4)}, pass=${latACross.passCount}/${latACross.total}`);

// Detail per memory
log(`\n  ${'Mem'.padEnd(4)} ${'BestA'.padEnd(10)} ${'SimA'.padEnd(8)} ${'BestB'.padEnd(10)} ${'SimB'.padEnd(8)} Dominant`);
log('  ' + '-'.repeat(55));
for (let m = 0; m < foldA.attractors.memories.length; m++) {
  const ownA = latAOwnAlign[m];
  const crossA = latACrossAlign[m];
  const dominant = ownA.best.sim >= crossA.best.sim ? 'A-own' : 'B-cross';
  log(`  ${String(m).padEnd(4)} div${ownA.best.div.padEnd(6)} ${ownA.best.sim.toFixed(4).padEnd(8)} div${crossA.best.div.padEnd(6)} ${crossA.best.sim.toFixed(4).padEnd(8)} ${dominant}`);
}

// Lateral Fold-B alignment
const latBOwnAlign = measureAlignment(foldB, groupBDivs, 'latB-own');
const latBCrossAlign = measureAlignment(foldB, groupADivs, 'latB-cross');
const latBOwn = meanAlignment(latBOwnAlign);
const latBCross = meanAlignment(latBCrossAlign);

log(`\nLateral Fold-B: ${foldB.attractors.memories.length} memories`);
log(`  Own-stream alignment (divB):   mean=${latBOwn.meanBestSim.toFixed(4)}, pass=${latBOwn.passCount}/${latBOwn.total}`);
log(`  Cross-stream alignment (divA): mean=${latBCross.meanBestSim.toFixed(4)}, pass=${latBCross.passCount}/${latBCross.total}`);

log(`\n  ${'Mem'.padEnd(4)} ${'BestB'.padEnd(10)} ${'SimB'.padEnd(8)} ${'BestA'.padEnd(10)} ${'SimA'.padEnd(8)} Dominant`);
log('  ' + '-'.repeat(55));
for (let m = 0; m < foldB.attractors.memories.length; m++) {
  const ownB = latBOwnAlign[m];
  const crossB = latBCrossAlign[m];
  const dominant = ownB.best.sim >= crossB.best.sim ? 'B-own' : 'A-cross';
  log(`  ${String(m).padEnd(4)} div${ownB.best.div.padEnd(6)} ${ownB.best.sim.toFixed(4).padEnd(8)} div${crossB.best.div.padEnd(6)} ${crossB.best.sim.toFixed(4).padEnd(8)} ${dominant}`);
}

// ════════════════════════════════════════════════════════════════════════════
// PHASE 4: COMPARISON TABLE
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('PHASE 4: COMPARISON TABLE — Isolated vs Lateral');
log('='.repeat(78));

const pad = (s, w) => String(s).padEnd(w);
const padr = (s, w) => String(s).padStart(w);

log(`\n${''.padEnd(25)} ${'Own-stream'.padEnd(20)} ${'Cross-stream'.padEnd(20)} ${'Memories'.padEnd(10)}`);
log(`${''.padEnd(25)} ${'mean sim (pass)'.padEnd(20)} ${'mean sim (pass)'.padEnd(20)}`);
log('-'.repeat(78));
log(`${pad('Isolated Fold-A', 25)} ${pad(`${isoAOwn.meanBestSim.toFixed(4)} (${isoAOwn.passCount}/${isoAOwn.total})`, 20)} ${pad(`${isoACross.meanBestSim.toFixed(4)} (${isoACross.passCount}/${isoACross.total})`, 20)} ${isoA.attractors.memories.length}`);
log(`${pad('Lateral  Fold-A', 25)} ${pad(`${latAOwn.meanBestSim.toFixed(4)} (${latAOwn.passCount}/${latAOwn.total})`, 20)} ${pad(`${latACross.meanBestSim.toFixed(4)} (${latACross.passCount}/${latACross.total})`, 20)} ${foldA.attractors.memories.length}`);
log(`${pad('  delta A', 25)} ${pad(`${(latAOwn.meanBestSim - isoAOwn.meanBestSim) >= 0 ? '+' : ''}${(latAOwn.meanBestSim - isoAOwn.meanBestSim).toFixed(4)}`, 20)} ${pad(`${(latACross.meanBestSim - isoACross.meanBestSim) >= 0 ? '+' : ''}${(latACross.meanBestSim - isoACross.meanBestSim).toFixed(4)}`, 20)}`);
log('-'.repeat(78));
log(`${pad('Isolated Fold-B', 25)} ${pad(`${isoBOwn.meanBestSim.toFixed(4)} (${isoBOwn.passCount}/${isoBOwn.total})`, 20)} ${pad(`${isoBCross.meanBestSim.toFixed(4)} (${isoBCross.passCount}/${isoBCross.total})`, 20)} ${isoB.attractors.memories.length}`);
log(`${pad('Lateral  Fold-B', 25)} ${pad(`${latBOwn.meanBestSim.toFixed(4)} (${latBOwn.passCount}/${latBOwn.total})`, 20)} ${pad(`${latBCross.meanBestSim.toFixed(4)} (${latBCross.passCount}/${latBCross.total})`, 20)} ${foldB.attractors.memories.length}`);
log(`${pad('  delta B', 25)} ${pad(`${(latBOwn.meanBestSim - isoBOwn.meanBestSim) >= 0 ? '+' : ''}${(latBOwn.meanBestSim - isoBOwn.meanBestSim).toFixed(4)}`, 20)} ${pad(`${(latBCross.meanBestSim - isoBCross.meanBestSim) >= 0 ? '+' : ''}${(latBCross.meanBestSim - isoBCross.meanBestSim).toFixed(4)}`, 20)}`);
log('-'.repeat(78));

// ════════════════════════════════════════════════════════════════════════════
// PHASE 5: INFORMATION TRANSFER VERDICT
// ════════════════════════════════════════════════════════════════════════════
log('\n' + '='.repeat(78));
log('PHASE 5: INFORMATION TRANSFER VERDICT');
log('='.repeat(78));

// Cross-stream contamination = lateral cross - isolated cross
const contaminationA = latACross.meanBestSim - isoACross.meanBestSim;
const contaminationB = latBCross.meanBestSim - isoBCross.meanBestSim;

log(`\nCross-stream contamination (lateral_cross - isolated_cross):`);
log(`  Fold-A: ${contaminationA >= 0 ? '+' : ''}${contaminationA.toFixed(4)} (did Fold-A absorb B's distributions?)`);
log(`  Fold-B: ${contaminationB >= 0 ? '+' : ''}${contaminationB.toFixed(4)} (did Fold-B absorb A's distributions?)`);

const TRANSFER_THRESHOLD = 0.01; // Meaningful if > 1% improvement in cross alignment
const transferA = contaminationA > TRANSFER_THRESHOLD;
const transferB = contaminationB > TRANSFER_THRESHOLD;

log(`\nInformation transfer threshold: >${TRANSFER_THRESHOLD.toFixed(3)} cross-alignment increase`);
log(`  Fold-A absorbed B info: ${transferA ? 'YES' : 'NO'} (delta=${contaminationA.toFixed(4)})`);
log(`  Fold-B absorbed A info: ${transferB ? 'YES' : 'NO'} (delta=${contaminationB.toFixed(4)})`);

// Own-stream impact: did lateral blending HURT own-stream accuracy?
const ownImpactA = latAOwn.meanBestSim - isoAOwn.meanBestSim;
const ownImpactB = latBOwn.meanBestSim - isoBOwn.meanBestSim;
log(`\nOwn-stream impact from lateral blending:`);
log(`  Fold-A: ${ownImpactA >= 0 ? '+' : ''}${ownImpactA.toFixed(4)} (${ownImpactA >= -0.01 ? 'preserved' : 'degraded'})`);
log(`  Fold-B: ${ownImpactB >= 0 ? '+' : ''}${ownImpactB.toFixed(4)} (${ownImpactB >= -0.01 ? 'preserved' : 'degraded'})`);

// Count memories that are dominantly cross-aligned
let crossDomA = 0, crossDomB = 0;
for (let m = 0; m < foldA.attractors.memories.length; m++) {
  if (latACrossAlign[m].best.sim > latAOwnAlign[m].best.sim) crossDomA++;
}
for (let m = 0; m < foldB.attractors.memories.length; m++) {
  if (latBCrossAlign[m].best.sim > latBOwnAlign[m].best.sim) crossDomB++;
}
log(`\nCross-dominant memories (higher cross than own alignment):`);
log(`  Fold-A: ${crossDomA}/${foldA.attractors.memories.length}`);
log(`  Fold-B: ${crossDomB}/${foldB.attractors.memories.length}`);

log('\n' + '='.repeat(78));
log('CONCLUSION');
log('='.repeat(78));
const anyTransfer = transferA || transferB;
if (anyTransfer) {
  log('\nInformation DOES transfer through lateral composition.');
  log('The 0.2 lateral blend weight creates measurable cross-stream contamination.');
  if (ownImpactA >= -0.02 && ownImpactB >= -0.02) {
    log('Own-stream fidelity is preserved — lateral blending adds information without');
    log('destroying the primary signal. This is compositional intelligence.');
  } else {
    log('However, own-stream fidelity is degraded — the lateral signal interferes.');
  }
} else {
  log('\nInformation does NOT meaningfully transfer through lateral composition.');
  log('The 0.8/0.2 blend ratio keeps each fold primarily on its own distribution.');
  if (contaminationA > 0 || contaminationB > 0) {
    log(`Minor leakage detected (A: ${contaminationA.toFixed(4)}, B: ${contaminationB.toFixed(4)}) but below threshold.`);
  }
}
log('='.repeat(78));

// Save results
const outPath = path.join(ROOT, 'results', 'lateral_different_inputs.txt');
writeFileSync(outPath, allLines.join('\n') + '\n');
log(`\nCaptured to ${outPath}`);
