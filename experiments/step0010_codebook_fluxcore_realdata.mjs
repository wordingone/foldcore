/**
 * run_fluxcore_realdata.mjs — Step 10: FluxCore on CSI embeddings
 *
 * Feeds 1920 unit-vector embeddings (384-dim, 33 CSI divisions) through
 * ActiveInferenceEngine and checks if attractors align with CSI divisions.
 */

import { readFileSync } from 'fs';
import { createWriteStream } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = path.join(__dirname, '..');

// ── utilities (copied from fluxcore_entity.mjs) ───────────────────────────
let _rng = 42;
const rand = () => { _rng = (Math.imul(1664525, _rng) + 1013904223) >>> 0; return (_rng >>> 0) / 4294967296; };
const randN = () => { const u = rand() + 1e-12, v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const normalize = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; const n = Math.sqrt(s) + 1e-12; for (let i = 0; i < v.length; i++) v[i] /= n; return v; };
const randUnit = d => { const v = new Float64Array(d); for (let i = 0; i < d; i++) v[i] = randN(); return normalize(v); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const clone = v => new Float64Array(v);

// ── DynamicAttractorField ─────────────────────────────────────────────────
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
}

// ── ActiveInferenceEngine ─────────────────────────────────────────────────
class ActiveInferenceEngine {
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
      s[idx] = si + alr * ri + (alr * 0.5) * d * grad + memW * mi;
    }
    normalize(s);

    for (let i = 0; i < dim; i++) {
      this.velocity[i] = velDecay * this.velocity[i] + velGain * (s[i] - this.prevSelf[i]);
    }
    this.prevSelf = clone(s);

    return { surprise, attractorAction: result.action, memoryCount: result.totalMemories };
  }
}

// ── main ──────────────────────────────────────────────────────────────────
const lines = [];
const log = s => { lines.push(s); process.stdout.write(s + '\n'); };

log('═'.repeat(72));
log('FluxCore — Real Data: CSI Corpus Embeddings');
log('═'.repeat(72));

const embedded  = JSON.parse(readFileSync(path.join(BASE, 'data', 'csi_embedded.json'), 'utf8'));
const centers   = JSON.parse(readFileSync(path.join(BASE, 'data', 'csi_division_centers.json'), 'utf8'));
const divNames  = Object.keys(centers).sort();

log(`\nLoaded ${embedded.length} records, ${divNames.length} division centers`);
log(`Vector dimension: ${embedded[0].vec.length}`);
log(`Divisions: ${divNames.join(', ')}\n`);

// Engine config
const DIM = 384;
const engine = new ActiveInferenceEngine(DIM, { spawnThreshold: 0.5, pruneThreshold: 2000 });

log('Running FluxCore on CSI embeddings...');
log('(61 records per division, natural distribution-switching pattern)\n');

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
    const division = embedded[tick].division;
    log(`  Tick ${String(tick + 1).padStart(4)}: memories=${result.memoryCount}, surprise=${result.surprise.toFixed(4)}, avg(last100)=${avgSurp.toFixed(4)}, div=${division}`);
  }
}

log(`\nTotal spawns: ${spawnCount}`);

// ── memory-division alignment ─────────────────────────────────────────────
log('\n' + '─'.repeat(72));
log('MEMORY-DIVISION ALIGNMENT');
log('─'.repeat(72));

const mems = engine.attractors.memories;
const meta = engine.attractors.memoryMeta;

log(`\nFinal memory count: ${mems.length}`);
log(`\n${'Mem'.padEnd(4)} ${'Uses'.padEnd(6)} ${'Age'.padEnd(6)} ${'Best div'.padEnd(10)} ${'Sim'.padEnd(7)} ${'2nd div'.padEnd(10)} ${'2nd sim'.padEnd(7)} Specificity`);
log('─'.repeat(72));

let passCount = 0;
const alignments = [];

for (let m = 0; m < mems.length; m++) {
  const mem = mems[m];
  const sims = divNames.map(div => ({
    div,
    sim: dot(mem, new Float64Array(centers[div]))
  })).sort((a, b) => b.sim - a.sim);

  const best = sims[0], second = sims[1];
  const specificity = best.sim - second.sim;  // how much better than 2nd
  const passes = best.sim > 0.3;
  if (passes) passCount++;

  alignments.push({ mem: m, best, second, specificity, passes });
  const useCount = meta[m]?.useCount || 0;
  const age = engine.attractors.tick - (meta[m]?.birthTick || 0);
  const mark = passes ? '✓' : ' ';
  log(`${String(m).padEnd(4)} ${String(useCount).padEnd(6)} ${String(age).padEnd(6)} div ${best.div.padEnd(5)} ${best.sim.toFixed(4)}  div ${second.div.padEnd(5)} ${second.sim.toFixed(4)}  +${specificity.toFixed(4)} ${mark}`);
}

log('\n' + '═'.repeat(72));
log('FINAL RESULTS');
log('═'.repeat(72));
log(`  Total memories       : ${mems.length}`);
log(`  Pass (dot > 0.3)     : ${passCount}/${mems.length}`);
log(`  Mean best sim        : ${(alignments.reduce((s, a) => s + a.best.sim, 0) / alignments.length).toFixed(4)}`);
log(`  Mean specificity     : ${(alignments.reduce((s, a) => s + a.specificity, 0) / alignments.length).toFixed(4)}`);
log(`  Total spawns         : ${spawnCount}`);

const pass = passCount > 0;
log(`\n  → ${pass ? 'PASS ✓ — memories align with recognizable CSI divisions' : 'FAIL ✗ — no memory exceeds dot > 0.3'}`);
log('═'.repeat(72));

import { writeFileSync } from 'fs';
writeFileSync(path.join(BASE, 'results', 'realdata_first.txt'), lines.join('\n') + '\n');
process.stdout.write('\nCaptured to results/realdata_first.txt\n');
