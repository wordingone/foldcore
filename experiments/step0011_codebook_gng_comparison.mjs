/**
 * run_gng.mjs — Step 11: Growing Neural Gas on CSI embeddings
 *
 * Implements GNG algorithm, feeds same csi_embedded.json, compares
 * node-division alignment to FluxCore attractor results from Step 10.
 *
 * GNG params: ε_b=0.2, ε_n=0.006, λ=100, max_age=50, α=0.5
 */

import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = path.join(__dirname, '..');

// ── utilities ─────────────────────────────────────────────────────────────
const dot  = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const dist2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i]-b[i]; s += d*d; } return s; };
const normalize = v => {
  let s = 0; for (let i = 0; i < v.length; i++) s += v[i]*v[i];
  const n = Math.sqrt(s) + 1e-12;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
};
const clone = v => new Float64Array(v);

// ── GNG implementation ─────────────────────────────────────────────────────
class GNG {
  constructor(dim, opts = {}) {
    this.dim      = dim;
    this.eps_b    = opts.eps_b    ?? 0.2;     // winner learning rate
    this.eps_n    = opts.eps_n    ?? 0.006;   // neighbor learning rate
    this.lambda   = opts.lambda   ?? 100;     // insertion interval
    this.max_age  = opts.max_age  ?? 50;      // edge removal threshold
    this.alpha    = opts.alpha    ?? 0.5;     // error decay on insertion
    this.d        = opts.d        ?? 0.995;   // global error decay per tick

    // nodes: array of {vec, error}
    // edges: Map of "i-j" (i<j) -> age
    // adj:   node -> Set of neighbor node indices
    this.nodes = [];
    this.edges = new Map();
    this.adj   = [];
    this.tick  = 0;
  }

  _addNode(vec) {
    const idx = this.nodes.length;
    this.nodes.push({ vec: clone(vec), error: 0 });
    this.adj.push(new Set());
    return idx;
  }

  _addEdge(i, j) {
    const key = i < j ? `${i}-${j}` : `${j}-${i}`;
    this.edges.set(key, 0);
    this.adj[i].add(j);
    this.adj[j].add(i);
  }

  _removeEdge(i, j) {
    const key = i < j ? `${i}-${j}` : `${j}-${i}`;
    this.edges.delete(key);
    this.adj[i].delete(j);
    this.adj[j].delete(i);
  }

  _edgeKey(i, j) { return i < j ? `${i}-${j}` : `${j}-${i}`; }

  init(samples) {
    // Seed with 2 random samples
    const i0 = Math.floor(Math.random() * samples.length);
    let   i1 = Math.floor(Math.random() * samples.length);
    while (i1 === i0) i1 = Math.floor(Math.random() * samples.length);
    const n0 = this._addNode(samples[i0]);
    const n1 = this._addNode(samples[i1]);
    this._addEdge(n0, n1);
  }

  step(input) {
    this.tick++;

    // 1. Find winner (s1) and runner-up (s2)
    let s1 = -1, s2 = -1, d1 = Infinity, d2 = Infinity;
    for (let i = 0; i < this.nodes.length; i++) {
      const d = dist2(this.nodes[i].vec, input);
      if (d < d1) { d2 = d1; s2 = s1; d1 = d; s1 = i; }
      else if (d < d2) { d2 = d; s2 = i; }
    }

    // 2. Accumulate error at winner
    this.nodes[s1].error += Math.sqrt(d1);

    // 3. Move winner and its neighbors toward input
    const vs1 = this.nodes[s1].vec;
    for (let i = 0; i < this.dim; i++) vs1[i] += this.eps_b * (input[i] - vs1[i]);
    normalize(vs1);

    for (const n of this.adj[s1]) {
      const vn = this.nodes[n].vec;
      for (let i = 0; i < this.dim; i++) vn[i] += this.eps_n * (input[i] - vn[i]);
      normalize(vn);
    }

    // 4. Update/create edge between s1 and s2, reset its age
    const key12 = this._edgeKey(s1, s2);
    if (!this.edges.has(key12)) {
      this._addEdge(s1, s2);
    }
    this.edges.set(key12, 0);

    // 5. Age all edges from winner; remove old ones
    const toRemove = [];
    for (const n of [...this.adj[s1]]) {
      const key = this._edgeKey(s1, n);
      const age = (this.edges.get(key) ?? 0) + 1;
      if (age > this.max_age) {
        toRemove.push([s1, n]);
      } else {
        this.edges.set(key, age);
      }
    }
    for (const [a, b] of toRemove) this._removeEdge(a, b);

    // 6. Remove isolated nodes
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      if (this.adj[i].size === 0 && i !== s1) {
        this._removeNode(i);
        if (s1 > i) s1--;
        if (s2 > i) s2--;
      }
    }

    // 7. Insert new node every λ steps
    if (this.tick % this.lambda === 0) {
      // Find node with highest error
      let qIdx = 0;
      for (let i = 1; i < this.nodes.length; i++) {
        if (this.nodes[i].error > this.nodes[qIdx].error) qIdx = i;
      }
      // Find neighbor of q with highest error
      let fIdx = -1;
      for (const n of this.adj[qIdx]) {
        if (fIdx < 0 || this.nodes[n].error > this.nodes[fIdx].error) fIdx = n;
      }
      if (fIdx >= 0) {
        // Insert new node between q and f
        const newVec = new Float64Array(this.dim);
        for (let i = 0; i < this.dim; i++) newVec[i] = 0.5 * (this.nodes[qIdx].vec[i] + this.nodes[fIdx].vec[i]);
        normalize(newVec);
        const rIdx = this._addNode(newVec);
        this._removeEdge(qIdx, fIdx);
        this._addEdge(qIdx, rIdx);
        this._addEdge(fIdx, rIdx);
        // Decay errors
        this.nodes[qIdx].error *= this.alpha;
        this.nodes[fIdx].error *= this.alpha;
        this.nodes[rIdx].error = (this.nodes[qIdx].error + this.nodes[fIdx].error) / 2;
      }
    }

    // 8. Decay all errors
    for (const node of this.nodes) node.error *= this.d;

    return { nodeCount: this.nodes.length, s1, d1 };
  }

  _removeNode(idx) {
    // Clean up edges
    for (const n of [...this.adj[idx]]) this._removeEdge(idx, n);
    // Remove node (swap-with-last)
    const last = this.nodes.length - 1;
    if (idx !== last) {
      // Remap last → idx
      this.nodes[idx] = this.nodes[last];
      this.adj[idx]   = this.adj[last];
      // Update all edges referencing 'last'
      for (const n of [...this.adj[idx]]) {
        const oldKey = this._edgeKey(last, n);
        const age = this.edges.get(oldKey) ?? 0;
        this.edges.delete(oldKey);
        this.adj[n].delete(last);
        this.adj[n].add(idx);
        const newKey = this._edgeKey(idx, n);
        this.edges.set(newKey, age);
      }
    }
    this.nodes.pop();
    this.adj.pop();
  }
}

// ── main ──────────────────────────────────────────────────────────────────
const lines = [];
const log = s => { lines.push(s); process.stdout.write(s + '\n'); };

log('═'.repeat(72));
log('GNG (Growing Neural Gas) — CSI Corpus Embeddings');
log('═'.repeat(72));

const embedded = JSON.parse(readFileSync(path.join(BASE, 'data', 'csi_embedded.json'), 'utf8'));
const centers  = JSON.parse(readFileSync(path.join(BASE, 'data', 'csi_division_centers.json'), 'utf8'));
const divNames = Object.keys(centers).sort();

log(`\nLoaded ${embedded.length} records, ${divNames.length} division centers`);
log(`Vector dimension: ${embedded[0].vec.length}`);

const DIM = 384;
const gng = new GNG(DIM, { eps_b: 0.2, eps_n: 0.006, lambda: 100, max_age: 50, alpha: 0.5, d: 0.995 });

const vecs = embedded.map(r => new Float64Array(r.vec));
gng.init(vecs);

log('\nGNG params: ε_b=0.2, ε_n=0.006, λ=100, max_age=50, α=0.5, d=0.995');
log('Running GNG on 1920 CSI embeddings...\n');

for (let tick = 0; tick < vecs.length; tick++) {
  const result = gng.step(vecs[tick]);
  if ((tick + 1) % 100 === 0 || tick === vecs.length - 1) {
    const div = embedded[tick].division;
    log(`  Tick ${String(tick+1).padStart(4)}: nodes=${result.nodeCount}, dist_to_winner=${Math.sqrt(result.d1).toFixed(4)}, div=${div}`);
  }
}

// ── node-division alignment ────────────────────────────────────────────────
log('\n' + '─'.repeat(72));
log('NODE-DIVISION ALIGNMENT');
log('─'.repeat(72));

log(`\nFinal node count: ${gng.nodes.length}`);
log(`\n${'Node'.padEnd(5)} ${'Best div'.padEnd(10)} ${'Sim'.padEnd(7)} ${'2nd div'.padEnd(10)} ${'2nd sim'.padEnd(7)} Specificity`);
log('─'.repeat(72));

let passCount = 0;
const alignments = [];

for (let n = 0; n < gng.nodes.length; n++) {
  const nodeVec = gng.nodes[n].vec;
  const sims = divNames.map(div => ({
    div,
    sim: dot(nodeVec, new Float64Array(centers[div]))
  })).sort((a, b) => b.sim - a.sim);

  const best = sims[0], second = sims[1];
  const specificity = best.sim - second.sim;
  const passes = best.sim > 0.3;
  if (passes) passCount++;
  alignments.push({ node: n, best, second, specificity, passes });

  const mark = passes ? '✓' : ' ';
  log(`${String(n).padEnd(5)} div ${best.div.padEnd(5)} ${best.sim.toFixed(4)}  div ${second.div.padEnd(5)} ${second.sim.toFixed(4)}  +${specificity.toFixed(4)} ${mark}`);
}

const meanSim  = alignments.reduce((s, a) => s + a.best.sim, 0) / alignments.length;
const meanSpec = alignments.reduce((s, a) => s + a.specificity, 0) / alignments.length;

log('\n' + '═'.repeat(72));
log('GNG FINAL RESULTS');
log('═'.repeat(72));
log(`  Total nodes          : ${gng.nodes.length}`);
log(`  Pass (dot > 0.3)     : ${passCount}/${gng.nodes.length}`);
log(`  Mean best sim        : ${meanSim.toFixed(4)}`);
log(`  Mean specificity     : ${meanSpec.toFixed(4)}`);

// ── comparison table ───────────────────────────────────────────────────────
log('\n' + '═'.repeat(72));
log('COMPARISON: FluxCore vs GNG');
log('═'.repeat(72));
log(`${'Metric'.padEnd(30)} ${'FluxCore'.padEnd(15)} ${'GNG'}`);
log('─'.repeat(60));
log(`${'Node/memory count'.padEnd(30)} ${'359'.padEnd(15)} ${gng.nodes.length}`);
log(`${'Pass rate (dot>0.3)'.padEnd(30)} ${'357/359 (99.4%)'.padEnd(15)} ${passCount}/${gng.nodes.length} (${(100*passCount/gng.nodes.length).toFixed(1)}%)`);
log(`${'Mean best sim'.padEnd(30)} ${'0.5710'.padEnd(15)} ${meanSim.toFixed(4)}`);
log(`${'Mean specificity'.padEnd(30)} ${'0.0584'.padEnd(15)} ${meanSpec.toFixed(4)}`);
log(`${'Online / streaming'.padEnd(30)} ${'Yes'.padEnd(15)} Yes`);
log(`${'Requires graph structure'.padEnd(30)} ${'No'.padEnd(15)} Yes (edges)`);
log(`${'Hyperparams'.padEnd(30)} ${'3 (spawn/merge/prune)'.padEnd(15)} 5 (ε_b,ε_n,λ,max_age,α)`);
log('═'.repeat(72));

writeFileSync(path.join(BASE, 'results', 'gng_comparison.txt'), lines.join('\n') + '\n');
process.stdout.write('\nCaptured to results/gng_comparison.txt\n');
