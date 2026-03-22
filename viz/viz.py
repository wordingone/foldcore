#!/usr/bin/env python3
"""
Search Space Visualizer — generates search_space.html from repo state.

Every point is a real experiment. No synthetic data.

Reads:
  - RESEARCH_STATE.md (experiment results — primary source)
  - experiments/run_step*.py, experiments/foldcore-steps/run_step*.py (fill gaps)

Clusters aligned with the paper:
  0: representation (encoding, centering, resolution, projection)
  1: navigation (argmin, action space, exploration strategy, death avoidance)
  2: depth (L2+, pipeline, mgu, puq, reward disconnect, mode map)
  3: R3 (self-modification, ops-as-data, eigenform, recode, multi-buffer)
  4: transfer (cross-game, chain, classification, domain isolation)
  5: architecture (family comparisons, structural tests, constraint validation)

Result → shape (not color override):
  SOLVED → icosahedron (bright)
  KILL → octahedron (angular)
  SIGNAL → diamond (rotated cube)
  other → sphere

Diagnostic kills score HIGHER than uninformative signals.
Trajectory lines connect experiments in the same thread (572→572b→...→572u).
"""
import os, re, json, math, glob
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
CLUSTERS = ['representation', 'navigation', 'depth', 'R3', 'transfer', 'architecture']
CC_LIST = ['#fdcb6e', '#55eedd', '#ff88bb', '#ff6b6b', '#bb99ff', '#00dd99']

# Fibonacci sphere for 6 cluster axis directions
golden = math.pi * (3 - math.sqrt(5))
CLUSTER_DIRS = []
for i in range(6):
    y = 1 - (i / 5) * 2
    radius = math.sqrt(max(0, 1 - y * y))
    theta = golden * i
    CLUSTER_DIRS.append([round(math.cos(theta) * radius, 3), round(y, 3), round(math.sin(theta) * radius, 3)])


# ── Parsers ──────────────────────────────────────────────────────────────────

def get_docstring(filepath):
    try:
        with open(filepath, 'r', errors='replace') as f:
            content = f.read(3000)
        m = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if m: return m.group(1).strip()[:200]
        m = re.search(r"'''(.*?)'''", content, re.DOTALL)
        if m: return m.group(1).strip()[:200]
    except: pass
    return ""


def parse_research_state():
    """Parse RESEARCH_STATE.md for experiment entries."""
    rs_path = REPO / 'constraints' / 'RESEARCH_STATE.md'
    if not rs_path.exists(): return {}
    content = rs_path.read_text(errors='replace')
    entries = {}

    # Match "Step NNN:" or "Step NNNx:" at line start or after newline
    for m in re.finditer(r'(?:^|\n)\s*Step (\d+[a-z]?)[\s:]+(.+?)(?=\n(?:\s*Step \d|\n|$))', content, re.DOTALL):
        sid, desc = m.groups()
        # Collapse to single line, trim
        desc = ' '.join(desc.split())[:200]
        result = classify_result(desc)
        if sid not in entries or len(desc) > len(entries[sid].get('desc', '')):
            entries[sid] = {'desc': desc, 'result': result}

    # Also catch "Step NNN-NNN:" ranges
    for m in re.finditer(r'(?:^|\n)\s*Step (\d+)-(\d+)[\s:]+(.+?)(?=\n(?:\s*Step \d|\n|$))', content, re.DOTALL):
        s1, s2, desc = m.groups()
        desc = ' '.join(desc.split())[:200]
        result = classify_result(desc)
        for s in range(int(s1), int(s2) + 1):
            sid = str(s)
            if sid not in entries:
                entries[sid] = {'desc': desc, 'result': result}

    return entries


def scan_scripts():
    """Scan experiment script files."""
    scripts = {}
    for d in ['experiments/foldcore-steps', 'experiments']:
        for fp in glob.glob(str(REPO / d / 'run_step*.py')):
            m = re.search(r'step(\d+)', os.path.basename(fp))
            if m:
                step = int(m.group(1))
                scripts[step] = get_docstring(fp)
    return scripts


# ── Classification ───────────────────────────────────────────────────────────

def classify_result(desc):
    d = desc.upper()
    if 'SOLVED' in d: return 'S'
    if '5/5' in d and any(w in d for w in ['WIN', 'L1', 'L2', 'L3', 'PASS']): return 'S'
    if '3/3' in d and 'WIN' in d: return 'S'
    if 'SIGNAL' in d: return 'I'
    if 'CONFIRMED' in d: return 'C'
    if 'CHALLENGED' in d: return 'H'
    if 'NEUTRAL' in d or 'NOT SIGNIFICANT' in d: return 'N'
    if 'BLOCKED' in d: return 'B'
    if 'KILL' in d or 'FAIL' in d or 'DEGENERATE' in d: return 'K'
    if '0/3' in d or '0/5' in d or '0/10' in d: return 'K'
    if 'WIN' in d or 'PASS' in d: return 'S'
    if 'MARGINAL' in d: return 'N'
    return 'O'


def classify_cluster(desc, step_num):
    d = desc.lower()

    # R3 / self-modification — THE question
    if any(w in d for w in ['self-mod', 'self-obs', 'ops as data', 'recode', 'self-ref',
                            'r3 ', 'r3.', 'r3:', 'eigenform', 'per-edge', 'cerebellar',
                            'interpreter', 'frozen frame', 'multi-buffer', 'evolutionary r3',
                            'grn', 'population', 'tape', 'program substrate']):
        return 3

    # Depth (L2+)
    if any(w in d for w in ['l2=', 'l3=', 'l4', 'level 2', 'level 3', 'level 4',
                            'mgu', 'puq', 'pipeline', 'bootstrap', 'mode map',
                            'reward disconnect', 'source analysis', 'background sub',
                            'rare-color', 'visit-all', 'budget', 'multi-level',
                            'sprite', 'palette', 'energy rout', 'dead reckoning',
                            'prev_cl', 'state puzzle', 'wall set']):
        return 2

    # Transfer / generalization
    if any(w in d for w in ['chain', 'cifar', 'cross-game', 'domain isol', 'p-mnist',
                            'classification', 'contamination', 'self-label',
                            'all 3 games', 'generaliz', 'transfer', 'per-domain',
                            'nmi']):
        return 4

    # Representation / encoding
    if any(w in d for w in ['encoding', 'centering', 'centered', 'avgpool', 'resolution',
                            'raw 64', 'diff-frame', 'concat', 'normalize',
                            'pca', 'projection', 'rbf', 'kernel', '16x16', '64x64',
                            'sparse codebook', 'threshold', 'spawn', 'codebook size',
                            'cosine satur', 'timer', 'feature select']):
        return 0

    # Architecture / family / structural
    if any(w in d for w in ['lsh', 'k-means', 'kmeans', 'l2 k', 'reservoir', 'hebbian',
                            'bloom', 'cellular', 'kd-tree', 'absorb', 'markov',
                            'structural', 'constraint', 'audit', 'r1 ', 'r2 ', 'r4 ',
                            'r5 ', 'r6 ', 'validated', 'algorithm invariance',
                            'invariance', 'split tree']):
        return 5

    # Navigation / mechanism (default for game experiments)
    if any(w in d for w in ['argmin', 'action', 'exploration', 'novelty', 'ucb',
                            'softmax', 'death penalty', 'death count', 'wall avoid',
                            'click', 'zone', 'bfs', 'graph', 'edge count',
                            'frontier', 'coverage', 'noisy tv', 'entrap']):
        return 1

    # Phase 1 default: representation
    if step_num <= 320:
        return 0

    return 1  # default: navigation


def resolution_score(result, desc):
    """Higher = more resolved. Center of sphere. Diagnostic kills valued."""
    d = desc.lower()

    # Milestones — must check SPECIFIC level=result patterns, not just co-occurrence
    # "L1=5/5 ... L2=0/5" should score as L1 solved (0.80), not L2 solved (0.90)
    import re as _re
    if _re.search(r'l[34]\S*[=: ]*5/5', d): return 0.95
    if _re.search(r'l2\S*[=: ]*5/5', d): return 0.90
    if 'first ever' in d: return 0.88
    if 'all 3 games' in d or 'all 7' in d or 'all 6' in d: return 0.85
    if 'solved' in d or ('5/5' in d and ('win' in d or 'pass' in d)): return 0.80
    if result == 'S': return 0.75

    # Confirmed / validated
    if result == 'C': return 0.70
    if 'challenged' in d: return 0.65

    # Diagnostic kills — MORE valuable than uninformative signals
    if result == 'K':
        if any(w in d for w in ['root cause', 'answered', 'precisely located',
                                'key finding', 'key:', 'critical', 'barrier',
                                'closed', 'thread closed', 'confirmed']):
            return 0.60
        if any(w in d for w in ['structural insight', 'key', 'deeper']):
            return 0.50
        if any(w in d for w in ['marginal', 'but', 'not significant', 'ns']):
            return 0.40
        return 0.28

    # Signals
    if result == 'I': return 0.55
    if result == 'N': return 0.35
    if result == 'B': return 0.15

    return 0.18


# ── Build ────────────────────────────────────────────────────────────────────

def build_experiments():
    rs_entries = parse_research_state()
    scripts = scan_scripts()
    all_exps = {}

    # From RESEARCH_STATE (richest data)
    for sid, entry in rs_entries.items():
        step_num = int(re.match(r'(\d+)', sid).group(1))
        desc = entry['desc']
        result = entry['result']
        ci = classify_cluster(desc, step_num)
        score = resolution_score(result, desc)
        # Game epoch: ARC Prize API updated LS20/FT09 on March 20 (step ~619)
        epoch = 'new' if step_num >= 619 else 'old'
        all_exps[sid] = {
            'step': step_num, 'desc': desc[:80],
            'ci': ci, 'res': result, 'score': score,
            'phase': 1 if step_num <= 416 else 2, 'epoch': epoch
        }

    # From scripts (fill gaps — phase 1 at lower opacity)
    for step_num, doc in scripts.items():
        sid = str(step_num)
        if sid in all_exps or any(k.startswith(sid) for k in all_exps):
            continue
        if not doc:
            doc = f'Step {step_num}'
        ci = classify_cluster(doc, step_num)
        phase = 1 if step_num <= 416 else 2
        epoch = 'new' if step_num >= 619 else 'old'
        all_exps[sid] = {
            'step': step_num, 'desc': doc[:80],
            'ci': ci, 'res': 'O', 'score': 0.15,
            'phase': phase, 'epoch': epoch
        }

    return all_exps


def find_trajectories(exps):
    """Find experiment threads: same base step (572, 572b, ...) or close sequential."""
    threads = defaultdict(list)
    for sid in exps:
        base = re.match(r'(\d+)', sid).group(1)
        threads[base].append(sid)

    trajs = []
    for base, sids in threads.items():
        if len(sids) >= 2:
            # Sort: numeric base first, then alphabetical suffix
            sids.sort(key=lambda s: (int(re.match(r'(\d+)', s).group(1)),
                                      re.sub(r'^\d+', '', s) or ''))
            trajs.append(sids)

    # Also detect close sequential runs within the same cluster
    # e.g., 477, 478, 479, 480, 481, 482 (targeted exploration kills)
    by_cluster = defaultdict(list)
    for sid, e in exps.items():
        if re.match(r'^\d+$', sid):  # only pure numeric
            by_cluster[e['ci']].append(int(sid))

    for ci, steps in by_cluster.items():
        steps.sort()
        run = [str(steps[0])]
        for i in range(1, len(steps)):
            if steps[i] - steps[i - 1] <= 2:  # within 2 steps
                run.append(str(steps[i]))
            else:
                if len(run) >= 3:  # only runs of 3+
                    trajs.append(run)
                run = [str(steps[i])]
        if len(run) >= 3:
            trajs.append(run)

    return trajs


def to_3d(exps):
    """Convert experiments to 3D coordinates on the sphere."""
    compact = []
    for sid, e in sorted(exps.items(), key=lambda x: (x[1]['step'], x[0])):
        ci = e['ci']
        score = e['score']
        r = (1 - score) * 2.2 + 0.12
        s = e['step']

        # Deterministic jitter within cluster cone
        h1 = ((s * 2654435761) % 1000) / 1000.0
        h2 = ((s * 40503) % 1000) / 1000.0
        h3 = ((s * 12345) % 1000) / 1000.0
        j1 = h1 * 0.55 - 0.275
        j2 = h2 * 0.55 - 0.275
        j3 = h3 * 0.55 - 0.275

        cd = CLUSTER_DIRS[ci]
        dx, dy, dz = cd[0] + j1, cd[1] + j2, cd[2] + j3
        mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if mag > 0: dx /= mag; dy /= mag; dz /= mag

        # [x, y, z, cluster, result, step, score*100, desc, phase]
        compact.append([
            int(dx * r * 100), int(dy * r * 100), int(dz * r * 100),
            ci, e['res'], s, int(score * 100), e['desc'], e['phase']
        ])

    return compact


# ── HTML ─────────────────────────────────────────────────────────────────────

def generate_html(compact, trajectories, exps):
    n = len(compact)
    # Build step→index map for trajectory rendering
    step_to_idx = {}
    for i, c in enumerate(compact):
        step_to_idx[str(c[5])] = i
        # Also try with letter suffix
    for sid in exps:
        step_num = exps[sid]['step']
        if sid not in step_to_idx:
            # Find matching compact entry
            for i, c in enumerate(compact):
                if c[5] == step_num and str(step_num) not in step_to_idx:
                    step_to_idx[sid] = i

    # Build trajectory index pairs
    traj_lines = []
    for thread in trajectories:
        for i in range(len(thread) - 1):
            s1, s2 = thread[i], thread[i + 1]
            i1 = step_to_idx.get(s1)
            i2 = step_to_idx.get(s2)
            if i1 is not None and i2 is not None:
                traj_lines.append([i1, i2])

    counts = Counter(CLUSTERS[c[3]] for c in compact)
    p1 = sum(1 for c in compact if c[8] == 1)
    p2 = sum(1 for c in compact if c[8] == 2)

    legend = ''.join(f'<span style="color:{CC_LIST[i]}">{CLUSTERS[i]}</span>' for i in range(6))
    count_str = ' · '.join(f'{CLUSTERS[i]}:{counts.get(CLUSTERS[i], 0)}' for i in range(6))

    html = f'''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>THE SEARCH</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;overflow:hidden;background:#000;touch-action:none}}
#ui{{position:fixed;top:10px;left:10px;z-index:10;font:clamp(10px,2.5vw,13px)/1.5 monospace;pointer-events:none;max-width:55vw}}
#ui h1{{font-size:clamp(18px,4.5vw,24px);color:#ddd;letter-spacing:3px;font-weight:300}}
#ui .sub{{color:#777;margin-top:3px;font-size:clamp(10px,2.5vw,13px)}}
#ui .counts{{color:#555;font-size:clamp(8px,2vw,10px);margin-top:2px}}
#ui .hint{{color:#333;font-size:clamp(8px,2vw,10px);margin-top:8px}}
#lg{{position:fixed;top:10px;right:10px;z-index:10;font:clamp(9px,2.2vw,12px)/1.5 monospace;pointer-events:none}}
#lg span{{display:block;margin-bottom:2px}}
#sh{{position:fixed;bottom:10px;right:10px;z-index:10;font:clamp(9px,2.2vw,11px)/1.5 monospace;pointer-events:none;color:#555}}
#sh span{{display:block;margin-bottom:1px}}
.lbl{{position:absolute;pointer-events:none;font:clamp(9px,2.2vw,11px)/1.3 monospace;padding:5px 10px;border-radius:3px;white-space:nowrap;z-index:5;background:rgba(0,0,0,.92)}}
</style></head><body>
<div id="ui">
<h1>THE SEARCH</h1>
<p class="sub">{n} experiments · 12 families · 3 games · 6 rules</p>
<p class="counts">{count_str}</p>
<p class="hint">center = resolved · edge = open<br>drag rotate · scroll zoom · click inspect<br><span style="color:#ffaa00">&#9675;</span> = new game epoch (LS20/FT09 updated March 20)</p>
</div>
<div id="lg">{legend}</div>
<div id="sh"><span style="color:#00ff88">&#9670; solved</span><span style="color:#ff4444">&#9671; kill</span><span style="color:#ffaa00">&#9650; signal</span><span style="color:#555">&#9679; open</span></div>
<svg id="lines" style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:4"></svg>
<div id="labels"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
var CN={json.dumps(CLUSTERS)};
var CC={json.dumps(CC_LIST)};
var CD={json.dumps(CLUSTER_DIRS)};
var D={json.dumps(compact)};
var TJ={json.dumps(traj_lines)};
var RN={{K:"KILL",S:"SOLVED",I:"SIGNAL",O:"",N:"NEUTRAL",C:"CONF",B:"BLOCKED",H:"CHALL"}};

var scene=new THREE.Scene();scene.background=new THREE.Color("#000");
var camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.05,60);
var camR=innerWidth<innerHeight?5.5:4.2;
var renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth,innerHeight);renderer.setPixelRatio(Math.min(devicePixelRatio,2));
document.body.appendChild(renderer.domElement);

// Center glow
var cMat=new THREE.MeshBasicMaterial({{color:"#ffff00",transparent:true,opacity:.4}});
var ctr=new THREE.Mesh(new THREE.SphereGeometry(.035,20,20),cMat);scene.add(ctr);
scene.add(new THREE.Mesh(new THREE.SphereGeometry(.12,16,16),new THREE.MeshBasicMaterial({{color:"#ffff22",transparent:true,opacity:.04}})));

// Cluster axis lines with labels
CD.forEach(function(d,i){{
  var pts=[new THREE.Vector3(0,0,0),new THREE.Vector3(d[0]*2.8,d[1]*2.8,d[2]*2.8)];
  var g=new THREE.BufferGeometry().setFromPoints(pts);
  scene.add(new THREE.Line(g,new THREE.LineBasicMaterial({{color:CC[i],transparent:true,opacity:.1}})));
}});

// Experiment points — shape encodes result
var meshes=[];
D.forEach(function(e){{
  var x=e[0]/100,y=e[1]/100,z=e[2]/100,ci=e[3],res=e[4],sc=e[6]/100,phase=e[8];
  var col=CC[ci];
  var sz=sc>.7?.028:sc>.4?.02:.012;
  var geo;
  if(res==="S"){{geo=new THREE.IcosahedronGeometry(sz*1.3,1)}}
  else if(res==="K"){{geo=new THREE.OctahedronGeometry(sz*1.1)}}
  else if(res==="I"||res==="H"||res==="C"){{geo=new THREE.TetrahedronGeometry(sz*1.2)}}
  else{{geo=new THREE.SphereGeometry(sz,6,6)}}
  var op=phase===1?(sc>.3?.5:.35):(sc>.5?1:sc>.2?.8:.6);
  var mat=new THREE.MeshBasicMaterial({{color:col,transparent:true,opacity:op}});
  var m=new THREE.Mesh(geo,mat);m.position.set(x,y,z);m.userData={{e:e,col:col}};
  scene.add(m);meshes.push(m);
  // Glow for high-resolution experiments
  if(sc>.65){{
    var gl=new THREE.Mesh(new THREE.SphereGeometry(sz*3.5,8,8),new THREE.MeshBasicMaterial({{color:col,transparent:true,opacity:.12}}));
    gl.position.set(x,y,z);scene.add(gl);
  }}
  // Epoch ring for new-game experiments (step >= 619)
  if(e[8]===2 && e[5]>=619){{
    var ring=new THREE.Mesh(new THREE.TorusGeometry(sz*2.2,.003,6,16),new THREE.MeshBasicMaterial({{color:"#ffaa00",transparent:true,opacity:.6}}));
    ring.position.set(x,y,z);scene.add(ring);
  }}
}});

// Trajectory lines connecting experiment threads
TJ.forEach(function(t){{
  var i1=t[0],i2=t[1];
  if(i1>=D.length||i2>=D.length)return;
  var p1=meshes[i1].position,p2=meshes[i2].position;
  var ci=D[i1][3];
  var g=new THREE.BufferGeometry().setFromPoints([p1.clone(),p2.clone()]);
  var line=new THREE.Line(g,new THREE.LineBasicMaterial({{color:CC[ci],transparent:true,opacity:.18}}));
  scene.add(line);
}});

// Label system
var labelsEl=document.getElementById("labels");
var svgEl=document.getElementById("lines");
var active=[];var MAX=6;var svgNS="http://www.w3.org/2000/svg";
function toScreen(pos){{var v=pos.clone().project(camera);return{{x:(v.x*.5+.5)*innerWidth,y:(-(v.y)*.5+.5)*innerHeight,behind:v.z>1}}}}
function clearLabels(){{while(active.length){{var a=active.pop();a.el.remove();a.line.remove()}}}}
function addLabel(e,mesh){{
  var step=e[5];for(var i=0;i<active.length;i++){{if(active[i].step===step)return}}
  if(active.length>=MAX){{var old=active.shift();old.el.remove();old.line.remove()}}
  var ci=e[3],res=e[4],sc=e[6]/100,desc=e[7],col=mesh.userData.col,rn=RN[res]||"";
  var rcol=res==="S"?"#00ff88":res==="K"?"#ff4444":res==="I"?"#ffaa00":"#555";
  var el=document.createElement("div");el.className="lbl";el.style.color=col;el.style.borderLeft="2px solid "+col;
  var txt="<b style='font-size:clamp(10px,2.6vw,12px)'>"+step+"</b> <span style='color:#555'>"+CN[ci]+"</span>";
  if(rn)txt+=" <span style='color:"+rcol+"'>"+rn+"</span>";
  if(e[5]>=619)txt+=" <span style='color:#ffaa00'>NEW</span>";
  txt+="<br><span style='color:#444;font-size:clamp(6px,1.6vw,8px)'>"+desc+"</span>";
  el.innerHTML=txt;labelsEl.appendChild(el);
  var line=document.createElementNS(svgNS,"line");line.setAttribute("stroke",col);line.setAttribute("stroke-opacity","0.3");line.setAttribute("stroke-width","1");svgEl.appendChild(line);
  active.push({{step:step,mesh:mesh,el:el,line:line}});
}}
function updateLabels(){{for(var i=active.length-1;i>=0;i--){{var a=active[i];var sp=toScreen(a.mesh.position);if(sp.behind){{a.el.style.display="none";a.line.style.display="none";continue}}a.el.style.display="block";a.line.style.display="block";var lx=sp.x+20,ly=sp.y-28;var ew=a.el.offsetWidth||100,eh=a.el.offsetHeight||30;if(lx+ew>innerWidth-10)lx=sp.x-ew-20;if(ly<8)ly=8;if(ly+eh>innerHeight-8)ly=innerHeight-eh-8;a.el.style.left=lx+"px";a.el.style.top=ly+"px";a.line.setAttribute("x1",sp.x);a.line.setAttribute("y1",sp.y);a.line.setAttribute("x2",lx<sp.x?lx+ew:lx);a.line.setAttribute("y2",ly+eh/2)}}}}

// Camera controls
var rot={{a:.8,b:1.3,drag:false,lx:0,ly:0,auto:true,pd:0,dd:0}};
function uc(){{camera.position.set(camR*Math.sin(rot.b)*Math.cos(rot.a),camR*Math.cos(rot.b),camR*Math.sin(rot.b)*Math.sin(rot.a));camera.lookAt(0,0,0)}}
uc();
var ray=new THREE.Raycaster(),mo=new THREE.Vector2();
ray.params.Points={{threshold:.05}};
function cast(x,y){{var r=renderer.domElement.getBoundingClientRect();mo.x=((x-r.left)/r.width)*2-1;mo.y=-((y-r.top)/r.height)*2+1;ray.setFromCamera(mo,camera);var h=ray.intersectObjects(meshes);if(h.length&&h[0].object.userData.e)addLabel(h[0].object.userData.e,h[0].object);else clearLabels()}}

var el=renderer.domElement;
el.addEventListener("mousedown",function(e){{rot.drag=true;rot.lx=e.clientX;rot.ly=e.clientY;rot.dd=0}});
window.addEventListener("mousemove",function(e){{if(rot.drag){{rot.auto=false;var dx=e.clientX-rot.lx,dy=e.clientY-rot.ly;rot.dd+=Math.abs(dx)+Math.abs(dy);rot.a+=dx*.005;rot.b=Math.max(.3,Math.min(2.8,rot.b+dy*.005));rot.lx=e.clientX;rot.ly=e.clientY;uc()}}}});
window.addEventListener("mouseup",function(e){{rot.drag=false;if(rot.dd<5)cast(e.clientX,e.clientY)}});
el.addEventListener("wheel",function(e){{e.preventDefault();camR=Math.max(1.2,Math.min(10,camR+e.deltaY*.004));uc()}},{{passive:false}});
var tstart=0;
el.addEventListener("touchstart",function(e){{e.preventDefault();tstart=Date.now();rot.dd=0;if(e.touches.length===1){{rot.drag=true;rot.lx=e.touches[0].clientX;rot.ly=e.touches[0].clientY}}else if(e.touches.length===2){{rot.pd=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY)}}}},{{passive:false}});
window.addEventListener("touchmove",function(e){{e.preventDefault();if(e.touches.length===1&&rot.drag){{rot.auto=false;var dx=e.touches[0].clientX-rot.lx,dy=e.touches[0].clientY-rot.ly;rot.dd+=Math.abs(dx)+Math.abs(dy);rot.a+=dx*.004;rot.b=Math.max(.3,Math.min(2.8,rot.b+dy*.004));rot.lx=e.touches[0].clientX;rot.ly=e.touches[0].clientY;uc()}}else if(e.touches.length===2){{var nd=Math.hypot(e.touches[0].clientX-e.touches[1].clientX,e.touches[0].clientY-e.touches[1].clientY);camR=Math.max(1.2,Math.min(10,camR-(nd-rot.pd)*.01));rot.pd=nd;uc()}}}},{{passive:false}});
window.addEventListener("touchend",function(e){{e.preventDefault();rot.drag=false;if(rot.dd<15&&Date.now()-tstart<400&&e.changedTouches.length)cast(e.changedTouches[0].clientX,e.changedTouches[0].clientY)}},{{passive:false}});
window.addEventListener("resize",function(){{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}});

// Animation
(function anim(){{
  requestAnimationFrame(anim);
  if(rot.auto){{rot.a+=.0008;uc()}}
  var p=1+Math.sin(Date.now()*.0012)*.3;
  ctr.scale.set(p,p,p);cMat.opacity=.3+Math.sin(Date.now()*.0012)*.15;
  updateLabels();
  renderer.render(scene,camera);
}})();
</script></body></html>'''

    return html


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Parsing repo...')
    exps = build_experiments()
    print(f'  {len(exps)} experiments found')

    p1 = sum(1 for e in exps.values() if e['phase'] == 1)
    p2 = sum(1 for e in exps.values() if e['phase'] == 2)
    print(f'  Phase 1: {p1}, Phase 2: {p2}')

    compact = to_3d(exps)
    trajs = find_trajectories(exps)
    print(f'  {len(trajs)} trajectory threads')

    counts = Counter(CLUSTERS[e['ci']] for e in exps.values())
    for name in CLUSTERS:
        print(f'  {name}: {counts.get(name, 0)}')

    html = generate_html(compact, trajs, exps)
    out_path = REPO / 'viz' / 'search_space.html'
    out_path.write_text(html, encoding='utf-8')
    print(f'\nWritten {len(html):,} bytes to {out_path}')
    print('Open search_space.html in a browser.')
