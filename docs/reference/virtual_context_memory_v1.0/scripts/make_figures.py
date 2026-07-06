from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

NAVY = '#17324d'
BLUE = '#2b628f'
TEAL = '#2c827b'
PURPLE = '#6f5a8e'
ORANGE = '#b76b2e'
GRAY = '#56616c'
RED = '#a3413d'
LIGHT_BLUE = '#e8f1f7'
LIGHT_TEAL = '#e5f1ef'
LIGHT_PURPLE = '#f0edf5'
LIGHT_ORANGE = '#f8eee8'
LIGHT_GRAY = '#f1f3f5'
LIGHT_RED = '#f7e9e8'
WHITE = '#ffffff'


def box(ax, x, y, w, h, text, edge=BLUE, face=WHITE, fs=12.5, lw=2,
        radius=0.018, weight='bold', align='center', pad=0.01, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={pad},rounding_size={radius}',
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=z
    )
    ax.add_patch(p)
    ha = 'center' if align == 'center' else 'left'
    tx = x + w / 2 if align == 'center' else x + 0.018
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs,
            color=NAVY, fontweight=weight, wrap=True, zorder=z + 1)
    return p


def arrow(ax, start, end, color=GRAY, lw=2, style='-|>', dash=None, rad=0, z=3):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color,
                        connectionstyle=f'arc3,rad={rad}', zorder=z)
    if dash:
        a.set_linestyle(dash)
    ax.add_patch(a)
    return a


def setup(title, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.text(0.5, 0.965, title, ha='center', va='top', fontsize=24,
            fontweight='bold', color=NAVY)
    return fig, ax


# Figure 1 in the paper: identity, views, residency (clean, non-crossing layout)
fig, ax = setup('Semantic pages separate identity, task-relative views, and residency', (17, 10.5))
box(ax, 0.05, 0.76, 0.25, 0.13,
    'STABLE ADDRESS\nvcm://principal/namespace/\nobject@version#view\n+ content hash',
    TEAL, LIGHT_TEAL, 14)
box(ax, 0.375, 0.76, 0.25, 0.13,
    'PAGE MANIFEST\ntype | authority | scope | validity\ndependencies | capabilities | risk',
    PURPLE, LIGHT_PURPLE, 13.5)
box(ax, 0.70, 0.76, 0.25, 0.13,
    'AUTHORITATIVE SUBSTRATE\nraw events | files | tool records\nmodality-native sources | tombstones',
    GRAY, LIGHT_GRAY, 13.5)
arrow(ax, (0.30, 0.825), (0.375, 0.825), TEAL, 2.5)
arrow(ax, (0.625, 0.825), (0.70, 0.825), PURPLE, 2.5)

box(ax, 0.20, 0.625, 0.60, 0.075,
    'USE CONTRACT: coverage | exactness | evidence | authority | freshness | purpose | deadline',
    BLUE, LIGHT_BLUE, 13.2)
arrow(ax, (0.50, 0.76), (0.50, 0.70), PURPLE, 2.0)
arrow(ax, (0.825, 0.76), (0.73, 0.70), GRAY, 1.7, rad=0.08)

ax.text(0.5, 0.575, 'ELIGIBLE TASK-RELATIVE REPRESENTATIONS (NOT A LINEAR LADDER)',
        ha='center', va='center', fontsize=16.5, fontweight='bold', color=BLUE)
reps = [
    ('HANDLE /\nROUTING CAPSULE\ncheap; omission-heavy', BLUE, LIGHT_BLUE),
    ('STRUCTURED\nSYNTHESIS\nbroad; lossy', BLUE, LIGHT_BLUE),
    ('EVIDENCE\nBUNDLE\nclaim-source bindings', TEAL, LIGHT_TEAL),
    ('EXACT\nEXCERPTS\nnarrow; source-exact', PURPLE, LIGHT_PURPLE),
    ('RUNTIME\nMATERIALIZATION\ntokens / prefix / KV', ORANGE, LIGHT_ORANGE),
]
xs = [0.03, 0.225, 0.42, 0.615, 0.81]
for x, (txt, edge, face) in zip(xs, reps):
    box(ax, x, 0.405, 0.16, 0.12, txt, edge, face, 10.8)
arrow(ax, (0.50, 0.625), (0.50, 0.525), BLUE, 2.0)
ax.text(0.5, 0.365,
        'The compiler selects the least costly representation that satisfies the current use contract.',
        ha='center', va='center', fontsize=13.2, color=NAVY, fontweight='bold')

ax.text(0.5, 0.315, 'DYNAMIC PHYSICAL RESIDENCY', ha='center', va='center',
        fontsize=16.5, fontweight='bold', color=ORANGE)
locs = [
    ('SEALED /\nOBJECT STORE', GRAY, LIGHT_GRAY),
    ('COMPRESSED\nHOST CACHE', TEAL, LIGHT_TEAL),
    ('NON-MODEL-VISIBLE\nSTAGING', BLUE, LIGHT_BLUE),
    ('ACTIVE ROLE-LABELED\nTEXT', ORANGE, LIGHT_ORANGE),
    ('HBM / PREFIX /\nKV', PURPLE, LIGHT_PURPLE),
]
loc_xs = [0.035, 0.23, 0.425, 0.62, 0.815]
for x, (txt, edge, face) in zip(loc_xs, locs):
    box(ax, x, 0.15, 0.15, 0.10, txt, edge, face, 10.4)
for i in range(len(loc_xs) - 1):
    arrow(ax, (loc_xs[i] + 0.15, 0.20), (loc_xs[i + 1], 0.20), GRAY, 1.7, style='<|-|>')
arrow(ax, (0.11, 0.145), (0.50, 0.145), TEAL, 1.8, rad=-0.07)
ax.text(0.30, 0.115, 'direct address fault or plan-guided promotion',
        ha='center', fontsize=10.8, color=TEAL)
arrow(ax, (0.89, 0.145), (0.50, 0.145), GRAY, 1.6, dash=(0, (4, 3)), rad=0.07)
ax.text(0.70, 0.115, 'evict | demote | invalidate | recompute',
        ha='center', fontsize=10.8, color=GRAY)
ax.text(0.5, 0.055,
        'Logical identity remains stable while representations and physical locations change with task demand and policy.',
        ha='center', va='center', fontsize=13.3, color=NAVY, fontweight='bold')
fig.savefig(OUT / 'figure2_page_residency.png', dpi=190, bbox_inches='tight')
plt.close(fig)


# Figure 5 in the paper: prefetch timeline
fig, ax = setup('Planner-guided staging can hide semantic page-fault latency', (16, 9))
ax.text(0.17, 0.80, 'Reactive path', fontsize=18, fontweight='bold', color=RED)
for x, w, txt, face, edge in [
    (0.17, 0.15, 'plan', LIGHT_BLUE, BLUE),
    (0.32, 0.18, 'fault + I/O', LIGHT_RED, RED),
    (0.50, 0.16, 'resume', LIGHT_BLUE, BLUE),
    (0.66, 0.14, 'generate', LIGHT_TEAL, TEAL),
]:
    box(ax, x, 0.66, w, 0.10, txt, edge, face, 13.8, radius=0, pad=0)
arrow(ax, (0.17, 0.61), (0.83, 0.61), GRAY, 1.8)
ax.text(0.41, 0.565, 'blocking stall', ha='center', fontsize=14.5,
        fontweight='bold', color=RED)
ax.text(0.845, 0.61, 'time', va='center', fontsize=12.5, color=GRAY)

ax.text(0.17, 0.45, 'VCM predictive path', fontsize=18, fontweight='bold', color=TEAL)
for x, w, txt, face, edge in [
    (0.17, 0.15, 'plan DAG', LIGHT_BLUE, BLUE),
    (0.32, 0.26, 'reason / tool work', LIGHT_BLUE, BLUE),
    (0.58, 0.13, 'promote', LIGHT_TEAL, TEAL),
    (0.71, 0.13, 'generate', LIGHT_TEAL, TEAL),
]:
    box(ax, x, 0.32, w, 0.10, txt, edge, face, 13.7, radius=0, pad=0)
box(ax, 0.23, 0.145, 0.23, 0.10, 'async fetch +\nmaterialize', ORANGE, LIGHT_ORANGE, 12.8, radius=0, pad=0)
box(ax, 0.46, 0.145, 0.15, 0.10, 'non-model-visible\nstaging gate', PURPLE, LIGHT_PURPLE, 12.0, radius=0, pad=0)
arrow(ax, (0.32, 0.32), (0.32, 0.245), ORANGE, 2.0)
arrow(ax, (0.61, 0.245), (0.645, 0.32), PURPLE, 2.0)
arrow(ax, (0.17, 0.095), (0.87, 0.095), GRAY, 1.8)
ax.text(0.885, 0.095, 'time', va='center', fontsize=12.5, color=GRAY)
ax.text(0.5, 0.035,
        'Staging is outside model-visible generation state, but it still consumes resources and can leak access patterns; promotion requires relevance and governance checks.',
        ha='center', fontsize=12.4, color=NAVY, fontweight='bold', wrap=True)
fig.savefig(OUT / 'figure3_prefetch_timeline.png', dpi=190, bbox_inches='tight')
plt.close(fig)


# Figure 4 in paper: evidence lineage
fig, ax = setup('Evidence-carrying derivation preserves an auditable path to sources', (16.5, 10.5))
steps = [
    (0.035, 'IMMUTABLE\nEVENTS', 'turns | files | tools', GRAY, LIGHT_GRAY),
    (0.235, 'TYPED\nCELLS', 'claims | constraints\nroles | decisions', BLUE, LIGHT_BLUE),
    (0.435, 'SEMANTIC\nPAGE', 'deduplication\nconflicts | scope', TEAL, LIGHT_TEAL),
    (0.635, 'CERTIFIED\nDERIVATIVE', 'synthesis | evidence\nexact view', PURPLE, LIGHT_PURPLE),
    (0.835, 'COMPILED\nTASK VIEW', 'eligible working set', ORANGE, LIGHT_ORANGE),
]
for x, title, sub, edge, face in steps:
    box(ax, x, 0.72, 0.14, 0.13, title, edge, face, 13.5)
    ax.text(x + 0.07, 0.69, sub, ha='center', va='top', fontsize=10.4, color=GRAY)
for i in range(len(steps) - 1):
    arrow(ax, (steps[i][0] + 0.14, 0.785), (steps[i + 1][0], 0.785), GRAY, 2)
box(ax, 0.19, 0.36, 0.62, 0.25,
    'REPRESENTATION CERTIFICATE\n\nparent hashes + source spans/regions + atomic claims\nscope + validity + uncertainty + contradictions\ndeclared omissions + tested use families + authority ceiling\ntransformation/model version + verifier results + permitted uses',
    PURPLE, LIGHT_PURPLE, 14.5, align='left')
for x, *_ in steps:
    arrow(ax, (x + 0.07, 0.69), (0.50, 0.61), PURPLE, 1.2,
          dash=(0, (4, 3)), z=1)
box(ax, 0.055, 0.10, 0.24, 0.13,
    'AUDIT / DISPUTE\nresolve exact sources\nor expose missing fallback', GRAY, LIGHT_GRAY, 12.7)
box(ax, 0.38, 0.10, 0.24, 0.13,
    'UPDATE / INVALIDATION\nmark dependent views dirty\npurge incompatible caches', TEAL, LIGHT_TEAL, 12.7)
box(ax, 0.705, 0.10, 0.24, 0.13,
    'DELETION CLOSURE\nremove derived views, staging,\nreplicas, prefixes, and KV', ORANGE, LIGHT_ORANGE, 12.7)
arrow(ax, (0.32, 0.36), (0.18, 0.23), GRAY, 2)
arrow(ax, (0.50, 0.36), (0.50, 0.23), TEAL, 2)
arrow(ax, (0.68, 0.36), (0.82, 0.23), ORANGE, 2)
ax.text(0.5, 0.035,
        'A certificate establishes lineage and machine-checkable obligations; it does not prove that a natural-language source is true or increase its authority.',
        ha='center', fontsize=12.8, color=NAVY, fontweight='bold', wrap=True)
fig.savefig(OUT / 'figure4_proof_lineage.png', dpi=190, bbox_inches='tight')
plt.close(fig)


# Figure 3 in paper: protected compiler, spacious layout
fig, ax = setup('The context compiler protects mandatory state before optimizing optional pages', (17, 12))
ax.text(0.05, 0.885, '1  AUTHORITY-CHECKED PROTECTED MINIMUM SET', fontsize=16.5,
        fontweight='bold', color=ORANGE)
mandatory = [
    ('policy', PURPLE, LIGHT_PURPLE),
    ('goal + request', BLUE, LIGHT_BLUE),
    ('constraints', ORANGE, LIGHT_ORANGE),
    ('corrections', RED, LIGHT_RED),
    ('commitments', TEAL, LIGHT_TEAL),
    ('procedures', GRAY, LIGHT_GRAY),
    ('evidence\nobligations', TEAL, LIGHT_TEAL),
]
mx = [0.045, 0.18, 0.315, 0.45, 0.585, 0.72, 0.855]
for x, (txt, edge, face) in zip(mx, mandatory):
    box(ax, x, 0.785, 0.11, 0.075, txt, edge, face, 10.8)

box(ax, 0.26, 0.655, 0.46, 0.085,
    'MINIMUM-FIT GATE\ncapability | snapshot | authority | freshness\nuse contract | all resource budgets',
    PURPLE, LIGHT_PURPLE, 11.4)
for x in mx:
    arrow(ax, (x + 0.055, 0.785), (0.49, 0.74), PURPLE, 0.9, z=1)
box(ax, 0.76, 0.655, 0.19, 0.085,
    'UNSAFE-FIT\nchange model, budget, scope,\nor task decomposition', RED, LIGHT_RED, 11.5)
arrow(ax, (0.72, 0.697), (0.76, 0.697), RED, 2)
ax.text(0.855, 0.625, 'mandatory state is never silently dropped',
        ha='center', fontsize=10.5, color=RED, fontweight='bold')

ax.text(0.05, 0.565, '2  OPTIONAL CANDIDATE CHANNELS', fontsize=16.5,
        fontweight='bold', color=BLUE)
optional = [
    ('semantic', BLUE, LIGHT_BLUE),
    ('decisions', TEAL, LIGHT_TEAL),
    ('rejections', RED, LIGHT_RED),
    ('dependencies', PURPLE, LIGHT_PURPLE),
    ('freshness', TEAL, LIGHT_TEAL),
    ('predicted use', ORANGE, LIGHT_ORANGE),
]
ox = [0.065, 0.215, 0.365, 0.515, 0.665, 0.815]
for x, (txt, edge, face) in zip(ox, optional):
    box(ax, x, 0.475, 0.12, 0.065, txt, edge, face, 10.8)
box(ax, 0.25, 0.345, 0.50, 0.085,
    'DISCRETIONARY OPTIMIZER\nmarginal utility - tokens - latency - interference - exposure - drift',
    BLUE, LIGHT_BLUE, 12.5)
for x in ox:
    arrow(ax, (x + 0.06, 0.475), (0.50, 0.43), BLUE, 0.9, z=1)

ax.text(0.05, 0.285, '3  COMPILED WORKING CONTEXT', fontsize=16.5,
        fontweight='bold', color=ORANGE)
lanes = [
    ('1  System policy and current request - reserved', PURPLE, LIGHT_PURPLE),
    ('2  Coherent task snapshot and plan frontier', BLUE, LIGHT_BLUE),
    ('3  Constraints, corrections, commitments, and procedures', ORANGE, LIGHT_ORANGE),
    ('4  Evidence and exactness required for imminent claims or actions', TEAL, LIGHT_TEAL),
    ('5  Optional background selected under the remaining multi-resource budget', GRAY, LIGHT_GRAY),
]
y = 0.235
for txt, edge, face in lanes:
    box(ax, 0.18, y, 0.64, 0.038, txt, edge, face, 10.7,
        radius=0.006, pad=0.003, align='left')
    y -= 0.044
arrow(ax, (0.50, 0.345), (0.50, 0.285), BLUE, 2)
ax.text(0.5, 0.012,
        'Prompt labels help interpretation, but capabilities, tool authorization, sealed-data access, and authority ceilings are enforced outside the model.',
        ha='center', fontsize=11.2, color=NAVY, fontweight='bold', wrap=True)
fig.savefig(OUT / 'figure5_compiler_lanes.png', dpi=190, bbox_inches='tight')
plt.close(fig)

print('Wrote revised figures to', OUT)
