import json, os
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict

ROOT = r"D:\aryan\Projects\Multi agent LLM System"
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

trials = [json.loads(l) for l in open(os.path.join(ROOT, "dataset", "trials.jsonl"), encoding="utf-8")]
topos = {t["topology_id"]: t for t in json.load(open(os.path.join(ROOT, "dataset", "topology_pool.json"), encoding="utf-8"))}
metrics = [json.loads(l) for l in open(os.path.join(ROOT, "model", "metrics_history.jsonl"), encoding="utf-8")]

# ---------- Chart 1: fault-type distribution ----------
label_order = ["clean", "noise", "contamination", "ceiling"]
counts = Counter(t["true_label"] for t in trials)
vals = [counts[l] for l in label_order]
colors = ["#8a8a8a", "#4C72B0", "#DD8452", "#C44E52"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(label_order, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 10, str(v), ha="center", fontsize=10)
ax.set_ylabel("Number of trials")
ax.set_title("Dataset composition by fault type (n=%d)" % len(trials))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_fault_type_distribution.png"))
plt.close(fig)

# ---------- Chart 2: train vs OOD split ----------
split_counts = Counter(topos[t["topology_id"]]["split"] for t in trials)
labels = ["Train topologies\n(14 shapes)", "OOD test topologies\n(6 unseen shapes)"]
vals2 = [split_counts["train"], split_counts["ood_test"]]
fig, ax = plt.subplots(figsize=(5.5, 4))
bars = ax.bar(labels, vals2, color=["#4C72B0", "#C44E52"])
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width() / 2, v + 10, str(v), ha="center", fontsize=10)
ax.set_ylabel("Number of trials")
ax.set_title("Train vs. out-of-distribution trial split")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_train_ood_split.png"))
plt.close(fig)

# ---------- Chart 3: topology size distribution ----------
node_counts = Counter(len(t["nodes"]) for t in topos.values())
sizes = sorted(node_counts)
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar([str(s) for s in sizes], [node_counts[s] for s in sizes], color="#55A868")
ax.set_xlabel("Nodes per topology")
ax.set_ylabel("Number of topologies")
ax.set_title("Topology pool: graph-size distribution (20 topologies)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_topology_size_distribution.png"))
plt.close(fig)

# ---------- Chart 4: OOD top-1 accuracy progression across experiment log ----------
# Restrict to the 1762-record dataset runs (drop the very first, superseded
# 493-record baseline run) so accuracy numbers are on a consistent OOD set (n=574).
metrics_sorted_all = sorted(metrics, key=lambda m: m["timestamp"])
metrics_sorted = [m for m in metrics_sorted_all if m["metrics"]["ood"]["n"] == 574]
xs = list(range(1, len(metrics_sorted) + 1))
ood_top1 = [m["metrics"]["ood"]["top1_accuracy"] for m in metrics_sorted]
baseline = metrics_sorted[0]["metrics"]["baseline_ood_top1_accuracy"]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(xs, ood_top1, marker="o", ms=4, color="#4C72B0", label="GNN OOD top-1 accuracy (each experiment run)")
ax.axhline(baseline, color="#C44E52", linestyle="--", label="Naive max-uncertainty baseline (%.3f)" % baseline)
best_idx = ood_top1.index(max(ood_top1))
ax.scatter([xs[best_idx]], [ood_top1[best_idx]], color="gold", edgecolor="black", zorder=5, s=90,
           label="Best run so far (%.3f)" % ood_top1[best_idx])
ax.set_xlabel("Experiment run (chronological)")
ax.set_ylabel("OOD top-1 accuracy")
ax.set_title("Out-of-distribution accuracy across the experiment log")
ax.legend(fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_ood_accuracy_progression.png"))
plt.close(fig)

# ---------- Chart 5: architecture comparison (GAT vs GCN vs SAGE), matched config ----------
arch_runs = {m["config"].get("conv_type", "gat"): m for m in metrics_sorted if m["config"].get("arch_sweep")}
archs = ["gat", "gcn", "sage"]
arch_vals = [arch_runs[a]["metrics"]["ood"]["top1_accuracy"] for a in archs]
fig, ax = plt.subplots(figsize=(5.5, 4))
bars = ax.bar([a.upper() for a in archs], arch_vals, color=["#4C72B0", "#DD8452", "#55A868"])
for b, v in zip(bars, arch_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, "%.3f" % v, ha="center", fontsize=10)
ax.set_ylabel("OOD top-1 accuracy")
ax.set_title("GNN layer type comparison\n(matched hyperparameters)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_architecture_comparison.png"))
plt.close(fig)

# ---------- Chart 6: GNN vs baselines (final comparison) ----------
rf_run = [m for m in metrics_sorted_all if m["config"].get("model_type") == "RandomForest"][0]
best_gnn = max(metrics_sorted, key=lambda m: m["metrics"]["ood"]["top1_accuracy"])
k10_preview_runs = [m for m in metrics_sorted_all if m["config"].get("preview_partial_k10")]
best_k10_preview = max(k10_preview_runs, key=lambda m: m["metrics"]["ood"]["top1_accuracy"]) if k10_preview_runs else None

names = ["Naive max-\nuncertainty", "Non-graph\nRandomForest", "Best GNN\n(k=5, finished\ndataset, n=1762)"]
comp_vals = [baseline, rf_run["metrics"]["ood"]["top1_accuracy"], best_gnn["metrics"]["ood"]["top1_accuracy"]]
comp_colors = ["#8a8a8a", "#DD8452", "#4C72B0"]
if best_k10_preview is not None:
    names.append("Same GNN, k=10\nPREVIEW (partial\ndataset, n=%d)" % best_k10_preview["config"]["dataset_size"])
    comp_vals.append(best_k10_preview["metrics"]["ood"]["top1_accuracy"])
    comp_colors.append("#55A868")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
bars = ax.bar(names, comp_vals, color=comp_colors)
for b, v in zip(bars, comp_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.007, "%.3f" % v, ha="center", fontsize=10)
ax.set_ylabel("OOD top-1 accuracy")
ax.set_title("Fault-localization approach comparison (OOD test set)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_model_comparison.png"))
plt.close(fig)

# ---------- Chart 7: best model — full metric breakdown across splits ----------
metric_names = ["top1_accuracy", "top2_accuracy", "precision", "recall", "f1"]
metric_labels = ["Top-1 Acc", "Top-2 Acc", "Precision", "Recall", "F1"]
splits = ["train", "val", "ood"]
split_labels = ["Train", "Validation", "OOD (held-out)"]
split_colors = ["#55A868", "#DD8452", "#4C72B0"]

import numpy as np
x = np.arange(len(metric_names))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
for i, (sp, lbl, col) in enumerate(zip(splits, split_labels, split_colors)):
    vals = [best_gnn["metrics"][sp][m] for m in metric_names]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=lbl, color=col)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("Best model (SAGE + attention pooling + 3-seed ensemble) — full metric breakdown by split")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_best_model_metrics.png"))
plt.close(fig)

# ---------- Chart 8: accuracy evolution — key milestones from start to now ----------
# Hand-picked milestones from the real experiment log (metrics_history.jsonl),
# each one a documented, individually-verified change (see docs/handoff notes).
milestone_specs = [
    ("Initial GNN\n(493-rec dataset)", lambda m: m["metrics"]["ood"]["n"] == 164),
    ("Dataset expanded\n(1762 records)", lambda m: m["metrics"]["ood"]["n"] == 574 and not m["config"].get("bidirectional_edges")),
    ("+ Bidirectional\nedges", lambda m: m["config"].get("bidirectional_edges") and not m["config"].get("use_embeddings")),
    ("+ Raw embeddings\n(overfit, no reg.)", lambda m: m["config"].get("use_embeddings") and not m["config"].get("pca_components") and not m["config"].get("input_projection")),
    ("+ PCA-32 +\nregularization", lambda m: m["config"].get("pca_components") == 32 and m["config"].get("input_projection") and not m["config"].get("conv_type")),
    ("+ Hyperparam\nsweep (hd=64)", lambda m: m["config"].get("hidden_dim") == 64 and not m["config"].get("conv_type") and m["config"].get("sweep")),
    ("+ GraphSAGE\nconvolution", lambda m: m["config"].get("conv_type") == "sage" and m["config"].get("pca_components") == 32 and not m["config"].get("use_embedding_variance")),
    ("+ Embedding\nvariance feature", lambda m: m["config"].get("use_embedding_variance") and m["config"].get("pca_components") == 128 and not m["config"].get("pool_type")),
    ("+ Attention\npooling", lambda m: m["config"].get("pool_type") == "attention" and not m["config"].get("ensemble")),
    ("+ Ensemble\n(k=5, finished\ndataset)", lambda m: m["config"].get("pool_type") == "attention" and m["config"].get("ensemble") and not m["config"].get("preview_partial_k10")),
]

milestone_labels, milestone_vals = [], []
for label, pred in milestone_specs:
    matches = [m for m in metrics_sorted_all if pred(m)]
    if not matches:
        continue
    run = sorted(matches, key=lambda m: m["timestamp"])[0]
    milestone_labels.append(label)
    milestone_vals.append(run["metrics"]["ood"]["top1_accuracy"])

fig, ax = plt.subplots(figsize=(12, 5))
xs_m = list(range(len(milestone_labels)))
ax.plot(xs_m, milestone_vals, marker="o", ms=9, lw=2.5, color="#4C72B0", label="k=5 dataset (n=1762, finished)")
for xi, v in zip(xs_m, milestone_vals):
    ax.text(xi, v + 0.012, "%.3f" % v, ha="center", fontsize=9, fontweight="bold")

# k=10 preview point(s) -- same architecture, a DIFFERENT (partial, still-growing)
# dataset, so plotted as a separate dashed segment, not folded into the k=5 trend.
if k10_preview_runs is not None and len(k10_preview_runs) > 0:
    k10_sorted = sorted(k10_preview_runs, key=lambda m: m["timestamp"])
    last_k5_x = xs_m[-1]
    k10_x = [last_k5_x + 1 + i for i in range(len(k10_sorted))]
    k10_vals = [m["metrics"]["ood"]["top1_accuracy"] for m in k10_sorted]
    k10_labels = ["k=10 PREVIEW\n(partial, n=%d)" % m["config"]["dataset_size"] for m in k10_sorted]
    ax.plot([last_k5_x] + k10_x, [milestone_vals[-1]] + k10_vals, marker="D", ms=9, lw=2, ls="--",
            color="#55A868", label="k=10 dataset (partial, in progress)")
    for xi, v in zip(k10_x, k10_vals):
        ax.text(xi, v + 0.012, "%.3f" % v, ha="center", fontsize=9, fontweight="bold", color="#2d6a3e")
    milestone_labels = milestone_labels + k10_labels
    xs_m = xs_m + k10_x

ax.axhline(baseline, color="#C44E52", linestyle="--", label="Naive max-uncertainty baseline (%.3f)" % baseline)
ax.set_xticks(xs_m)
ax.set_xticklabels(milestone_labels, fontsize=8)
ax.set_ylabel("OOD top-1 accuracy")
ax.set_title("Evolution of OOD fault-localization accuracy, start to now")
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.margins(x=0.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_accuracy_evolution.png"))
plt.close(fig)

print("done")
print("baseline", baseline, "best_gnn", best_gnn["metrics"]["ood"]["top1_accuracy"], best_gnn["config"])
print("rf", rf_run["metrics"]["ood"]["top1_accuracy"])
print("milestones:", list(zip(milestone_labels, milestone_vals)))
