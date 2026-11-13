import json, os
import matplotlib.pyplot as plt

ROOT = r"D:\aryan\Projects\Multi agent LLM System"
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

metrics = [json.loads(l) for l in open(os.path.join(ROOT, "model", "metrics_history.jsonl"), encoding="utf-8")]
metrics_sorted_all = sorted(metrics, key=lambda m: m["timestamp"])
baseline = [m for m in metrics_sorted_all if m["metrics"]["ood"]["n"] == 574][0]["metrics"]["baseline_ood_top1_accuracy"]

# Same milestone list as fig8, but stops at "+ Attention pooling" (0.387) --
# excludes the 3-seed ensemble (0.395) and the k=10 preview points.
milestone_specs = [
    ("Initial GNN\n(493-rec dataset)", lambda m: m["metrics"]["ood"]["n"] == 164),
    ("Dataset expanded\n(1762 records)", lambda m: m["metrics"]["ood"]["n"] == 574 and not m["config"].get("bidirectional_edges")),
    ("+ Bidirectional\nedges", lambda m: m["config"].get("bidirectional_edges") and not m["config"].get("use_embeddings")),
    ("+ Raw embeddings\n(overfit, no reg.)", lambda m: m["config"].get("use_embeddings") and not m["config"].get("pca_components") and not m["config"].get("input_projection")),
    ("+ PCA-32 +\nregularization", lambda m: m["config"].get("pca_components") == 32 and m["config"].get("input_projection") and not m["config"].get("conv_type")),
    ("+ Hyperparam\nsweep (hd=64)", lambda m: m["config"].get("hidden_dim") == 64 and not m["config"].get("conv_type") and m["config"].get("sweep")),
    ("+ GraphSAGE\nconvolution", lambda m: m["config"].get("conv_type") == "sage" and m["config"].get("pca_components") == 32 and not m["config"].get("use_embedding_variance")),
    ("+ Embedding\nvariance feature", lambda m: m["config"].get("use_embedding_variance") and m["config"].get("pca_components") == 128 and not m["config"].get("pool_type")),
    ("+ Attention\npooling\n(stopping point)", lambda m: m["config"].get("pool_type") == "attention" and not m["config"].get("ensemble")),
]

milestone_labels, milestone_vals = [], []
for label, pred in milestone_specs:
    matches = [m for m in metrics_sorted_all if pred(m)]
    if not matches:
        continue
    run = sorted(matches, key=lambda m: m["timestamp"])[0]
    milestone_labels.append(label)
    milestone_vals.append(run["metrics"]["ood"]["top1_accuracy"])

fig, ax = plt.subplots(figsize=(11, 5))
xs_m = list(range(len(milestone_labels)))
ax.plot(xs_m, milestone_vals, marker="o", ms=9, lw=2.5, color="#4C72B0")
for xi, v in zip(xs_m, milestone_vals):
    ax.text(xi, v + 0.012, "%.3f" % v, ha="center", fontsize=9, fontweight="bold")
# highlight the final point (attention pooling, 0.387)
ax.scatter([xs_m[-1]], [milestone_vals[-1]], color="gold", edgecolor="black", zorder=5, s=110)

ax.axhline(baseline, color="#C44E52", linestyle="--", label="Naive max-uncertainty baseline (%.3f)" % baseline)
ax.set_xticks(xs_m)
ax.set_xticklabels(milestone_labels, fontsize=8.5)
ax.set_ylabel("OOD top-1 accuracy")
ax.set_title("Evolution of OOD accuracy through attention pooling (k=5 dataset)")
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.margins(x=0.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8b_evolution_to_attention_pooling.png"))
plt.close(fig)

print("milestones:", list(zip(milestone_labels, milestone_vals)))
