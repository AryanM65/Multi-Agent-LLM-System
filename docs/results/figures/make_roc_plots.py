import json, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

node_recs = json.load(open(os.path.join(OUT, "ood_node_predictions.json"), encoding="utf-8"))
graph_recs = json.load(open(os.path.join(OUT, "ood_graph_predictions.json"), encoding="utf-8"))

y_true = np.array([r["is_true_fault_node"] for r in node_recs])
y_score = np.array([r["prob_is_fault"] for r in node_recs])

# ---------- Fig 9: ROC curve (pooled per-node binary "is this the fault node") ----------
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, color="#4C72B0", lw=2.2, label=f"GNN (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#8a8a8a", lw=1.2, linestyle="--", label="Random guess (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — node-level fault identification\n(pooled over all OOD graphs, n=%d node instances)" % len(y_true))
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_roc_curve.png"))
plt.close(fig)

# ---------- Fig 10: Precision-Recall curve (same pooled binary task) ----------
precision, recall, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)
base_rate = y_true.mean()

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(recall, precision, color="#DD8452", lw=2.2, label=f"GNN (AP = {ap:.3f})")
ax.axhline(base_rate, color="#8a8a8a", lw=1.2, linestyle="--", label=f"Random guess (AP = {base_rate:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve — node-level fault identification\n(pooled over all OOD graphs)")
ax.legend(loc="upper right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_pr_curve.png"))
plt.close(fig)

# ---------- Fig 11: ROC curve broken down by node role ----------
fig, ax = plt.subplots(figsize=(6, 5.5))
colors = {"retriever": "#4C72B0", "reasoner": "#DD8452", "writer": "#55A868", "no_fault": "#8a8a8a"}
for role in ["retriever", "reasoner", "writer", "no_fault"]:
    mask = np.array([r["role"] == role for r in node_recs])
    if mask.sum() == 0 or y_true[mask].sum() == 0:
        continue
    fpr_r, tpr_r, _ = roc_curve(y_true[mask], y_score[mask])
    auc_r = auc(fpr_r, tpr_r)
    ax.plot(fpr_r, tpr_r, color=colors[role], lw=2, label=f"{role} (AUC={auc_r:.3f}, n={mask.sum()})")
ax.plot([0, 1], [0, 1], color="#c0c0c0", lw=1, linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve by node role (OOD set)")
ax.legend(loc="lower right", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig11_roc_by_role.png"))
plt.close(fig)

# ---------- Fig 12: role-level confusion matrix (true role vs predicted role) ----------
roles = ["retriever", "reasoner", "writer", "no_fault"]
true_roles = [r["true_role"] for r in graph_recs]
pred_roles = [r["pred_role"] for r in graph_recs]
cm = confusion_matrix(true_roles, pred_roles, labels=roles)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(roles)))
ax.set_yticks(range(len(roles)))
ax.set_xticklabels(roles, rotation=30, ha="right")
ax.set_yticklabels(roles)
ax.set_xlabel("Predicted role")
ax.set_ylabel("True role")
ax.set_title("Role-level confusion matrix (OOD set)\n(row-normalized; raw counts in cells)")
for i in range(len(roles)):
    for j in range(len(roles)):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center", va="center", color=color, fontsize=9)
fig.colorbar(im, ax=ax, label="Row-normalized fraction")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig12_role_confusion_matrix.png"))
plt.close(fig)

# ---------- Fig 13: accuracy by true fault type (bar chart) ----------
from collections import defaultdict
by_type = defaultdict(lambda: [0, 0])
for r in graph_recs:
    by_type[r["true_label"]][1] += 1
    if r["correct"]:
        by_type[r["true_label"]][0] += 1

order = ["clean", "noise", "contamination", "ceiling"]
accs = [by_type[t][0] / by_type[t][1] if by_type[t][1] else 0 for t in order]
ns = [by_type[t][1] for t in order]

fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(order, accs, color=["#8a8a8a", "#4C72B0", "#DD8452", "#C44E52"])
for b, v, n in zip(bars, accs, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}\n(n={n})", ha="center", fontsize=8.5)
ax.set_ylabel("Top-1 localization accuracy")
ax.set_title("OOD accuracy by true fault type")
ax.set_ylim(0, max(accs) + 0.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig13_accuracy_by_fault_type.png"))
plt.close(fig)

print("ROC AUC (pooled):", roc_auc)
print("Average precision (pooled):", ap, "base rate:", base_rate)
print("confusion matrix (rows=true, cols=pred):", roles)
print(cm)
print("accuracy by fault type:", dict(zip(order, accs)), "counts:", dict(zip(order, ns)))
