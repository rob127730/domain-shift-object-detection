"""Retention niveau 5 / niveau 1 par modele et par perturbation (a partir du CSV de resultats)."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv = sys.argv[1] if len(sys.argv) > 1 else "results/weather_metrics_all_models.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "docs/retention_level5_over_level1.png"
df = pd.read_csv(csv)
order = ["fog", "glare", "night", "rain", "snow"]
models = ["YOLOv8n", "Faster R-CNN", "DETR", "RT-DETR"]
colors = dict(zip(models, ["#2563eb", "#16a34a", "#d97706", "#dc2626"]))


def retention(metric):
    p = df.pivot_table(index=["model", "condition"], columns="level", values=metric)
    return (p[5] / p[1]).unstack(0).reindex(order)


fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for a, (metric, title) in zip(ax, [("map50", "mAP50 (definition propre a chaque modele)"), ("f1", "F1")]):
    r = retention(metric)
    x = np.arange(len(order))
    for i, m in enumerate(models):
        a.bar(x + i * 0.2 - 0.3, r[m], 0.2, label=m, color=colors[m])
    a.set_xticks(x)
    a.set_xticklabels(["Brouillard", "Eblouissement", "Nuit", "Pluie", "Neige"])
    a.set_title("Retention niveau 5 / niveau 1 - " + title, fontsize=9)
    a.axhline(1, color="k", lw=0.6)
    a.grid(axis="y", alpha=0.3)
ax[0].set_ylabel("Performance conservee au niveau 5")
ax[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=150)
print(retention("map50").round(3))
