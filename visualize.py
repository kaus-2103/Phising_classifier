"""
Visualizes what the from-scratch logistic regression model learned.

This is a separate, optional script -- it uses matplotlib ONLY for
drawing the picture. The classifier itself (phishing_classifier_scratch.py)
still has zero ML/plotting libraries in its training logic.

Produces: decision_boundary.png
  - Blue dots  = Safe emails
  - Red dots   = Phishing emails
  - Shaded background = what the model would predict for that region
  - Solid line = the decision boundary (where probability = 0.5)
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from p_classifier import dataset, train, predict_probability

# ---------------------------------------------------------------
# 1. Train the model (reuses the exact same from-scratch code)
# ---------------------------------------------------------------
w1, w2, b = train(dataset, learning_rate=0.1, epochs=2000)

# ---------------------------------------------------------------
# 2. Split dataset into the two classes for plotting
#
# NOTE: several safe emails land on the exact same (x1, x2) point
# (e.g. 4 different emails all had 0 suspicious words + 0 links),
# so their dots would otherwise stack invisibly on top of each
# other. We count duplicates per point and label overlapping dots
# with "xN" so no email silently disappears from the picture.
# ---------------------------------------------------------------
safe_counts = Counter((x1, x2) for x1, x2, label in dataset if label == 0)
phishing_counts = Counter((x1, x2) for x1, x2, label in dataset if label == 1)

safe_x = [pt[0] for pt in safe_counts]
safe_y = [pt[1] for pt in safe_counts]
phishing_x = [pt[0] for pt in phishing_counts]
phishing_y = [pt[1] for pt in phishing_counts]

# ---------------------------------------------------------------
# 3. Build a grid of points covering the feature space, and ask
#    the model for a prediction at every point -> this gives us
#    the shaded "decision regions" in the background
# ---------------------------------------------------------------
x_min, x_max = -1, 19
y_min, y_max = -1, 6
grid_step = 0.1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, grid_step),
    np.arange(y_min, y_max, grid_step),
)

zz = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i, j] = predict_probability(xx[i, j], yy[i, j], w1, w2, b)

# ---------------------------------------------------------------
# 4. Plot everything
# ---------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Background shading: probability of being phishing (0 -> blue, 1 -> red)
plt.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.35, vmin=0, vmax=1)

# Decision boundary: the line where probability == 0.5
plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2, linestyles="--")

# Data points (marker size scales with how many emails share that
# exact point, so overlapping emails are still visible as bigger dots)
safe_sizes = [120 + 60 * (safe_counts[(x, y)] - 1) for x, y in zip(safe_x, safe_y)]
phishing_sizes = [120 + 60 * (phishing_counts[(x, y)] - 1) for x, y in zip(phishing_x, phishing_y)]

plt.scatter(safe_x, safe_y, c="blue", edgecolors="black", s=safe_sizes,
            label="Safe email (0)", zorder=3)
plt.scatter(phishing_x, phishing_y, c="red", edgecolors="black", s=phishing_sizes,
            marker="^", label="Phishing email (1)", zorder=3)

# Label any point where more than one email landed on the same spot
for (x, y), count in safe_counts.items():
    if count > 1:
        plt.annotate(f"x{count}", (x, y), textcoords="offset points",
                     xytext=(10, 8), fontsize=10, fontweight="bold", color="darkblue")

for (x, y), count in phishing_counts.items():
    if count > 1:
        plt.annotate(f"x{count}", (x, y), textcoords="offset points",
                     xytext=(10, 8), fontsize=10, fontweight="bold", color="darkred")

plt.xlabel("Number of suspicious words")
plt.ylabel("Number of links")
plt.title("Phishing Email Classifier — Learned Decision Boundary\n"
          f"z = {w1:.2f}·x1 + {w2:.2f}·x2 + ({b:.2f})")
plt.legend(loc="upper left")
plt.colorbar(label="Predicted probability of Phishing")
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
print("\nSaved plot to decision_boundary.png")
