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

import matplotlib.pyplot as plt
import numpy as np

from p_classifier import dataset, train, predict_probability

# ---------------------------------------------------------------
# 1. Train the model (reuses the exact same from-scratch code)
# ---------------------------------------------------------------
w1, w2, b = train(dataset, learning_rate=0.1, epochs=2000)

# ---------------------------------------------------------------
# 2. Split dataset into the two classes for plotting
# ---------------------------------------------------------------
safe_x, safe_y = [], []
phishing_x, phishing_y = [], []

for x1, x2, label in dataset:
    if label == 0:
        safe_x.append(x1)
        safe_y.append(x2)
    else:
        phishing_x.append(x1)
        phishing_y.append(x2)

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

# Data points
plt.scatter(safe_x, safe_y, c="blue", edgecolors="black", s=120,
            label="Safe email (0)", zorder=3)
plt.scatter(phishing_x, phishing_y, c="red", edgecolors="black", s=120,
            marker="^", label="Phishing email (1)", zorder=3)

plt.xlabel("Number of suspicious words")
plt.ylabel("Number of links")
plt.title("Phishing Email Classifier — Learned Decision Boundary\n"
          f"z = {w1:.2f}·x1 + {w2:.2f}·x2 + ({b:.2f})")
plt.legend(loc="upper left")
plt.colorbar(label="Predicted probability of Phishing")
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
print("\nSaved plot to decision_boundary.png")
