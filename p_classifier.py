"""
Phishing Email Classifier - Built From Scratch (No ML Libraries)
===================================================================

GOAL
----
Predict whether an email is "Phishing" (1) or "Safe" (0) using only
2 simple features, with NO scikit-learn, NO numpy, NO pandas.
Everything (math, training loop, prediction) is written in plain
Python so it's easy to explain line-by-line.

FEATURES USED (kept to 2 so it's easy to reason about / plot)
---------------------------------------------------------------
1. num_suspicious_words -> count of words like "urgent", "verify",
   "password", "click", "bank", "winner" in the email
2. num_links            -> number of links in the email

LABEL
-----
0 = Safe email
1 = Phishing email

THE MODEL: Logistic Regression
-------------------------------
Logistic Regression is the simplest classifier to explain:

    z = w1*x1 + w2*x2 + b
    prediction_probability = sigmoid(z) = 1 / (1 + e^-z)

If probability >= 0.5 -> predict class 1 (Phishing)
If probability <  0.5 -> predict class 0 (Safe)

We "learn" w1, w2, b using Gradient Descent: start with random-ish
weights, measure how wrong we are (loss), and nudge the weights in
the direction that reduces the error. Repeat many times.

This is exactly what scikit-learn's LogisticRegression does
internally -- we're just doing it by hand so every step is visible.
"""

import math
import random

# ---------------------------------------------------------------
# STEP 1: The dataset
# ---------------------------------------------------------------
# The dataset is NOT hand-typed here anymore. It comes from
# preprocess.py, which reads raw dummy emails (emails.py) and turns
# them into numbers. See preprocess.py to trace the full pipeline:
#   emails.py (raw text) -> preprocess.py (feature extraction) -> dataset
#
# Each row: [num_suspicious_words, num_links, label]
# label: 1 = Phishing, 0 = Safe
from preprocess import dataset

random.seed(42)  # reproducible results


# ---------------------------------------------------------------
# STEP 2: The math building blocks
# ---------------------------------------------------------------
def sigmoid(z):
    """Squashes any number into a probability between 0 and 1."""
    # guard against overflow for very negative z
    if z < -700:
        return 0.0
    return 1 / (1 + math.exp(-z))


def predict_probability(x1, x2, w1, w2, b):
    """Runs one email's features through the model."""
    z = (w1 * x1) + (w2 * x2) + b
    return sigmoid(z)


# ---------------------------------------------------------------
# STEP 3: Training with Gradient Descent (the "learning" part)
# ---------------------------------------------------------------
def train(data, learning_rate=0.1, epochs=30000):
    # Start with small random weights
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    b = random.uniform(-1, 1)

    n = len(data)

    for epoch in range(epochs):
        dw1 = 0.0
        dw2 = 0.0
        db = 0.0
        total_loss = 0.0

        for x1, x2, y in data:
            y_pred = predict_probability(x1, x2, w1, w2, b)

            # error = how far off the prediction was
            error = y_pred - y

            # gradients: how much each weight contributed to the error
            dw1 += error * x1
            dw2 += error * x2
            db += error

            # binary cross-entropy loss (clipped to avoid log(0))
            y_pred_clipped = min(max(y_pred, 1e-10), 1 - 1e-10)
            total_loss += -(y * math.log(y_pred_clipped) +
                             (1 - y) * math.log(1 - y_pred_clipped))

        # average the gradients over all examples
        dw1 /= n
        dw2 /= n
        db /= n

        # nudge the weights a little bit in the direction that reduces error
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * db

        if epoch % 400 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {total_loss / n:.4f} | "
                  f"w1={w1:.3f} w2={w2:.3f} b={b:.3f}")

    return w1, w2, b


# ---------------------------------------------------------------
# STEP 4: Turn probability into a final class (0 or 1)
# ---------------------------------------------------------------
def classify(x1, x2, w1, w2, b, threshold=0.5):
    prob = predict_probability(x1, x2, w1, w2, b)
    label = 1 if prob >= threshold else 0
    return label, prob


# ---------------------------------------------------------------
# STEP 5: Measure accuracy on the same data (simple self-check)
# ---------------------------------------------------------------
def evaluate(data, w1, w2, b):
    correct = 0
    for x1, x2, y in data:
        pred, _ = classify(x1, x2, w1, w2, b)
        if pred == y:
            correct += 1
    accuracy = correct / len(data) * 100
    return accuracy


# ---------------------------------------------------------------
# MAIN: run training, then test on new/unseen emails
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Training logistic regression from scratch...\n")
    w1, w2, b = train(dataset)

    print("\nFinal learned weights:")
    print(f"  w1 (suspicious words weight) = {w1:.3f}")
    print(f"  w2 (links weight)            = {w2:.3f}")
    print(f"  b  (bias)                    = {b:.3f}")

    acc = evaluate(dataset, w1, w2, b)
    print(f"\nTraining accuracy: {acc:.1f}%")

    # Try it on brand-new emails the model has never seen
    test_emails = [
        (5, 3, "Suspicious-looking email"),
        (0, 1, "Normal work email"),
        (2, 2, "Borderline case"),
        (6, 6, "Very spammy email"),
    ]

    print("\nPredictions on new emails:")
    for x1, x2, desc in test_emails:
        label, prob = classify(x1, x2, w1, w2, b)
        verdict = "PHISHING" if label == 1 else "SAFE"
        print(f"  {desc:28s} -> {verdict:8s} (confidence: {prob:.2f})")
