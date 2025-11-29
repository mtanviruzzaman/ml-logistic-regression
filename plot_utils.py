import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# # Load dataset and use 2 features
# X, y = load_breast_cancer(return_X_y=True)
# X2 = X[:, :2]

def add_intercept(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def sigmoid(z):
    # stable sigmoid to avoid overflow in exp
    out = np.empty_like(z)
    positive = z >= 0
    negative = ~positive

    out[positive] = 1 / (1 + np.exp(-z[positive]))
    
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1 + exp_z)

    return out

def loss(w, X, y):
    h = sigmoid(X @ w)
    return -(y*np.log(h+1e-12) + (1-y)*np.log(1-h+1e-12)).mean()

def gradient(w, X, y):
    return X.T @ (sigmoid(X @ w) - y) / len(y)

# Compute GD path: only store arrows every N iterations
def gd_path(Xb, y, w_start, lr, total_iters, arrow_every):
    w = w_start.copy()
    arrows = [w.copy()]
    for i in range(total_iters):
        w -= lr * gradient(w, Xb, y)
        if (i + 1) % arrow_every == 0:
            arrows.append(w.copy())
    return np.array(arrows)

# Make contour grid
def contour_data(Xb, y, w_center):
    w1_vals = np.linspace(w_center[1] - 8, w_center[1] + 8, 250)
    w2_vals = np.linspace(w_center[2] - 8, w_center[2] + 8, 250)
    Z = np.zeros((len(w1_vals), len(w2_vals)))
    for i, w1 in enumerate(w1_vals):
        for j, w2 in enumerate(w2_vals):
            Z[i, j] = loss(np.array([0, w1, w2]), Xb, y)
    return w1_vals, w2_vals, Z

def plot_gd_progress(X, y):
    # Raw (unscaled) version
    Xb_raw = add_intercept(X)
    w_start_raw = np.array([0, 15.0, -15.0])
    path_raw = gd_path(Xb_raw, y, w_start_raw, lr=0.002, total_iters=1000, arrow_every=100)
    w1_raw, w2_raw, Z_raw = contour_data(Xb_raw, y, w_start_raw)

    # Scaled version
    scaler = StandardScaler()
    X2_scaled = scaler.fit_transform(X)
    Xb_scaled = add_intercept(X2_scaled)
    w_start_scaled = np.array([0, 10.0, -10.0])
    path_scaled = gd_path(Xb_scaled, y, w_start_scaled, lr=0.05, total_iters=1000, arrow_every=100)
    w1_scaled, w2_scaled, Z_scaled = contour_data(Xb_scaled, y, w_start_scaled)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Without normalization
    ax = axes[0]
    ax.contour(w1_raw, w2_raw, Z_raw.T, levels=50, cmap='viridis')
    for i in range(len(path_raw) - 1):
        wA = path_raw[i]
        wB = path_raw[i + 1]
        ax.arrow(wA[1], wA[2], wB[1] - wA[1], wB[2] - wA[2],
                head_width=0.3, head_length=0.3, fc='orange', ec='orange')
    ax.set_title("Without normalization")
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.grid(True)

    # With normalization
    ax = axes[1]
    ax.contour(w1_scaled, w2_scaled, Z_scaled.T, levels=50, cmap='viridis')
    for i in range(len(path_scaled) - 1):
        wA = path_scaled[i]
        wB = path_scaled[i + 1]
        ax.arrow(wA[1], wA[2], wB[1] - wA[1], wB[2] - wA[2],
                head_width=0.3, head_length=0.3, fc='orange', ec='orange')
    ax.set_title("With normalization")
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.grid(True)

    plt.tight_layout()
    plt.show()