"""Module figures.

Each draw function is self-contained: its body is embedded verbatim into the
module's collapsible "the code that draws this" block, so what a reader copies
is exactly what produced the image. Keep imports inside each function.
"""


# --------------------------------------------------------------- PY1 NumPy ---
def broadcasting_rules():
    import matplotlib.pyplot as plt
    import numpy as np

    A = np.arange(12).reshape(4, 3)
    b = np.array([10, 20, 30])

    f, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for ax, data, title in [
        (axes[0], A, "A  (4, 3)"),
        (axes[1], np.broadcast_to(b, (4, 3)), "b  (3,) stretched"),
        (axes[2], A + b, "A + b  (4, 3)"),
    ]:
        ax.imshow(data, cmap="Blues", vmin=0, vmax=42)
        for (i, j), v in np.ndenumerate(data):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=9, color="#111827")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])

    # b is never copied: NumPy reads the same 3 values down all 4 rows.
    axes[1].set_xlabel("stride 0 down the rows", fontsize=9)
    f.suptitle("Broadcasting stretches without copying", fontsize=13, fontweight="bold")
    return f


def view_versus_copy():
    import matplotlib.pyplot as plt
    import numpy as np

    a = np.arange(10)
    view = a[2:5]          # a view: shares the buffer
    copy = a[[2, 3, 4]]    # fancy indexing: a copy
    view[0] = 999
    copy[0] = -1

    f, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.bar(np.arange(10) - 0.2, a, width=0.4, color="#2563eb", label="a after view[0]=999")
    ax.bar(np.arange(10) + 0.2, np.arange(10), width=0.4, color="#e5e7eb",
           label="a as originally built")
    ax.annotate("slice wrote through", xy=(2, 999), xytext=(3.4, 720),
                arrowprops=dict(arrowstyle="->", color="#dc2626"), color="#dc2626",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xticks(range(10))
    ax.set_xlabel("index")
    ax.set_title("A slice is a view; fancy indexing is a copy")
    ax.legend(loc="upper right")
    return f


# -------------------------------------------------------------- PY2 pandas ---
def split_apply_combine():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    groups = ["A", "B", "C"]
    values = [rng.normal(loc, 1.0, 40) for loc in (3.0, 5.0, 4.0)]

    f, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for i, (g, v) in enumerate(zip(groups, values)):
        axes[0].scatter(np.full_like(v, i) + rng.normal(0, 0.06, v.size), v,
                        s=14, alpha=0.65, color=["#2563eb", "#dc2626", "#059669"][i])
    axes[0].set_xticks(range(3), groups)
    axes[0].set_title("split: rows grouped by key")
    axes[0].set_ylabel("value")

    means = [v.mean() for v in values]
    axes[1].bar(groups, means, color=["#2563eb", "#dc2626", "#059669"])
    for i, m in enumerate(means):
        axes[1].text(i, m + 0.1, f"{m:.2f}", ha="center", fontsize=10)
    axes[1].set_title("apply + combine: one row per group")
    axes[1].set_ylim(0, 6.4)
    return f


# ------------------------------------------------------------ PY3 plotting ---
def distribution_shapes():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(3)
    data = {
        "symmetric": rng.normal(0, 1, 2000),
        "right-skewed": rng.lognormal(0, 0.7, 2000),
        "bimodal": np.concatenate([rng.normal(-2, 0.6, 1000), rng.normal(2, 0.6, 1000)]),
    }

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, (name, x) in zip(axes, data.items()):
        ax.hist(x, bins=40, color="#2563eb", alpha=0.85)
        ax.axvline(np.mean(x), color="#dc2626", lw=2, label="mean")
        ax.axvline(np.median(x), color="#059669", lw=2, ls="--", label="median")
        ax.set_title(name, fontsize=11)
        ax.set_yticks([])
    axes[0].legend(fontsize=9)
    f.suptitle("A histogram shows what a summary statistic hides",
               fontsize=13, fontweight="bold")
    return f


# --------------------------------------------------------- PY5 pipelines -----
def leakage_vs_pipeline():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(7)
    # 30 samples, 400 pure-noise features: no real signal exists at all.
    n, p = 30, 400
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n)

    # Leaky: pick the features that look best on ALL the data, then cross-validate.
    corr = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(p)])
    best = np.argsort(corr)[-5:]
    leaky = np.mean([
        (np.mean(X[np.arange(n) % 5 == k][:, best], axis=1) > 0).mean() for k in range(5)
    ])
    # Honest: selection happens inside each fold, so it sees only training rows.
    honest = []
    for k in range(5):
        tr = np.arange(n) % 5 != k
        c = np.array([abs(np.corrcoef(X[tr, j], y[tr])[0, 1]) for j in range(p)])
        sel = np.argsort(c)[-5:]
        honest.append((np.mean(X[~tr][:, sel], axis=1) > 0).mean())
    honest = np.mean(honest)

    f, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh(["select inside the fold\n(Pipeline)", "select on all data\n(leakage)"],
            [honest, leaky], color=["#059669", "#dc2626"], height=0.55)
    ax.axvline(0.5, color="#111827", ls="--", lw=1.5)
    ax.text(0.505, -0.42, "true accuracy: the data is pure noise", fontsize=9.5, color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlim(0, 1)
    ax.set_xlabel("cross-validated accuracy")
    ax.set_title("Fitting a transform outside the fold invents skill")
    return f


# ------------------------------------------------------- MA1 linear algebra ---
def linear_map_unit_square():
    import matplotlib.pyplot as plt
    import numpy as np

    A = np.array([[2.0, 1.0], [0.5, 1.5]])
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    mapped = A @ square

    f, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(square[0], square[1], color="#9ca3af", lw=2, label="unit square")
    ax.fill(square[0], square[1], color="#9ca3af", alpha=0.15)
    ax.plot(mapped[0], mapped[1], color="#2563eb", lw=2.5, label="after A")
    ax.fill(mapped[0], mapped[1], color="#2563eb", alpha=0.15)

    for vec, color in [(A[:, 0], "#dc2626"), (A[:, 1], "#059669")]:
        ax.annotate("", xy=vec, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(*A[:, 0], "  A e₁", color="#dc2626", fontsize=11)
    ax.text(*A[:, 1], "  A e₂", color="#059669", fontsize=11)
    ax.text(1.4, 0.25, f"area × |det A| = {abs(np.linalg.det(A)):.2f}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("A matrix is what it does to the unit square")
    ax.legend(loc="upper left")
    return f


# ------------------------------------------------------ MA2 eigen / spectral ---
def eigenvectors_are_fixed_directions():
    import matplotlib.pyplot as plt
    import numpy as np

    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    vals, vecs = np.linalg.eigh(A)

    theta = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    mapped = A @ circle

    f, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(circle[0], circle[1], color="#9ca3af", lw=1.8, label="unit circle")
    ax.plot(mapped[0], mapped[1], color="#2563eb", lw=2.2, label="image under A")
    for i, color in enumerate(["#dc2626", "#059669"]):
        v = vecs[:, i]
        ax.annotate("", xy=v * vals[i], xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.5),
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.text(*(v * vals[i] * 1.06), f"  λ={vals[i]:.0f}", color=color, fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_aspect("equal")
    ax.set_title("Eigenvectors keep their direction; only the length changes")
    ax.legend(loc="lower right", fontsize=9)
    return f


# --------------------------------------------------------- MA3 gradients ------
def gradient_descent_path():
    import matplotlib.pyplot as plt
    import numpy as np

    def f_val(x, y):
        return 0.5 * x ** 2 + 4.0 * y ** 2

    xs = np.linspace(-4.5, 4.5, 200)
    ys = np.linspace(-2.2, 2.2, 200)
    X, Y = np.meshgrid(xs, ys)

    f, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.contour(X, Y, f_val(X, Y), levels=18, colors="#cbd5e1", linewidths=0.9)

    for lr, color, label in [(0.06, "#2563eb", "lr = 0.06"), (0.22, "#dc2626", "lr = 0.22")]:
        p = np.array([-4.0, 1.8])
        path = [p.copy()]
        for _ in range(40):
            grad = np.array([p[0], 8.0 * p[1]])
            p = p - lr * grad
            path.append(p.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color=color, ms=3.2, lw=1.6, label=label)

    ax.plot(0, 0, "*", color="#059669", ms=16)
    ax.set_title("Steep directions set the largest stable step")
    ax.legend(loc="upper right")
    return f


# ------------------------------------------------------------ PS1 Bayes ------
def bayes_posterior_update():
    import matplotlib.pyplot as plt
    import numpy as np

    theta = np.linspace(0, 1, 400)

    def beta_pdf(a, b):
        d = theta ** (a - 1) * (1 - theta) ** (b - 1)
        return d / np.trapezoid(d, theta)

    f, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(theta, beta_pdf(2, 2), color="#9ca3af", lw=2, label="prior  Beta(2, 2)")
    for heads, flips, color in [(3, 5, "#2563eb"), (12, 20, "#059669"), (60, 100, "#dc2626")]:
        ax.plot(theta, beta_pdf(2 + heads, 2 + flips - heads), color=color, lw=2.2,
                label=f"after {heads}/{flips} heads")
    ax.axvline(0.6, color="#111827", ls="--", lw=1.2)
    ax.text(0.615, 6.0, "true rate 0.6", fontsize=9.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("P(heads)")
    ax.set_ylabel("density")
    ax.set_title("Evidence concentrates the posterior")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------ PS3 inference --------
def sampling_distribution_and_ci():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(11)
    true_mu, sigma, n = 5.0, 2.0, 25

    means = np.array([rng.normal(true_mu, sigma, n).mean() for _ in range(4000)])

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].hist(means, bins=45, color="#2563eb", alpha=0.85)
    axes[0].axvline(true_mu, color="#dc2626", lw=2)
    axes[0].set_title(f"means of {n}-sample draws")
    axes[0].set_xlabel("sample mean")
    axes[0].set_yticks([])

    covered = 0
    for k in range(30):
        s = rng.normal(true_mu, sigma, n)
        half = 1.96 * s.std(ddof=1) / np.sqrt(n)
        lo, hi = s.mean() - half, s.mean() + half
        hit = lo <= true_mu <= hi
        covered += hit
        axes[1].plot([lo, hi], [k, k], color="#059669" if hit else "#dc2626", lw=2)
    axes[1].axvline(true_mu, color="#111827", ls="--", lw=1.5)
    axes[1].set_title(f"95% intervals: {covered}/30 contain μ")
    axes[1].set_yticks([])
    return f


# ---------------------------------------------------- ML1 feature engineering ---
def scaling_changes_distance():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(5)
    age = rng.normal(40, 8, 120)
    income = rng.normal(60000, 15000, 120)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].scatter(age, income, s=16, color="#2563eb", alpha=0.7)
    axes[0].set_title("raw units: income dominates")
    axes[0].set_xlabel("age (years)"), axes[0].set_ylabel("income ($)")

    z = lambda v: (v - v.mean()) / v.std()
    axes[1].scatter(z(age), z(income), s=16, color="#059669", alpha=0.7)
    axes[1].set_title("standardized: both features count")
    axes[1].set_xlabel("age (z)"), axes[1].set_ylabel("income (z)")
    axes[1].set_aspect("equal")
    return f


# ------------------------------------------------- ML2 supervised learning ----
def bias_variance_tradeoff():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(-1, 1, 25))
    y = np.sin(3 * x) + rng.normal(0, 0.22, x.size)
    grid = np.linspace(-1, 1, 300)

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    for ax, deg, title in [(axes[0], 1, "degree 1: underfit"),
                           (axes[1], 4, "degree 4: about right"),
                           (axes[2], 16, "degree 16: overfit")]:
        coef = np.polyfit(x, y, deg)
        ax.scatter(x, y, s=18, color="#111827", zorder=3)
        ax.plot(grid, np.polyval(coef, grid), color="#2563eb", lw=2.2)
        ax.plot(grid, np.sin(3 * grid), color="#059669", lw=1.4, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-2, 2)
    axes[0].set_ylabel("y")
    f.suptitle("Too simple misses the signal; too flexible fits the noise",
               fontsize=13, fontweight="bold")
    return f


def logistic_decision_boundary():
    import matplotlib.pyplot as plt
    import numpy as np

    z = np.linspace(-8, 8, 400)
    sigmoid = 1 / (1 + np.exp(-z))

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    axes[0].plot(z, sigmoid, color="#2563eb", lw=2.5)
    axes[0].axhline(0.5, color="#9ca3af", ls="--", lw=1.2)
    axes[0].axvline(0, color="#9ca3af", ls="--", lw=1.2)
    axes[0].set_title("σ maps the log-odds to a probability")
    axes[0].set_xlabel("z = wᵀx + b"), axes[0].set_ylabel("P(y=1)")

    # Loss for a single positive example: -log σ(z) blows up when z is confidently wrong.
    axes[1].plot(z, -np.log(sigmoid), color="#dc2626", lw=2.5, label="y = 1")
    axes[1].plot(z, -np.log(1 - sigmoid), color="#059669", lw=2.5, label="y = 0")
    axes[1].set_ylim(0, 8)
    axes[1].set_title("Log loss punishes confident mistakes")
    axes[1].set_xlabel("z"), axes[1].set_ylabel("loss")
    axes[1].legend()
    return f


# ------------------------------------------------------------- ML5 clustering ---
def kmeans_iterations():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(4)
    centres = np.array([[0, 0], [4, 1], [2, 4.5]])
    X = np.vstack([c + rng.normal(0, 0.8, (45, 2)) for c in centres])

    mu = np.array([[-1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])  # deliberately bad start
    snapshots = []
    for step in range(6):
        d = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        snapshots.append((mu.copy(), labels.copy()))
        mu = np.array([X[labels == k].mean(0) if np.any(labels == k) else mu[k]
                       for k in range(3)])

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    for ax, idx in zip(axes, [0, 1, 5]):
        m, lab = snapshots[idx]
        for k, color in enumerate(["#2563eb", "#dc2626", "#059669"]):
            ax.scatter(*X[lab == k].T, s=12, color=color, alpha=0.6)
            ax.scatter(*m[k], marker="X", s=110, color=color, edgecolor="white", lw=1.5)
        ax.set_title(f"iteration {idx}", fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    f.suptitle("Assign, move, repeat — until nothing changes",
               fontsize=13, fontweight="bold")
    return f


def choosing_k_silhouette():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(6)
    centres = np.array([[0, 0], [4, 1], [2, 4.5], [6, 4]])
    X = np.vstack([c + rng.normal(0, 0.7, (50, 2)) for c in centres])

    def kmeans(X, k, iters=40):
        mu = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - mu[None]) ** 2).sum(-1).argmin(1)
            mu = np.array([X[lab == j].mean(0) if np.any(lab == j) else mu[j]
                           for j in range(k)])
        return lab, mu

    ks = range(2, 9)
    inertia = []
    for k in ks:
        lab, mu = kmeans(X, k)
        inertia.append(((X - mu[lab]) ** 2).sum())

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(list(ks), inertia, "-o", color="#2563eb", lw=2.2)
    ax.axvline(4, color="#dc2626", ls="--", lw=1.6)
    ax.annotate("true K = 4", xy=(4, inertia[2]), xytext=(4.6, inertia[1]),
                arrowprops=dict(arrowstyle="->", color="#dc2626"), color="#dc2626",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("K"), ax.set_ylabel("inertia (within-cluster SSE)")
    ax.set_title("Inertia always falls, so the elbow is a judgement call")
    return f


# ----------------------------------------------------- DL1 neural networks ----
def activation_functions():
    import matplotlib.pyplot as plt
    import numpy as np

    z = np.linspace(-4, 4, 400)
    acts = {
        "ReLU": np.maximum(0, z),
        "tanh": np.tanh(z),
        "sigmoid": 1 / (1 + np.exp(-z)),
        "GELU": 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3))),
    }
    grads = {
        "ReLU": (z > 0).astype(float),
        "tanh": 1 - np.tanh(z) ** 2,
        "sigmoid": (s := 1 / (1 + np.exp(-z))) * (1 - s),
        "GELU": np.gradient(acts["GELU"], z),
    }

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    for (name, v), color in zip(acts.items(), ["#2563eb", "#dc2626", "#059669", "#7c3aed"]):
        axes[0].plot(z, v, lw=2.2, color=color, label=name)
        axes[1].plot(z, grads[name], lw=2.2, color=color, label=name)
    axes[0].set_title("activation"), axes[1].set_title("its gradient")
    axes[1].axhline(0, color="#9ca3af", lw=1)
    axes[0].legend(fontsize=9)
    axes[1].text(-3.8, 0.85, "sigmoid and tanh\nflatten out here", fontsize=9, color="#dc2626",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    return f


def xor_needs_a_hidden_layer():
    import matplotlib.pyplot as plt
    import numpy as np

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    # One hidden layer of two ReLUs is enough to separate XOR.
    W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, -1.0])
    W2 = np.array([1.0, -2.0])
    b2 = 0.0

    g = np.linspace(-0.3, 1.3, 220)
    XX, YY = np.meshgrid(g, g)
    grid = np.stack([XX.ravel(), YY.ravel()], 1)
    H = np.maximum(0, grid @ W1.T + b1)
    Z = (H @ W2 + b2).reshape(XX.shape)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].scatter(*X[y == 0].T, s=110, color="#dc2626", marker="s", label="class 0")
    axes[0].scatter(*X[y == 1].T, s=110, color="#2563eb", marker="o", label="class 1")
    axes[0].plot([-0.3, 1.3], [1.1, -0.1], color="#9ca3af", ls="--")
    axes[0].set_title("no straight line separates XOR")
    axes[0].legend(fontsize=9)

    axes[1].contourf(XX, YY, Z > 0.5, levels=[-0.5, 0.5, 1.5],
                     colors=["#fee2e2", "#dbeafe"])
    axes[1].scatter(*X[y == 0].T, s=110, color="#dc2626", marker="s")
    axes[1].scatter(*X[y == 1].T, s=110, color="#2563eb", marker="o")
    axes[1].set_title("two hidden ReLUs bend it into shape")
    return f


# ------------------------------------------------------ DL2 regularization ----
def regularization_curves():
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = np.arange(1, 61)
    train_plain = 0.9 * np.exp(-epochs / 12) + 0.03
    val_plain = 0.9 * np.exp(-epochs / 14) + 0.16 + 0.0045 * np.maximum(0, epochs - 22)
    train_reg = 0.9 * np.exp(-epochs / 15) + 0.10
    val_reg = 0.9 * np.exp(-epochs / 17) + 0.145

    f, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(epochs, train_plain, color="#2563eb", lw=2, label="train, no dropout")
    ax.plot(epochs, val_plain, color="#2563eb", lw=2, ls="--", label="val, no dropout")
    ax.plot(epochs, train_reg, color="#059669", lw=2, label="train, dropout")
    ax.plot(epochs, val_reg, color="#059669", lw=2, ls="--", label="val, dropout")
    ax.axvline(22, color="#dc2626", lw=1.4, ls=":")
    ax.annotate("validation turns up:\nmemorising starts here", xy=(22, val_plain[21]),
                xytext=(30, 0.62), arrowprops=dict(arrowstyle="->", color="#dc2626"),
                color="#dc2626", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("epoch"), ax.set_ylabel("loss")
    ax.set_title("Regularization trades training loss for the gap")
    ax.legend(fontsize=9, ncol=2)
    return f


# ------------------------------------------------------------- DL4 optimizers ---
def optimizer_trajectories():
    import matplotlib.pyplot as plt
    import numpy as np

    # A narrow valley: gradient descent zig-zags, momentum and Adam do not.
    def grad(p):
        return np.array([2 * p[0], 20 * p[1]])

    xs = np.linspace(-2.6, 2.6, 200)
    ys = np.linspace(-1.1, 1.1, 200)
    X, Y = np.meshgrid(xs, ys)

    f, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.contour(X, Y, X ** 2 + 10 * Y ** 2, levels=16, colors="#cbd5e1", linewidths=0.9)

    start = np.array([-2.4, 0.95])
    paths = {}

    # lr just past 1/curvature in y, so the step overshoots and oscillates.
    p, hist = start.copy(), [start.copy()]
    for _ in range(60):
        p = p - 0.09 * grad(p)
        hist.append(p.copy())
    paths["SGD"] = np.array(hist)

    p, v, hist = start.copy(), np.zeros(2), [start.copy()]
    for _ in range(60):
        v = 0.85 * v - 0.012 * grad(p)
        p = p + v
        hist.append(p.copy())
    paths["momentum"] = np.array(hist)

    p, m, v, hist = start.copy(), np.zeros(2), np.zeros(2), [start.copy()]
    for t in range(1, 61):
        g = grad(p)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        mh, vh = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
        p = p - 0.12 * mh / (np.sqrt(vh) + 1e-8)
        hist.append(p.copy())
    paths["Adam"] = np.array(hist)

    for (name, path), color in zip(paths.items(), ["#dc2626", "#d97706", "#2563eb"]):
        ax.plot(path[:, 0], path[:, 1], "-o", ms=2.6, lw=1.6, color=color, label=name)
    ax.plot(0, 0, "*", color="#059669", ms=16)
    ax.set_title("In a narrow valley, plain SGD zig-zags")
    ax.legend(loc="upper right")
    return f


# ------------------------------------------------------------------ CV1 CNNs ---
def convolution_operation():
    import matplotlib.pyplot as plt
    import numpy as np

    img = np.zeros((9, 9))
    img[2:7, 4] = 1.0          # a vertical bar
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)  # Sobel-x

    out = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            out[i, j] = np.sum(img[i:i + 3, j:j + 3] * kernel)

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    for ax, data, title, cmap in [
        (axes[0], img, "input", "Greys"),
        (axes[1], kernel, "vertical-edge kernel", "RdBu"),
        (axes[2], out, "feature map", "RdBu"),
    ]:
        m = np.abs(data).max()
        ax.imshow(data, cmap=cmap, vmin=-m, vmax=m)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    f.suptitle("One kernel, slid everywhere, finds the same feature anywhere",
               fontsize=13, fontweight="bold")
    return f


# ------------------------------------------------------------ MA4 backprop ----
def gradient_flow_by_depth():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(1)
    depth = 40
    layers = np.arange(1, depth + 1)

    f, ax = plt.subplots(figsize=(7.2, 3.3))
    for gain, color, label in [(0.6, "#dc2626", "weights too small: vanishing"),
                               (1.0, "#059669", "well-scaled: stable"),
                               (1.6, "#d97706", "weights too large: exploding")]:
        # Product of per-layer Jacobian norms as the signal walks back.
        norms = np.cumprod(gain * np.exp(rng.normal(0, 0.05, depth)))
        ax.semilogy(layers, norms, lw=2.4, color=color, label=label)
    ax.axhline(1.0, color="#9ca3af", ls="--", lw=1.2)
    ax.set_xlabel("layers back from the loss")
    ax.set_ylabel("gradient magnitude (log)")
    ax.set_title("The chain rule multiplies, so depth compounds the scale")
    ax.legend(fontsize=9)
    return f


# ---------------------------------------------------------------- MA5 SVD ----
def svd_low_rank_approximation():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(0)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    img = (np.sin(xx / 6.0) + np.cos(yy / 9.0) + 0.35 * rng.normal(size=(n, n)))

    U, S, Vt = np.linalg.svd(img, full_matrices=False)

    f, axes = plt.subplots(1, 4, figsize=(7.8, 2.5))
    for ax, k in zip(axes[:3], [2, 6, 20]):
        approx = (U[:, :k] * S[:k]) @ Vt[:k]
        ax.imshow(approx, cmap="Greys_r")
        kept = S[:k].sum() / S.sum()
        ax.set_title(f"rank {k}\n{kept:.0%} of the mass", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    axes[3].semilogy(S, color="#2563eb", lw=2.2)
    axes[3].set_title("singular values", fontsize=10)
    axes[3].set_xlabel("index")
    return f


# ------------------------------------------------------- ML6 curves ----------
def roc_and_pr_under_imbalance():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(9)
    n_neg, n_pos = 2000, 40          # heavy imbalance: 2% positives
    neg = rng.normal(0.0, 1.0, n_neg)
    pos = rng.normal(1.6, 1.0, n_pos)
    scores = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / y.sum()
    fpr = fp / (len(y) - y.sum())
    precision = tp / (tp + fp)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].plot(fpr, tpr, color="#2563eb", lw=2.4)
    axes[0].plot([0, 1], [0, 1], color="#9ca3af", ls="--", lw=1.2)
    axes[0].set_title(f"ROC — AUC {np.trapezoid(tpr, fpr):.2f}")
    axes[0].set_xlabel("false positive rate"), axes[0].set_ylabel("true positive rate")

    axes[1].plot(tpr, precision, color="#dc2626", lw=2.4)
    axes[1].axhline(y.sum() / len(y), color="#9ca3af", ls="--", lw=1.2)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"PR — baseline {y.sum() / len(y):.2f}")
    axes[1].set_xlabel("recall"), axes[1].set_ylabel("precision")
    f.suptitle("The same model looks strong on ROC and weak on PR",
               fontsize=13, fontweight="bold")
    return f


def calibration_reliability():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(15)
    p_true = rng.uniform(0, 1, 4000)
    y = (rng.uniform(size=p_true.size) < p_true).astype(int)
    overconfident = np.clip(p_true + 0.28 * (p_true - 0.5) * 2, 0.001, 0.999)

    f, ax = plt.subplots(figsize=(7.2, 3.3))
    bins = np.linspace(0, 1, 11)
    for probs, color, label in [(p_true, "#059669", "calibrated"),
                                (overconfident, "#dc2626", "overconfident")]:
        idx = np.digitize(probs, bins) - 1
        xs = [probs[idx == b].mean() for b in range(10) if np.any(idx == b)]
        ys = [y[idx == b].mean() for b in range(10) if np.any(idx == b)]
        brier = np.mean((probs - y) ** 2)
        ax.plot(xs, ys, "-o", color=color, lw=2.2, label=f"{label} (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", ls="--", lw=1.4)
    ax.set_xlabel("predicted probability"), ax.set_ylabel("observed frequency")
    ax.set_title("A reliability diagram compares the claim to reality")
    ax.legend(fontsize=9)
    return f


# --------------------------------------------------------------- ML7 PCA -----
def pca_variance_explained():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(8)
    latent = rng.normal(size=(400, 3))
    mixing = rng.normal(size=(3, 12))
    X = latent @ mixing + 0.35 * rng.normal(size=(400, 12))
    X = X - X.mean(0)

    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].bar(range(1, 13), var, color="#2563eb")
    axes[0].plot(range(1, 13), np.cumsum(var), "-o", color="#dc2626", ms=4)
    axes[0].axhline(0.95, color="#059669", ls="--", lw=1.4)
    axes[0].set_xlabel("component"), axes[0].set_ylabel("share of variance")
    axes[0].set_title("3 directions carry almost everything")

    proj = X @ Vt[:2].T
    axes[1].scatter(proj[:, 0], proj[:, 1], s=12, color="#2563eb", alpha=0.6)
    axes[1].set_xlabel("PC 1"), axes[1].set_ylabel("PC 2")
    axes[1].set_title("12 columns projected onto 2")
    return f


# ------------------------------------------------------- DL5 LR schedules ----
def learning_rate_schedules():
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.arange(0, 1000)
    base = 1e-3
    warmup = 100

    cosine = np.where(steps < warmup, base * steps / warmup,
                      0.5 * base * (1 + np.cos(np.pi * (steps - warmup) / (1000 - warmup))))
    step_decay = base * 0.5 ** (steps // 300)
    constant = np.full_like(steps, base, dtype=float)

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(steps, cosine, color="#2563eb", lw=2.4, label="warmup + cosine")
    ax.plot(steps, step_decay, color="#dc2626", lw=2.2, label="step decay")
    ax.plot(steps, constant, color="#9ca3af", lw=2.0, ls="--", label="constant")
    ax.set_xlabel("step"), ax.set_ylabel("learning rate")
    ax.set_title("Warmup protects the first few hundred steps")
    ax.legend(fontsize=9)
    return f


# --------------------------------------------------------- TF1 attention -----
def attention_heatmap():
    import matplotlib.pyplot as plt
    import numpy as np

    tokens = ["The", "cat", "that", "ate", "the", "fish", "was", "full"]
    n = len(tokens)
    rng = np.random.default_rng(21)

    scores = rng.normal(0, 0.5, (n, n))
    # "was" should look back at its subject "cat", not the nearer "fish".
    scores[6, 1] += 3.2
    scores[7, 1] += 2.1
    scores[5, 3] += 2.4
    scores = scores - scores.max(1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(1, keepdims=True)

    f, ax = plt.subplots(figsize=(7.8, 5.5))
    im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=attn.max())
    ax.set_xticks(range(n), tokens, rotation=45, ha="right")
    ax.set_yticks(range(n), tokens)
    ax.set_xlabel("attends to"), ax.set_ylabel("query token")
    ax.set_title("Attention can reach past nearby words")
    ax.grid(False)
    f.colorbar(im, ax=ax, shrink=0.8, label="weight")
    return f


def causal_versus_bidirectional_mask():
    import matplotlib.pyplot as plt
    import numpy as np

    n = 8
    causal = np.tril(np.ones((n, n)))
    bidir = np.ones((n, n))

    f, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, m, title in [(axes[0], bidir, "BERT: every token sees every token"),
                         (axes[1], causal, "GPT: a token sees only the past")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1.4)
        ax.set_xlabel("key position"), ax.set_ylabel("query position")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(n)), ax.set_yticks(range(n))
        ax.grid(False)
    return f


# ------------------------------------------------- RP2 contrastive learning ---
def contrastive_embedding_space():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(12)
    theta = np.array([0.4, 2.2, 4.1])
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))

    for ax, spread, title in [(axes[0], 1.15, "before training: classes overlap"),
                              (axes[1], 0.16, "after InfoNCE: tight, separated")]:
        circle = plt.Circle((0, 0), 1.0, fill=False, color="#9ca3af", lw=1.4)
        ax.add_patch(circle)
        for t, color in zip(theta, ["#2563eb", "#dc2626", "#059669"]):
            a = t + rng.normal(0, spread, 45)
            ax.scatter(np.cos(a), np.sin(a), s=16, color=color, alpha=0.75)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35), ax.set_ylim(-1.35, 1.35)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=11)
    f.suptitle("Embeddings live on the unit sphere; the loss moves them around it",
               fontsize=12.5, fontweight="bold")
    return f


def infonce_temperature():
    import matplotlib.pyplot as plt
    import numpy as np

    sims = np.array([0.9, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.1])
    f, ax = plt.subplots(figsize=(7.2, 3.1))
    width = 0.26
    for i, (tau, color) in enumerate(zip([0.5, 0.1, 0.02], ["#9ca3af", "#2563eb", "#dc2626"])):
        p = np.exp(sims / tau)
        p /= p.sum()
        ax.bar(np.arange(len(sims)) + (i - 1) * width, p, width=width,
               color=color, label=f"τ = {tau}")
    ax.set_xticks(range(len(sims)), ["pos"] + [f"neg{j}" for j in range(1, len(sims))])
    ax.set_ylabel("softmax weight")
    ax.set_title("Low temperature concentrates the loss on the hardest negatives")
    ax.legend(fontsize=9)
    return f


# -------------------------------------------------------- PS4 information ----
def entropy_of_a_coin():
    import matplotlib.pyplot as plt
    import numpy as np

    p = np.linspace(0.001, 0.999, 400)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(p, H, color="#2563eb", lw=2.6)
    ax.axvline(0.5, color="#9ca3af", ls="--", lw=1.2)
    ax.annotate("1 bit: maximum surprise", xy=(0.5, 1.0), xytext=(0.56, 0.72),
                arrowprops=dict(arrowstyle="->", color="#111827"), fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.annotate("a certain outcome\ncosts nothing to send", xy=(0.03, 0.19),
                xytext=(0.1, 0.42), arrowprops=dict(arrowstyle="->", color="#dc2626"),
                color="#dc2626", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("P(heads)"), ax.set_ylabel("entropy (bits)")
    ax.set_title("Entropy peaks when you know least")
    return f


def kl_divergence_asymmetry():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-6, 8, 500)
    gauss = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    p = 0.5 * gauss(-1.5, 0.8) + 0.5 * gauss(3.5, 0.8)   # bimodal target
    q_wide = gauss(1.0, 2.6)
    q_narrow = gauss(3.5, 0.8)

    kl = lambda a, b: np.trapezoid(a * np.log((a + 1e-12) / (b + 1e-12)), x)

    f, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.plot(x, p, color="#111827", lw=2.4, label="p (target)")
    ax.plot(x, q_wide, color="#2563eb", lw=2.2,
            label=f"q covers both  KL(p‖q)={kl(p, q_wide):.2f}")
    ax.plot(x, q_narrow, color="#dc2626", lw=2.2,
            label=f"q picks one  KL(p‖q)={kl(p, q_narrow):.2f}")
    ax.set_title("Forward KL punishes missing a mode")
    ax.set_yticks([])
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------ CV5 object detection ---
def iou_and_nms():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
        return inter / (area(a) + area(b) - inter)

    gt = (1.0, 1.0, 4.0, 3.5)
    preds = [(1.2, 1.1, 4.1, 3.4), (2.4, 1.6, 5.2, 4.0)]

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    for ax, pred in zip(axes, preds):
        for box, color, label in [(gt, "#059669", "ground truth"), (pred, "#2563eb", "prediction")]:
            ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                           fill=False, lw=2.4, edgecolor=color, label=label))
        ax.set_xlim(0, 6), ax.set_ylim(0, 5)
        ax.set_title(f"IoU = {iou(gt, pred):.2f}", fontsize=12)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].text(0.2, 0.3, "kept: above the 0.5 threshold", fontsize=9, color="#059669")
    axes[1].text(0.2, 0.3, "suppressed by NMS", fontsize=9, color="#dc2626")
    return f


# ------------------------------------------------------------- CV6 ViT -------
def patch_embedding_grid():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(31)
    n, patch = 24, 6
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.sin(xx / 3.0) * np.cos(yy / 4.0) + 0.25 * rng.normal(size=(n, n))

    f, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    axes[0].imshow(img, cmap="Greys_r")
    for k in range(patch, n, patch):
        axes[0].axhline(k - 0.5, color="#2563eb", lw=1.6)
        axes[0].axvline(k - 0.5, color="#2563eb", lw=1.6)
    axes[0].set_title(f"image split into {(n // patch) ** 2} patches", fontsize=11)
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[0].grid(False)

    tokens = img.reshape(n // patch, patch, n // patch, patch).transpose(0, 2, 1, 3)
    tokens = tokens.reshape((n // patch) ** 2, patch * patch)
    axes[1].imshow(tokens, cmap="RdBu", aspect="auto")
    axes[1].set_xlabel("flattened pixels"), axes[1].set_ylabel("patch token")
    axes[1].set_title("each patch becomes one token", fontsize=11)
    axes[1].grid(False)
    return f


# --------------------------------------------------------- NL1 tokenization ---
def token_frequency_zipf():
    import matplotlib.pyplot as plt
    import numpy as np

    ranks = np.arange(1, 5001)
    freq = 1.0 / ranks ** 1.07          # Zipf's law
    freq = freq / freq.sum()

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.loglog(ranks, freq, color="#2563eb", lw=2.4)
    cum = np.cumsum(freq)
    k = int(np.searchsorted(cum, 0.8) + 1)
    ax.axvline(k, color="#dc2626", ls="--", lw=1.6)
    ax.annotate(f"{k} tokens cover 80% of text", xy=(k, freq[k]),
                xytext=(k * 1.6, freq[k] * 12),
                arrowprops=dict(arrowstyle="->", color="#dc2626"), color="#dc2626",
                fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("rank"), ax.set_ylabel("frequency")
    ax.set_title("A few tokens do most of the work; the tail is enormous")
    return f


# -------------------------------------------------------------- NL2 RNNs -----
def rnn_gradient_horizon():
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.arange(1, 81)
    f, ax = plt.subplots(figsize=(7.2, 3.2))
    for w, color, label in [(0.85, "#dc2626", "|w| = 0.85 (vanilla RNN)"),
                            (0.99, "#d97706", "|w| = 0.99"),
                            (1.0, "#059669", "gated cell (LSTM/GRU)")]:
        ax.semilogy(steps, w ** steps, lw=2.4, color=color, label=label)
    ax.axhline(1e-3, color="#9ca3af", ls="--", lw=1.2)
    ax.text(2, 1.4e-3, "below here the signal is lost in float noise", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("timesteps back")
    ax.set_ylabel("gradient magnitude (log)")
    ax.set_title("How far back a recurrent gradient survives")
    ax.legend(fontsize=9)
    return f


# --------------------------------------------------------------- TF5 RAG -----
def retrieval_similarity_ranking():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(41)
    d = 64
    query = rng.normal(size=d)
    query /= np.linalg.norm(query)

    docs = rng.normal(size=(300, d))
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    # Plant three genuinely relevant documents.
    for i, mix in enumerate([0.92, 0.80, 0.66]):
        v = mix * query + (1 - mix) * docs[i]
        docs[i] = v / np.linalg.norm(v)

    sims = docs @ query
    order = np.argsort(-sims)

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#dc2626" if i < 3 else "#cbd5e1" for i in order]
    ax.bar(range(40), sims[order][:40], color=colors[:40])
    ax.axhline(sims[order][:40].mean(), color="#111827", ls="--", lw=1.2)
    ax.set_xlabel("rank"), ax.set_ylabel("cosine similarity")
    ax.set_title("Retrieval keeps the top-k; the rest never reach the model")
    ax.text(4.5, sims[order][0] * 0.93, "planted relevant documents",
            color="#dc2626", fontsize=9.5)
    return f


# --------------------------------------------------------------- TF6 RoPE ----
def rope_relative_decay():
    import matplotlib.pyplot as plt
    import numpy as np

    d = 64
    theta = 10000 ** (-np.arange(0, d, 2) / d)

    def rope(x, pos):
        ang = pos * theta
        c, s = np.cos(ang), np.sin(ang)
        xe, xo = x[0::2], x[1::2]
        return np.stack([xe * c - xo * s, xe * s + xo * c], 1).ravel()

    rng = np.random.default_rng(17)
    q = rng.normal(size=d)
    offsets = np.arange(0, 128)
    scores = [rope(q, 0) @ rope(q, k) / (np.linalg.norm(q) ** 2) for k in offsets]

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(offsets, scores, color="#2563eb", lw=2.2)
    ax.axhline(0, color="#9ca3af", lw=1.2)
    ax.set_xlabel("relative offset  m − n")
    ax.set_ylabel("dot product (normalised)")
    ax.set_title("RoPE makes the score depend only on the distance between tokens")
    return f


# ------------------------------------------------------- RP1/RP3 autoencoders ---
def autoencoder_bottleneck():
    import matplotlib.pyplot as plt
    import numpy as np

    widths = [12, 8, 4, 2, 4, 8, 12]
    labels = ["in", "", "", "z", "", "", "out"]

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (w, lab) in enumerate(zip(widths, labels)):
        ys = np.linspace(-w / 2, w / 2, w)
        color = "#dc2626" if i == 3 else "#2563eb"
        ax.scatter(np.full(w, i), ys, s=52, color=color, zorder=3)
        if lab:
            ax.text(i, w / 2 + 1.2, lab, ha="center", fontsize=11, color="#111827")
        if i:
            prev = np.linspace(-widths[i - 1] / 2, widths[i - 1] / 2, widths[i - 1])
            for a in prev:
                for b in ys:
                    ax.plot([i - 1, i], [a, b], color="#e5e7eb", lw=0.35, zorder=1)
    ax.set_ylim(-8, 9)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("The bottleneck forces the network to keep only what matters")
    ax.grid(False)
    return f


def vae_latent_interpolation():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(23)
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))

    z = rng.normal(size=(600, 2))
    axes[0].scatter(z[:, 0], z[:, 1], s=10, color="#2563eb", alpha=0.5)
    circle = plt.Circle((0, 0), 2.0, fill=False, color="#dc2626", lw=1.8, ls="--")
    axes[0].add_patch(circle)
    axes[0].set_title("KL keeps the code close to N(0, I)", fontsize=11)
    axes[0].set_aspect("equal")

    a, b = np.array([-1.8, -1.2]), np.array([1.7, 1.4])
    ts = np.linspace(0, 1, 9)
    path = np.array([(1 - t) * a + t * b for t in ts])
    axes[1].scatter(z[:, 0], z[:, 1], s=8, color="#e5e7eb")
    axes[1].plot(path[:, 0], path[:, 1], "-o", color="#059669", lw=2)
    axes[1].set_title("so a straight walk decodes to valid samples", fontsize=11)
    axes[1].set_aspect("equal")
    return f


# ------------------------------------------------------------- GM1 GANs ------
def gan_training_dynamics():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(19)
    steps = np.arange(600)
    d_loss = 0.69 + 0.25 * np.exp(-steps / 150) * np.cos(steps / 22) + rng.normal(0, 0.03, 600)
    g_loss = 1.4 - 0.5 * np.exp(-steps / 180) + 0.3 * np.sin(steps / 26) + rng.normal(0, 0.04, 600)

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(steps, d_loss, color="#2563eb", lw=1.4, label="discriminator")
    ax.plot(steps, g_loss, color="#dc2626", lw=1.4, label="generator")
    ax.axhline(np.log(2), color="#059669", ls="--", lw=1.6)
    ax.text(430, np.log(2) + 0.05, "log 2: the balanced point", color="#059669", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("step"), ax.set_ylabel("loss")
    ax.set_title("Neither loss falls: the two players chase each other")
    ax.legend(fontsize=9)
    return f


# -------------------------------------------------------- GM2 diffusion ------
def diffusion_forward_process():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(27)
    n = 400
    t_ = rng.uniform(0, 2 * np.pi, n)
    x0 = np.stack([np.cos(t_) + 0.06 * rng.normal(size=n),
                   np.sin(t_) + 0.06 * rng.normal(size=n)], 1)

    T = 400
    betas = np.linspace(1e-4, 0.02, T)
    alpha_bar = np.cumprod(1 - betas)

    f, axes = plt.subplots(1, 4, figsize=(7.8, 2.4))
    for ax, t in zip(axes, [0, 60, 180, 399]):
        ab = alpha_bar[t]
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * rng.normal(size=x0.shape)
        ax.scatter(xt[:, 0], xt[:, 1], s=7, color="#2563eb", alpha=0.6)
        ax.set_title(f"t = {t}", fontsize=10)
        ax.set_xlim(-3.2, 3.2), ax.set_ylim(-3.2, 3.2)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal")
    f.suptitle("Forward diffusion dissolves structure into noise on a fixed schedule",
               fontsize=12.5, fontweight="bold")
    return f


# --------------------------------------------------------------- XT1 PINNs ---
def pinn_residual_loss():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 1, 300)
    exact = np.sin(np.pi * x)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].plot(x, exact, color="#059669", lw=2.4, label="exact")
    for k, color in [(3, "#dc2626"), (12, "#d97706"), (60, "#2563eb")]:
        approx = exact * (1 - np.exp(-k / 18)) + 0.25 * np.exp(-k / 18) * np.sin(6 * np.pi * x)
        axes[0].plot(x, approx, lw=1.8, color=color, alpha=0.9, label=f"{k * 100} steps")
    axes[0].set_title("the network is fit to the equation, not to data", fontsize=11)
    axes[0].legend(fontsize=8.5)

    steps = np.arange(1, 400)
    axes[1].semilogy(steps, 3.0 * np.exp(-steps / 60) + 1e-3, color="#2563eb",
                     lw=2.2, label="PDE residual")
    axes[1].semilogy(steps, 1.2 * np.exp(-steps / 25) + 2e-4, color="#dc2626",
                     lw=2.2, label="boundary loss")
    axes[1].set_xlabel("step"), axes[1].set_ylabel("loss (log)")
    axes[1].set_title("two loss terms, competing", fontsize=11)
    axes[1].legend(fontsize=9)
    return f


# ------------------------------------------------ TF8/TF9 attention cost -----
def attention_cost_scaling():
    import matplotlib.pyplot as plt
    import numpy as np

    n = np.logspace(2, 5.3, 200)
    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.loglog(n, n ** 2, color="#dc2626", lw=2.4, label="full attention  O(n²)")
    ax.loglog(n, n * np.log2(n), color="#d97706", lw=2.2, label="sparse / windowed  O(n log n)")
    ax.loglog(n, n, color="#059669", lw=2.2, label="linear attention  O(n)")
    ax.axvline(4096, color="#9ca3af", ls="--", lw=1.2)
    ax.text(4400, 3e3, "typical context", fontsize=9.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("sequence length n"), ax.set_ylabel("relative work (log)")
    ax.set_title("Why long context forces a different attention")
    ax.legend(fontsize=9)
    return f


def kv_cache_growth():
    import matplotlib.pyplot as plt
    import numpy as np

    n = np.arange(0, 32769, 256)
    layers, heads, dim, bytes_per = 32, 32, 128, 2

    mha = 2 * layers * heads * dim * n * bytes_per / 1e9
    gqa8 = 2 * layers * 8 * dim * n * bytes_per / 1e9
    mqa = 2 * layers * 1 * dim * n * bytes_per / 1e9

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(n, mha, color="#dc2626", lw=2.4, label="MHA: 32 KV heads")
    ax.plot(n, gqa8, color="#d97706", lw=2.2, label="GQA: 8 KV groups")
    ax.plot(n, mqa, color="#059669", lw=2.2, label="MQA: 1 KV head")
    ax.set_xlabel("tokens in the cache"), ax.set_ylabel("KV cache (GB, fp16)")
    ax.set_title("Sharing KV heads is what makes long context affordable")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------------------ RF1 RL ---
def value_iteration_convergence():
    import matplotlib.pyplot as plt
    import numpy as np

    n, gamma = 12, 0.9
    R = np.zeros(n)
    R[-1] = 1.0
    V = np.zeros(n)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    deltas = []
    for k in range(40):
        V_new = np.array([R[s] + (gamma * V[min(s + 1, n - 1)] if s < n - 1 else 0)
                          for s in range(n)])
        deltas.append(np.abs(V_new - V).max())
        V = V_new
        if k in (0, 2, 6, 39):
            axes[0].plot(range(n), V, "-o", ms=3.5, lw=1.8, label=f"sweep {k + 1}")
    axes[0].set_xlabel("state"), axes[0].set_ylabel("V(s)")
    axes[0].set_title("value propagates back from the reward", fontsize=11)
    axes[0].legend(fontsize=8.5)

    axes[1].semilogy(deltas, color="#2563eb", lw=2.2)
    axes[1].set_xlabel("sweep"), axes[1].set_ylabel("max change (log)")
    axes[1].set_title("contraction: error shrinks by γ each sweep", fontsize=11)
    return f


def exploration_tradeoff():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(13)
    true = np.array([0.2, 0.5, 0.75, 0.55])
    steps = 2000

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    for eps, color in [(0.0, "#dc2626"), (0.1, "#2563eb"), (0.5, "#d97706")]:
        q = np.zeros(4)
        counts = np.zeros(4)
        rewards = []
        for t in range(steps):
            a = rng.integers(4) if rng.uniform() < eps else int(np.argmax(q))
            r = float(rng.uniform() < true[a])
            counts[a] += 1
            q[a] += (r - q[a]) / counts[a]
            rewards.append(r)
        ax.plot(np.cumsum(rewards) / np.arange(1, steps + 1), color=color, lw=2.0,
                label=f"ε = {eps}")
    ax.axhline(true.max(), color="#059669", ls="--", lw=1.4)
    ax.text(60, true.max() + 0.012, "best arm", color="#059669", fontsize=9.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("step"), ax.set_ylabel("average reward")
    ax.set_title("Pure greed locks onto the first arm that looks good")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------------------ XT2 GNN ---
def message_passing_receptive_field():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(37)
    n = 60
    pos = rng.uniform(0, 1, (n, 2))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    A = (d < 0.23) & (d > 0)

    reach = {0: {12}}
    for k in range(1, 4):
        prev = reach[k - 1]
        reach[k] = prev | {j for i in prev for j in np.flatnonzero(A[i])}

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, k in zip(axes, [1, 2, 3]):
        for i in range(n):
            for j in np.flatnonzero(A[i]):
                if j > i:
                    ax.plot(*pos[[i, j]].T, color="#e5e7eb", lw=0.7, zorder=1)
        inside = np.array(sorted(reach[k]))
        outside = np.setdiff1d(np.arange(n), inside)
        ax.scatter(*pos[outside].T, s=16, color="#cbd5e1", zorder=2)
        ax.scatter(*pos[inside].T, s=26, color="#2563eb", zorder=3)
        ax.scatter(*pos[12], s=90, color="#dc2626", zorder=4)
        ax.set_title(f"{k} layer{'s' if k > 1 else ''}: {len(inside)} nodes", fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    f.suptitle("Each layer widens the receptive field by one hop",
               fontsize=12.5, fontweight="bold")
    return f


# ------------------------------------------------------------- PY6 PyTorch ---
def autograd_graph_and_loss():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, 200)
    y = 2.0 * x - 1.0 + rng.normal(0, 1.0, x.size)

    w, b, lr = 0.0, 0.0, 0.02
    losses = []
    for _ in range(120):
        pred = w * x + b
        losses.append(np.mean((pred - y) ** 2))
        # exactly what autograd would hand back
        gw = np.mean(2 * (pred - y) * x)
        gb = np.mean(2 * (pred - y))
        w -= lr * gw
        b -= lr * gb

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].scatter(x, y, s=12, color="#cbd5e1")
    grid = np.linspace(-3, 3, 50)
    axes[0].plot(grid, 2.0 * grid - 1.0, color="#059669", lw=2, ls="--", label="true")
    axes[0].plot(grid, w * grid + b, color="#2563eb", lw=2.4, label=f"fit  w={w:.2f}, b={b:.2f}")
    axes[0].legend(fontsize=9)
    axes[0].set_title("the fit after 120 steps", fontsize=11)

    axes[1].plot(losses, color="#2563eb", lw=2.2)
    axes[1].set_xlabel("step"), axes[1].set_ylabel("MSE")
    axes[1].set_title("the loss autograd is minimising", fontsize=11)
    return f


# ----------------------------------------------------- PS2 expectation -------
def law_of_large_numbers():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(5)
    n = 3000
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))

    for _ in range(12):
        draws = rng.exponential(2.0, n)
        axes[0].plot(np.cumsum(draws) / np.arange(1, n + 1), lw=1.0, alpha=0.7)
    axes[0].axhline(2.0, color="#dc2626", lw=2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("samples"), axes[0].set_ylabel("running mean")
    axes[0].set_title("each run wanders, then settles", fontsize=11)

    for k, color in [(1, "#cbd5e1"), (5, "#93c5fd"), (30, "#2563eb")]:
        means = rng.exponential(2.0, (4000, k)).mean(1)
        axes[1].hist(means, bins=50, density=True, alpha=0.75, color=color, label=f"n = {k}")
    axes[1].set_title("and the mean turns Gaussian", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].set_yticks([])
    return f


# ------------------------------------------------------ DL3 model training ---
def learning_curves_diagnosis():
    import matplotlib.pyplot as plt
    import numpy as np

    sizes = np.array([50, 100, 200, 400, 800, 1600, 3200])
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=True)

    axes[0].plot(sizes, 0.42 + 0.02 * np.exp(-sizes / 900), "-o", color="#2563eb", label="train")
    axes[0].plot(sizes, 0.46 + 0.03 * np.exp(-sizes / 900), "-o", color="#dc2626", label="val")
    axes[0].set_title("high bias: both plateau high", fontsize=11)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training examples"), axes[0].set_ylabel("error")
    axes[0].legend(fontsize=9)

    axes[1].plot(sizes, 0.02 + 0.03 * np.exp(-sizes / 400), "-o", color="#2563eb", label="train")
    axes[1].plot(sizes, 0.09 + 0.36 * np.exp(-sizes / 700), "-o", color="#dc2626", label="val")
    axes[1].set_title("high variance: a gap that more data closes", fontsize=11)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training examples")
    axes[1].legend(fontsize=9)
    return f


# ------------------------------------------------- CV2 augmentation ----------
def augmentation_transforms():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(2)
    n = 28
    yy, xx = np.mgrid[0:n, 0:n]
    base = np.exp(-(((xx - 12) / 6.0) ** 2 + ((yy - 14) / 8.0) ** 2))
    base += 0.15 * rng.normal(size=(n, n))

    variants = {
        "original": base,
        "flip": base[:, ::-1],
        "shift": np.roll(base, 4, axis=1),
        "brightness": np.clip(base * 1.5, 0, 1.6),
        "noise": base + 0.35 * rng.normal(size=(n, n)),
    }

    f, axes = plt.subplots(1, 5, figsize=(7.8, 2.0))
    for ax, (name, im) in zip(axes, variants.items()):
        ax.imshow(im, cmap="Greys_r")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    f.suptitle("Augmentation says: the label survives all of these",
               fontsize=12.5, fontweight="bold")
    return f


# ------------------------------------------------------ CV3 segmentation -----
def segmentation_metrics():
    import matplotlib.pyplot as plt
    import numpy as np

    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    truth = ((xx - 30) ** 2 + (yy - 32) ** 2) < 15 ** 2
    pred = ((xx - 34) ** 2 + (yy - 30) ** 2) < 16 ** 2

    inter = (truth & pred).sum()
    union = (truth | pred).sum()
    dice = 2 * inter / (truth.sum() + pred.sum())

    overlay = np.zeros((n, n, 3))
    overlay[truth & ~pred] = [0.86, 0.15, 0.15]
    overlay[pred & ~truth] = [0.15, 0.39, 0.92]
    overlay[truth & pred] = [0.02, 0.59, 0.41]
    overlay[~truth & ~pred] = [1, 1, 1]

    f, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.imshow(overlay)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"IoU = {inter / union:.2f}   Dice = {dice:.2f}")
    ax.text(1, 61, "green: correct   red: missed   blue: false alarm", fontsize=9)
    return f


# ---------------------------------------------------------- CV4 ResNets ------
def residual_connection_depth():
    import matplotlib.pyplot as plt
    import numpy as np

    depths = np.array([8, 14, 20, 32, 44, 56, 110])
    plain = np.array([9.0, 8.6, 8.9, 10.1, 11.6, 13.2, 17.5])
    resnet = np.array([9.2, 8.1, 7.5, 6.9, 6.5, 6.2, 6.0])

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(depths, plain, "-o", color="#dc2626", lw=2.2, label="plain stack")
    ax.plot(depths, resnet, "-o", color="#2563eb", lw=2.2, label="with residual connections")
    ax.set_xlabel("layers"), ax.set_ylabel("test error (%)")
    ax.set_title("Without a shortcut, deeper eventually means worse")
    ax.legend(fontsize=9)
    return f


# ----------------------------------------------- TF4 encoder-decoder ---------
def encoder_decoder_attention():
    import matplotlib.pyplot as plt
    import numpy as np

    src = ["le", "chat", "noir", "dort"]
    tgt = ["the", "black", "cat", "sleeps"]
    A = np.array([
        [0.82, 0.08, 0.05, 0.05],
        [0.05, 0.10, 0.78, 0.07],
        [0.06, 0.80, 0.09, 0.05],
        [0.04, 0.06, 0.06, 0.84],
    ])

    f, ax = plt.subplots(figsize=(7.6, 5.1))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), src)
    ax.set_yticks(range(4), tgt)
    ax.set_xlabel("source (encoder)"), ax.set_ylabel("target (decoder)")
    ax.set_title("Cross-attention learns the alignment")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(A):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.5 else "#111827")
    f.colorbar(im, ax=ax, shrink=0.8)
    return f


# ------------------------------------------------- RP4 autoregressive --------
def sampling_temperature():
    import matplotlib.pyplot as plt
    import numpy as np

    logits = np.array([3.2, 2.6, 2.1, 1.4, 0.9, 0.5, 0.1, -0.4, -1.0, -1.8])
    words = [f"w{i}" for i in range(10)]

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    width = 0.26
    for i, (T, color) in enumerate(zip([0.5, 1.0, 1.6], ["#2563eb", "#059669", "#d97706"])):
        p = np.exp(logits / T)
        p /= p.sum()
        ax.bar(np.arange(10) + (i - 1) * width, p, width=width, color=color, label=f"T = {T}")
    ax.set_xticks(range(10), words)
    ax.set_ylabel("P(next token)")
    ax.set_title("Temperature reshapes the same logits")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------ PS5 applied info theory ----
def cross_entropy_and_perplexity():
    import matplotlib.pyplot as plt
    import numpy as np

    p = np.linspace(0.01, 1.0, 300)
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))

    axes[0].plot(p, -np.log2(p), color="#2563eb", lw=2.4)
    axes[0].set_xlabel("probability given to the true token")
    axes[0].set_ylabel("bits of surprise")
    axes[0].set_title("−log p: the cost of being wrong", fontsize=11)

    ce = np.linspace(0, 5, 300)
    axes[1].plot(ce, 2 ** ce, color="#dc2626", lw=2.4)
    for bits, label in [(1, "coin flip"), (3.3, "1 in 10"), (4.7, "1 in 26")]:
        axes[1].plot(bits, 2 ** bits, "o", color="#111827", ms=6)
        axes[1].annotate(label, xy=(bits, 2 ** bits), xytext=(bits - 1.5, 2 ** bits + 3),
                         fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    axes[1].set_xlabel("cross-entropy (bits)"), axes[1].set_ylabel("perplexity")
    axes[1].set_title("perplexity = effective number of choices", fontsize=11)
    return f


# ------------------------------------------------------ RP5 multimodal -------
def multimodal_alignment_matrix():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(29)
    n, d = 6, 32
    text = rng.normal(size=(n, d))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    image = text + 0.55 * rng.normal(size=(n, d))
    image /= np.linalg.norm(image, axis=1, keepdims=True)

    sim = image @ text.T / 0.07
    sim = np.exp(sim - sim.max(1, keepdims=True))
    sim /= sim.sum(1, keepdims=True)

    f, ax = plt.subplots(figsize=(7.4, 5.1))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("captions"), ax.set_ylabel("images")
    ax.set_title("CLIP trains the diagonal to win each row and column")
    ax.set_xticks(range(n)), ax.set_yticks(range(n))
    ax.grid(False)
    f.colorbar(im, ax=ax, shrink=0.8)
    return f


# -------------------------------------------------------- ML3 SVM margin -----
def svm_margin_and_c():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(14)
    n = 40
    Xa = rng.normal([-1.6, -0.6], 0.75, (n, 2))
    Xb = rng.normal([1.6, 0.9], 0.75, (n, 2))
    X = np.vstack([Xa, Xb])
    y = np.hstack([-np.ones(n), np.ones(n)])

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    for ax, C, title in [(axes[0], 0.02, "small C: wide margin, some violations"),
                         (axes[1], 8.0, "large C: narrow margin, few violations")]:
        w, b = np.zeros(2), 0.0
        for epoch in range(400):                      # sub-gradient descent on hinge loss
            lr = 0.5 / (1 + epoch)
            margins = y * (X @ w + b)
            viol = margins < 1
            w -= lr * (w - C * (y[viol, None] * X[viol]).sum(0))
            b += lr * C * y[viol].sum()
        gx = np.linspace(-4, 4, 200)
        gy = -(w[0] * gx + b) / w[1]
        ax.scatter(*Xa.T, s=16, color="#dc2626"), ax.scatter(*Xb.T, s=16, color="#2563eb")
        ax.plot(gx, gy, color="#111827", lw=2)
        for off, ls in [(1, "--"), (-1, "--")]:
            ax.plot(gx, -(w[0] * gx + b - off) / w[1], color="#9ca3af", ls=ls, lw=1.4)
        ax.set_ylim(-3.4, 3.6), ax.set_xlim(-4, 4)
        ax.set_title(title, fontsize=10.5)
    return f


# --------------------------------------------------------- ML4 tree depth ----
def tree_depth_overfits():
    import matplotlib.pyplot as plt
    import numpy as np

    depths = np.arange(1, 17)
    train = 1 - np.exp(-depths / 2.6) * 0.55 - 0.02
    val = 0.72 + 0.22 * np.exp(-((depths - 4) ** 2) / 9.0) - 0.012 * np.maximum(0, depths - 6)

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(depths, train, "-o", color="#2563eb", lw=2.2, label="train accuracy")
    ax.plot(depths, val, "-o", color="#dc2626", lw=2.2, label="validation accuracy")
    best = depths[np.argmax(val)]
    ax.axvline(best, color="#059669", ls="--", lw=1.6)
    ax.annotate(f"best depth ≈ {best}", xy=(best, val.max()), xytext=(best + 1.6, 0.80),
                arrowprops=dict(arrowstyle="->", color="#059669"), color="#059669", fontsize=9.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("max_depth"), ax.set_ylabel("accuracy")
    ax.set_title("A deep tree reaches perfect training accuracy and stops generalising")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------------------ kNN ------
def knn_k_and_boundary():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(18)
    Xa = rng.normal([-1.0, 0.0], 0.95, (60, 2))
    Xb = rng.normal([1.4, 0.7], 0.95, (60, 2))
    X = np.vstack([Xa, Xb])
    y = np.hstack([np.zeros(60), np.ones(60)])

    g = np.linspace(-4, 4.5, 160)
    XX, YY = np.meshgrid(g, g)
    grid = np.stack([XX.ravel(), YY.ravel()], 1)

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, k in zip(axes, [1, 15, 61]):
        d = ((grid[:, None, :] - X[None]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, :k]
        pred = (y[nn].mean(1) > 0.5).reshape(XX.shape)
        ax.contourf(XX, YY, pred, levels=[-0.5, 0.5, 1.5], colors=["#fee2e2", "#dbeafe"])
        ax.scatter(*Xa.T, s=10, color="#dc2626")
        ax.scatter(*Xb.T, s=10, color="#2563eb")
        ax.set_title(f"k = {k}", fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    f.suptitle("Small k memorises the noise; large k smooths the boundary flat",
               fontsize=12.5, fontweight="bold")
    return f


# ---------------------------------------------------------------- DBSCAN -----
def density_clustering_shapes():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(22)
    t = rng.uniform(0, np.pi, 200)
    moon1 = np.stack([np.cos(t), np.sin(t)], 1) + rng.normal(0, 0.07, (200, 2))
    moon2 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1) + rng.normal(0, 0.07, (200, 2))
    X = np.vstack([moon1, moon2])

    def kmeans(X, k=2, iters=40):
        mu = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - mu[None]) ** 2).sum(-1).argmin(1)
            mu = np.array([X[lab == j].mean(0) for j in range(k)])
        return lab

    # DBSCAN by hand: grow clusters through dense neighbourhoods.
    eps, min_pts = 0.22, 5
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    core = (D <= eps).sum(1) >= min_pts
    labels = -np.ones(len(X), int)
    cid = 0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]:
            continue
        stack, labels[i] = [i], cid
        while stack:
            j = stack.pop()
            for n in np.flatnonzero(D[j] <= eps):
                if labels[n] == -1:
                    labels[n] = cid
                    if core[n]:
                        stack.append(n)
        cid += 1

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].scatter(*X.T, c=kmeans(X), cmap="coolwarm", s=10)
    axes[0].set_title("k-means: forced to cut with a straight edge", fontsize=11)
    axes[1].scatter(*X[labels >= 0].T, c=labels[labels >= 0], cmap="coolwarm", s=10)
    axes[1].scatter(*X[labels < 0].T, color="#9ca3af", s=10, marker="x")
    axes[1].set_title("DBSCAN: follows density, and labels noise", fontsize=11)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
    return f


# ------------------------------------------------------- weight decay --------
def weight_decay_shrinks_weights():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(33)
    lambdas = np.logspace(-4, 1, 40)
    n, p = 40, 30
    X = rng.normal(size=(n, p))
    w_true = rng.normal(size=p) * (rng.uniform(size=p) < 0.3)
    y = X @ w_true + rng.normal(0, 0.6, n)

    norms, train, test = [], [], []
    Xte = rng.normal(size=(400, p))
    yte = Xte @ w_true + rng.normal(0, 0.6, 400)
    for lam in lambdas:
        w = np.linalg.solve(X.T @ X + lam * n * np.eye(p), X.T @ y)
        norms.append(np.linalg.norm(w))
        train.append(np.mean((X @ w - y) ** 2))
        test.append(np.mean((Xte @ w - yte) ** 2))

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    axes[0].semilogx(lambdas, norms, color="#2563eb", lw=2.4)
    axes[0].set_xlabel("λ"), axes[0].set_ylabel("‖w‖")
    axes[0].set_title("the penalty pulls the weights in", fontsize=11)

    axes[1].semilogx(lambdas, train, color="#2563eb", lw=2.2, label="train")
    axes[1].semilogx(lambdas, test, color="#dc2626", lw=2.2, label="test")
    axes[1].axvline(lambdas[int(np.argmin(test))], color="#059669", ls="--", lw=1.5)
    axes[1].set_xlabel("λ"), axes[1].set_ylabel("MSE")
    axes[1].set_title("and there is a best amount of it", fontsize=11)
    axes[1].legend(fontsize=9)
    return f


# ------------------------------------------------------- receptive field -----
def receptive_field_growth():
    import matplotlib.pyplot as plt
    import numpy as np

    layers = np.arange(1, 13)
    k3 = 1 + 2 * layers                      # stacked 3x3, stride 1
    k5 = 1 + 4 * layers                      # stacked 5x5
    pooled = 2 ** layers                     # with a stride-2 pool each layer

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(layers, k3, "-o", color="#2563eb", lw=2.2, label="3×3 convs")
    ax.plot(layers, k5, "-o", color="#d97706", lw=2.2, label="5×5 convs")
    ax.semilogy(layers, pooled, "-o", color="#dc2626", lw=2.2, label="3×3 + stride-2 pooling")
    ax.set_xlabel("layers"), ax.set_ylabel("receptive field (pixels, log)")
    ax.set_title("Downsampling is what buys a global view cheaply")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------------- kernel PCA ----
def kernel_pca_unrolls():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(44)
    n = 150
    t1 = rng.uniform(0, 2 * np.pi, n)
    inner = np.stack([np.cos(t1), np.sin(t1)], 1) * (1 + 0.12 * rng.normal(size=(n, 1)))
    t2 = rng.uniform(0, 2 * np.pi, n)
    outer = np.stack([np.cos(t2), np.sin(t2)], 1) * (3 + 0.12 * rng.normal(size=(n, 1)))
    X = np.vstack([inner, outer])
    y = np.hstack([np.zeros(n), np.ones(n)])

    D = ((X[:, None] - X[None]) ** 2).sum(-1)
    K = np.exp(-D / (2 * 0.9 ** 2))
    one = np.ones_like(K) / len(K)
    Kc = K - one @ K - K @ one + one @ K @ one
    vals, vecs = np.linalg.eigh(Kc)
    proj = vecs[:, ::-1][:, :2] * np.sqrt(np.abs(vals[::-1][:2]))

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].scatter(*X[y == 0].T, s=12, color="#2563eb")
    axes[0].scatter(*X[y == 1].T, s=12, color="#dc2626")
    axes[0].set_title("no line separates two rings", fontsize=11)
    axes[0].set_aspect("equal")

    axes[1].scatter(proj[y == 0, 0], proj[y == 0, 1], s=12, color="#2563eb")
    axes[1].scatter(proj[y == 1, 0], proj[y == 1, 1], s=12, color="#dc2626")
    axes[1].set_title("RBF kernel PCA: now one does", fontsize=11)
    axes[1].set_xlabel("component 1"), axes[1].set_ylabel("component 2")
    return f


# ------------------------------------------------------------- t-SNE ---------
def tsne_perplexity_effect():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(55)
    sigma = np.linspace(0.05, 3.0, 300)
    d = np.array([0.2, 0.5, 0.9, 1.4, 2.2, 3.1, 4.0])

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    for s, color in zip([0.2, 0.6, 1.6], ["#dc2626", "#2563eb", "#059669"]):
        p = np.exp(-d ** 2 / (2 * s ** 2))
        p /= p.sum()
        H = -np.sum(p * np.log2(p + 1e-12))
        ax.plot(d, p, "-o", color=color, lw=2.2,
                label=f"σ = {s}  →  perplexity {2 ** H:.1f}")
    ax.set_xlabel("distance to neighbour")
    ax.set_ylabel("P(neighbour | point)")
    ax.set_title("Perplexity is how many neighbours each point really has")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------- non-convex landscape ------
def nonconvex_landscape():
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(-3.2, 3.2, 600)
    y = 0.4 * x ** 4 - 1.6 * x ** 2 + 0.4 * x + 2.0

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(x, y, color="#2563eb", lw=2.6)
    for x0, color, label in [(-2.6, "#dc2626", "starts left"), (2.6, "#059669", "starts right")]:
        p = x0
        path = [p]
        for _ in range(60):
            p -= 0.02 * (1.6 * p ** 3 - 3.2 * p + 0.4)
            path.append(p)
        path = np.array(path)
        ax.plot(path, 0.4 * path ** 4 - 1.6 * path ** 2 + 0.4 * path + 2.0,
                "o", ms=3.5, color=color, label=label)
    ax.set_title("Non-convex: where you land depends on where you started")
    ax.set_xlabel("parameter"), ax.set_ylabel("loss")
    ax.legend(fontsize=9)
    return f


# ---------------------------------------------- transformer families ---------
def transformer_family_shapes():
    import matplotlib.pyplot as plt
    import numpy as np

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    specs = [
        ("encoder-only\n(BERT)", np.ones((6, 6)), "#2563eb"),
        ("decoder-only\n(GPT)", np.tril(np.ones((6, 6))), "#dc2626"),
        ("encoder–decoder\n(T5)", np.block([[np.ones((3, 3)), np.zeros((3, 3))],
                                            [np.ones((3, 3)), np.tril(np.ones((3, 3)))]]),
         "#059669"),
    ]
    for ax, (title, mask, color) in zip(axes, specs):
        ax.imshow(mask, cmap="Blues", vmin=0, vmax=1.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("attends to", fontsize=9)
        ax.set_xticks(range(6)), ax.set_yticks(range(6))
        ax.grid(False)
    axes[0].set_ylabel("token", fontsize=9)
    f.suptitle("The family is decided by what each token is allowed to see",
               fontsize=12.5, fontweight="bold")
    return f


# ------------------------------------------------ seq2seq bottleneck ---------
def seq2seq_bottleneck():
    import matplotlib.pyplot as plt
    import numpy as np

    lengths = np.arange(5, 65, 5)
    no_attn = 32 * np.exp(-(lengths - 5) / 26) + 4
    with_attn = 30 - 0.03 * lengths

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(lengths, no_attn, "-o", color="#dc2626", lw=2.2,
            label="fixed-length context vector")
    ax.plot(lengths, with_attn, "-o", color="#2563eb", lw=2.2, label="with attention")
    ax.set_xlabel("source sentence length (tokens)")
    ax.set_ylabel("BLEU")
    ax.set_title("One vector cannot hold a long sentence")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------ collapse in SSL ------------
def representation_collapse():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(66)
    steps = np.arange(300)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    collapsed = 1.0 * np.exp(-steps / 40) + 0.01
    healthy = 0.35 + 0.6 * (1 - np.exp(-steps / 90))
    axes[0].plot(steps, collapsed, color="#dc2626", lw=2.4, label="no negatives / no stop-grad")
    axes[0].plot(steps, healthy, color="#059669", lw=2.4, label="with the collapse guard")
    axes[0].set_xlabel("step"), axes[0].set_ylabel("effective rank of the embedding")
    axes[0].set_title("collapse is a rank problem", fontsize=11)
    axes[0].legend(fontsize=8.5)

    for ax_data, color, label in [(0.02, "#dc2626", "collapsed"), (0.9, "#059669", "healthy")]:
        z = rng.normal(0, ax_data, (250, 2))
        axes[1].scatter(z[:, 0] + (0 if color == "#dc2626" else 0), z[:, 1],
                        s=10, color=color, alpha=0.55, label=label)
    axes[1].set_title("every input mapped to one point", fontsize=11)
    axes[1].set_xticks([]), axes[1].set_yticks([])
    axes[1].legend(fontsize=8.5)
    return f


# ------------------------------------------------ mutual information ---------
def mutual_information_bound():
    import matplotlib.pyplot as plt
    import numpy as np

    n_neg = np.array([1, 3, 7, 15, 31, 63, 127, 255, 511, 1023])
    bound = np.log2(n_neg + 1)

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.semilogx(n_neg + 1, bound, "-o", color="#2563eb", lw=2.4)
    ax.axhline(6, color="#dc2626", ls="--", lw=1.6)
    ax.text(2, 6.15, "true MI of the data: the bound has to reach it", color="#dc2626",
            fontsize=9.5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("batch size (1 positive + negatives)")
    ax.set_ylabel("InfoNCE bound on I(x; y)  (bits)")
    ax.set_title("InfoNCE can never report more than log N bits")
    return f


# ----------------------------------------------------------- PINN residual ---
def pinn_collocation_points():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(77)
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))

    pts = rng.uniform(0, 1, (220, 2))
    axes[0].scatter(pts[:, 0], pts[:, 1], s=9, color="#2563eb", label="collocation")
    edge = np.linspace(0, 1, 40)
    for a, b in [(edge, np.zeros_like(edge)), (edge, np.ones_like(edge)),
                 (np.zeros_like(edge), edge), (np.ones_like(edge), edge)]:
        axes[0].scatter(a, b, s=14, color="#dc2626")
    axes[0].set_title("residual sampled inside, data only on the edge", fontsize=11)
    axes[0].set_xlabel("x"), axes[0].set_ylabel("t")

    w = np.logspace(-2, 3, 60)
    pde_err = 0.9 / (1 + w ** 0.6) + 0.01
    bc_err = 0.01 + 0.0008 * w ** 0.8
    axes[1].loglog(w, pde_err, color="#2563eb", lw=2.2, label="PDE residual")
    axes[1].loglog(w, bc_err, color="#dc2626", lw=2.2, label="boundary error")
    axes[1].set_xlabel("weight on the PDE term")
    axes[1].set_title("the two terms trade against each other", fontsize=11)
    axes[1].legend(fontsize=9)
    return f


# -------------------------------------------------------- attention windows --
def sparse_attention_patterns():
    import matplotlib.pyplot as plt
    import numpy as np

    n = 24
    idx = np.arange(n)
    full = np.tril(np.ones((n, n)))
    window = full * (np.abs(idx[:, None] - idx[None]) < 5)
    glob = window.copy()
    glob[:, :2] = full[:, :2]
    strided = full * ((idx[:, None] - idx[None]) % 4 == 0)

    f, axes = plt.subplots(1, 4, figsize=(7.8, 2.3))
    for ax, m, t in [(axes[0], full, "full"), (axes[1], window, "sliding window"),
                     (axes[2], glob, "window + global"), (axes[3], strided, "strided")]:
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1.3)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    f.suptitle("Sparse patterns keep a fraction of the pairs and most of the quality",
               fontsize=12, fontweight="bold")
    return f


def linear_attention_associativity():
    import matplotlib.pyplot as plt
    import numpy as np

    # Start below n = d so the crossing point is actually on screen.
    n = np.logspace(1.3, 4.6, 200)
    d = 64
    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.loglog(n, n ** 2 * d, color="#dc2626", lw=2.4, label="(QKᵀ)V  —  O(n²d)")
    ax.loglog(n, n * d ** 2, color="#059669", lw=2.4, label="Q(KᵀV)  —  O(nd²)")
    cross = d
    ax.axvline(cross, color="#9ca3af", ls="--", lw=1.4)
    ax.text(cross * 1.35, 3e8, f"they cross at n = d = {d}", fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("sequence length n"), ax.set_ylabel("multiply-adds (log)")
    ax.set_title("Reassociating the product is the whole trick")
    ax.legend(fontsize=9)
    return f


def state_space_scan():
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.arange(0, 120)
    f, ax = plt.subplots(figsize=(7.2, 3.1))
    for a, color in [(0.999, "#059669"), (0.97, "#2563eb"), (0.85, "#dc2626")]:
        ax.plot(steps, a ** steps, lw=2.4, color=color, label=f"decay a = {a}")
    ax.set_xlabel("timesteps since the input")
    ax.set_ylabel("remaining contribution")
    ax.set_title("An SSM's memory is a decay rate you can learn")
    ax.legend(fontsize=9)
    return f


# ------------------------------------------------------------- RL details ----
def td_versus_monte_carlo():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(88)
    episodes = np.arange(1, 301)

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    for label, scale, decay, color in [("Monte Carlo (high variance)", 0.55, 260, "#dc2626"),
                                       ("TD(0) (biased, low variance)", 0.16, 90, "#2563eb"),
                                       ("TD(λ = 0.7)", 0.20, 70, "#059669")]:
        err = scale * np.exp(-episodes / decay) + 0.04 + rng.normal(0, scale * 0.06, 300)
        ax.plot(episodes, err, lw=1.8, color=color, label=label)
    ax.set_xlabel("episodes"), ax.set_ylabel("RMS value error")
    ax.set_title("Bootstrapping trades a little bias for much less variance")
    ax.legend(fontsize=9)
    return f


def policy_gradient_variance():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(99)
    batches = np.arange(1, 201)

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    for label, noise, color in [("REINFORCE, no baseline", 1.0, "#dc2626"),
                                ("with a value baseline", 0.35, "#2563eb"),
                                ("with GAE", 0.18, "#059669")]:
        ret = 100 * (1 - np.exp(-batches / 70)) + rng.normal(0, 26 * noise, 200)
        ax.plot(batches, ret, lw=1.6, color=color, alpha=0.9, label=label)
    ax.set_xlabel("update"), ax.set_ylabel("episode return")
    ax.set_title("Subtracting a baseline removes variance, not the gradient")
    ax.legend(fontsize=9)
    return f


def rlhf_kl_penalty():
    import matplotlib.pyplot as plt
    import numpy as np

    kl = np.linspace(0, 30, 300)
    f, ax = plt.subplots(figsize=(7.2, 3.1))
    true_quality = 4.2 * (1 - np.exp(-kl / 5.5)) - 0.06 * kl
    proxy_reward = 5.2 * (1 - np.exp(-kl / 4.0))
    ax.plot(kl, proxy_reward, color="#dc2626", lw=2.4, label="reward model score")
    ax.plot(kl, true_quality, color="#2563eb", lw=2.4, label="actual quality")
    best = kl[int(np.argmax(true_quality))]
    ax.axvline(best, color="#059669", ls="--", lw=1.6)
    ax.annotate("past here the policy is\ngaming the reward model",
                xy=(best, true_quality.max()), xytext=(best + 3, 1.6),
                arrowprops=dict(arrowstyle="->", color="#059669"), color="#059669", fontsize=9.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("KL from the reference policy"), ax.set_ylabel("score")
    ax.set_title("Why RLHF keeps a leash on the KL")
    ax.legend(fontsize=9)
    return f


# ----------------------------------------------------------- GNN details -----
def spectral_graph_filter():
    import matplotlib.pyplot as plt
    import numpy as np

    lam = np.linspace(0, 2, 300)
    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(lam, 1 - lam / 2, color="#2563eb", lw=2.4, label="low-pass (GCN)")
    ax.plot(lam, (1 - lam / 2) ** 3, color="#059669", lw=2.4, label="3 GCN layers stacked")
    ax.plot(lam, lam / 2, color="#dc2626", lw=2.4, label="high-pass")
    ax.set_xlabel("graph frequency λ (Laplacian eigenvalue)")
    ax.set_ylabel("gain")
    ax.set_title("A GCN layer is a low-pass filter, and stacking sharpens it")
    ax.legend(fontsize=9)
    return f


def oversmoothing_with_depth():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(101)
    n = 80
    A = (rng.uniform(size=(n, n)) < 0.06).astype(float)
    A = np.triu(A, 1)
    A = A + A.T + np.eye(n)
    D = np.diag(1 / A.sum(1))
    P = D @ A

    H = rng.normal(size=(n, 16))
    spread = []
    for _ in range(25):
        H = P @ H
        H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        spread.append(np.mean(np.linalg.norm(H - H.mean(0), axis=1)))

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(range(1, 26), spread, "-o", color="#dc2626", lw=2.2, ms=4)
    ax.set_xlabel("message-passing layers")
    ax.set_ylabel("spread of node features")
    ax.set_title("Oversmoothing: enough layers and every node looks the same")
    return f


def wl_test_limit():
    import matplotlib.pyplot as plt
    import numpy as np

    f, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    # Two 6-node graphs a message-passing GNN cannot tell apart: a hexagon and
    # two triangles. Every node has degree 2 in both.
    hexagon = [(i, (i + 1) % 6) for i in range(6)]
    triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    pos_hex = np.stack([np.cos(np.linspace(0, 2 * np.pi, 7)[:6]),
                        np.sin(np.linspace(0, 2 * np.pi, 7)[:6])], 1)
    pos_tri = np.array([[-1.5, 0.6], [-2.1, -0.5], [-0.9, -0.5],
                        [1.5, 0.6], [0.9, -0.5], [2.1, -0.5]])

    for ax, edges, pos, title in [(axes[0], hexagon, pos_hex, "one 6-cycle"),
                                  (axes[1], triangles, pos_tri, "two 3-cycles")]:
        for i, j in edges:
            ax.plot(*pos[[i, j]].T, color="#9ca3af", lw=2)
        ax.scatter(*pos.T, s=110, color="#2563eb", zorder=3)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    f.suptitle("Every node has degree 2 in both, so message passing cannot separate them",
               fontsize=11.5, fontweight="bold")
    return f


# ---------------------------------------------------- vision-language --------
def zero_shot_classification():
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["a photo of a cat", "a photo of a dog", "a photo of a car",
              "a photo of a plane", "a photo of a bird"]
    sims = np.array([0.31, 0.24, 0.09, 0.07, 0.19])
    logits = sims / 0.07
    p = np.exp(logits - logits.max())
    p /= p.sum()

    f, ax = plt.subplots(figsize=(7.2, 3.0))
    colors = ["#2563eb" if v == p.max() else "#cbd5e1" for v in p]
    ax.barh(labels, p, color=colors)
    ax.set_xlabel("P(caption | image)")
    ax.set_title("Zero-shot classification is just retrieval over prompts")
    ax.invert_yaxis()
    return f
def q_learning_vs_sarsa():
    import matplotlib.pyplot as plt
    import numpy as np

    # Cliff walk: SARSA learns the safe path, Q-learning the optimal risky one.
    rng = np.random.default_rng(7)
    episodes = np.arange(1, 401)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    sarsa = -40 - 55 * np.exp(-episodes / 70) + rng.normal(0, 4, 400)
    qlearn = -32 - 60 * np.exp(-episodes / 70) + rng.normal(0, 13, 400)
    axes[0].plot(episodes, qlearn, lw=1.2, color="#dc2626", alpha=0.8, label="Q-learning (off-policy)")
    axes[0].plot(episodes, sarsa, lw=1.2, color="#2563eb", alpha=0.8, label="SARSA (on-policy)")
    axes[0].set_xlabel("episode"), axes[0].set_ylabel("return")
    axes[0].set_title("SARSA earns more while still exploring", fontsize=11)
    axes[0].legend(fontsize=8.5)

    grid = np.zeros((4, 12))
    grid[3, 1:11] = -1                       # the cliff
    axes[1].imshow(grid, cmap="Reds_r", vmin=-1.4, vmax=0.4)
    axes[1].plot(range(0, 12), [2] * 12, "-o", color="#2563eb", ms=4, label="SARSA path")
    axes[1].plot(range(0, 12), [3] * 12, "-o", color="#dc2626", ms=4, label="greedy optimum")
    axes[1].set_title("the greedy path hugs the cliff", fontsize=11)
    axes[1].set_xticks([]), axes[1].set_yticks([])
    axes[1].grid(False)
    axes[1].legend(fontsize=8.5, loc="upper left")
    return f


# ------------------------------- replacements for hand-drawn diagrams --------
def threshold_precision_recall():
    import matplotlib.pyplot as plt
    import numpy as np

    scores = np.array([0.95, 0.88, 0.81, 0.74, 0.67, 0.45, 0.38, 0.29, 0.18, 0.05])
    labels = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0])

    thresholds = np.linspace(0.01, 0.99, 200)
    precision, recall = [], []
    for t in thresholds:
        pred = scores >= t
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / (tp + fn) if tp + fn else 0.0)

    f, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(thresholds, precision, color="#2563eb", lw=2.5, label="precision")
    ax.plot(thresholds, recall, color="#dc2626", lw=2.5, label="recall")
    ax.axvline(0.5, color="#111827", ls="--", lw=1.5)
    ax.annotate("threshold = 0.5", xy=(0.52, 0.12), color="#111827", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Raising the threshold trades recall for precision")
    ax.legend(loc="upper left", ncol=2)
    plt.show()


def accuracy_hides_failure():
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(["healthy", "sick"], [990, 10], color=["#2563eb", "#dc2626"])
    for x, v in zip([0, 1], [990, 10]):
        ax.text(x, v + 20, f"{v}", ha="center", fontsize=11)
    ax.set_ylabel("patients")
    ax.set_title('Predicting "healthy" for everyone: 99% accuracy, 0% recall')
    # Sits over the empty half of the axes, clear of both bars.
    ax.text(1, 620, "the model never predicts\nthe class the test exists to find",
            ha="center", fontsize=10, color="#dc2626")
    ax.set_ylim(0, 1120)
    plt.show()
def max_margin_hyperplane():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(3)
    A = rng.normal([-1.7, -0.9], 0.62, (24, 2))
    B = rng.normal([1.7, 1.0], 0.62, (24, 2))
    X, y = np.vstack([A, B]), np.hstack([-np.ones(24), np.ones(24)])

    w, b = np.zeros(2), 0.0
    for epoch in range(4000):                       # hinge loss, sub-gradient descent
        lr = 0.5 / (1 + epoch)
        margins = y * (X @ w + b)
        viol = margins < 1
        w -= lr * (w - 2.0 * (y[viol, None] * X[viol]).sum(0))
        b += lr * 2.0 * y[viol].sum()

    gx = np.linspace(-3.6, 3.6, 200)
    f, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(gx, -(w[0] * gx + b) / w[1], color="#111827", lw=2.2, label="decision boundary")
    for off, c in [(1, "#9ca3af"), (-1, "#9ca3af")]:
        ax.plot(gx, -(w[0] * gx + b - off) / w[1], color=c, ls="--", lw=1.5)

    sv = np.abs(y * (X @ w + b) - 1) < 0.12
    ax.scatter(*A.T, s=22, color="#dc2626")
    ax.scatter(*B.T, s=22, color="#2563eb")
    ax.scatter(*X[sv].T, s=140, facecolor="none", edgecolor="#111827", lw=1.8,
               label="support vectors")
    ax.set_xlim(-3.6, 3.6), ax.set_ylim(-3.2, 3.4)
    ax.set_title(f"Only the circled points set the boundary (margin {2/np.linalg.norm(w):.2f})")
    ax.legend(fontsize=9, loc="upper left")
    plt.show()


def primal_versus_dual():
    import matplotlib.pyplot as plt
    import numpy as np

    n_features = np.logspace(1, 5, 100)
    n_samples = 500
    f, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.loglog(n_features, n_features ** 2, color="#dc2626", lw=2.4,
              label="primal: solve in $d$ unknowns")
    ax.axhline(n_samples ** 2, color="#2563eb", lw=2.4,
               label="dual: solve in $n$ unknowns")
    ax.axvline(n_samples, color="#9ca3af", ls="--", lw=1.4)
    ax.text(n_samples * 1.25, 4e3, "d = n", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("feature dimension $d$")
    ax.set_ylabel("work (log)")
    ax.set_title("The dual replaces $d$ by $n$, so infinite-dimensional features become usable")
    ax.legend(fontsize=9)
    plt.show()


def rbf_infinite_expansion():
    import matplotlib.pyplot as plt
    import numpy as np

    r = np.linspace(0, 3.2, 300)
    gamma = 1.0
    exact = np.exp(-gamma * r ** 2)

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    partial = np.zeros_like(r)
    term = np.ones_like(r)
    for k in range(6):
        if k:
            term = term * (-gamma * r ** 2) / k
        partial = partial + term
        if k in (0, 1, 3, 5):
            axes[0].plot(r, partial, lw=1.8, label=f"up to degree {2*k}")
    axes[0].plot(r, exact, color="#111827", lw=2.4, ls="--", label="RBF kernel")
    axes[0].set_ylim(-1.2, 1.4)
    axes[0].set_xlabel("$\\|x-x'\\|$")
    axes[0].set_title("the series never terminates", fontsize=11)
    axes[0].legend(fontsize=8)

    degree = np.arange(0, 9)
    from math import factorial
    weight = (gamma ** degree) / np.array([factorial(int(d)) for d in degree])
    axes[1].bar(degree, weight, color="#2563eb")
    axes[1].set_xlabel("polynomial degree")
    axes[1].set_ylabel("weight in the expansion")
    axes[1].set_title("every degree is present, with decaying weight", fontsize=11)
    plt.show()


def rbf_gamma_overfits():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(9)
    n = 60
    t = rng.uniform(0, 2 * np.pi, n)
    X = np.vstack([np.stack([np.cos(t), np.sin(t)], 1) * (1 + 0.16 * rng.normal(size=(n, 1))),
                   np.stack([np.cos(t), np.sin(t)], 1) * (2.4 + 0.16 * rng.normal(size=(n, 1)))])
    y = np.hstack([-np.ones(n), np.ones(n)])

    g = np.linspace(-3.4, 3.4, 150)
    XX, YY = np.meshgrid(g, g)
    grid = np.stack([XX.ravel(), YY.ravel()], 1)

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.8))
    for ax, gamma in zip(axes, [0.15, 1.2, 26.0]):
        # kernel ridge in the dual: alpha = (K + lambda I)^-1 y
        D = ((X[:, None] - X[None]) ** 2).sum(-1)
        K = np.exp(-gamma * D)
        alpha = np.linalg.solve(K + 1e-3 * np.eye(len(X)), y)
        Dg = ((grid[:, None] - X[None]) ** 2).sum(-1)
        Z = (np.exp(-gamma * Dg) @ alpha).reshape(XX.shape)
        ax.contourf(XX, YY, Z > 0, levels=[-0.5, 0.5, 1.5], colors=["#fee2e2", "#dbeafe"])
        ax.scatter(*X[y < 0].T, s=8, color="#dc2626")
        ax.scatter(*X[y > 0].T, s=8, color="#2563eb")
        ax.set_title(f"γ = {gamma}", fontsize=11)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].set_ylabel("too smooth → underfit", fontsize=9)
    axes[2].set_ylabel("islands around points → overfit", fontsize=9)
    f.suptitle("γ sets how far one training point's influence reaches",
               fontsize=12.5, fontweight="bold")
    plt.show()


def decision_tree_splits_space():
    import matplotlib.pyplot as plt
    import numpy as np

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))

    # Left: the tree as a sequence of threshold questions.
    nodes = {"root": (0.5, 1.0), "L": (0.25, 0.62), "R": (0.75, 0.62),
             "LL": (0.12, 0.24), "LR": (0.38, 0.24), "RL": (0.62, 0.24), "RR": (0.88, 0.24)}
    for a, b in [("root", "L"), ("root", "R"), ("L", "LL"), ("L", "LR"),
                 ("R", "RL"), ("R", "RR")]:
        axes[0].plot(*zip(nodes[a], nodes[b]), color="#9ca3af", lw=1.6, zorder=1)
    for key, label, colour in [("root", "$x_1 \\leq 3$", "#111827"), ("L", "$x_2 \\leq 2$", "#111827"),
                               ("R", "$x_2 \\leq 4$", "#111827"), ("LL", "A", "#2563eb"),
                               ("LR", "B", "#dc2626"), ("RL", "B", "#dc2626"), ("RR", "A", "#2563eb")]:
        x, y = nodes[key]
        axes[0].add_patch(plt.Circle((x, y), 0.085, facecolor="white", edgecolor=colour, lw=2, zorder=2))
        axes[0].text(x, y, label, ha="center", va="center", fontsize=9.5, zorder=3)
    axes[0].set_xlim(0, 1), axes[0].set_ylim(0.05, 1.2)
    axes[0].axis("off"), axes[0].grid(False)
    axes[0].set_title("a sequence of threshold questions", fontsize=11)

    # Right: the same tree as axis-aligned rectangles.
    axes[1].add_patch(plt.Rectangle((0, 0), 3, 2, facecolor="#dbeafe"))
    axes[1].add_patch(plt.Rectangle((0, 2), 3, 4, facecolor="#fee2e2"))
    axes[1].add_patch(plt.Rectangle((3, 0), 3, 4, facecolor="#fee2e2"))
    axes[1].add_patch(plt.Rectangle((3, 4), 3, 2, facecolor="#dbeafe"))
    axes[1].axvline(3, color="#111827", lw=2)
    axes[1].plot([0, 3], [2, 2], color="#111827", lw=2)
    axes[1].plot([3, 6], [4, 4], color="#111827", lw=2)
    axes[1].set_xlim(0, 6), axes[1].set_ylim(0, 6)
    axes[1].set_xlabel("$x_1$"), axes[1].set_ylabel("$x_2$")
    axes[1].set_title("axis-aligned rectangles", fontsize=11)
    plt.show()


def split_purity_gain():
    import matplotlib.pyplot as plt
    import numpy as np

    def gini(p):
        return 1 - p ** 2 - (1 - p) ** 2

    def entropy(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    parent = (6, 6)                       # 6 of each class
    splits = {"useless split": ((3, 3), (3, 3)), "useful split": ((5, 1), (1, 5)),
              "perfect split": ((6, 0), (0, 6))}

    f, ax = plt.subplots(figsize=(7.2, 3.2))
    names, gains = [], []
    for name, (l, r) in splits.items():
        pp = parent[0] / sum(parent)
        nl, nr = sum(l), sum(r)
        child = (nl * entropy(l[0] / nl) + nr * entropy(r[0] / nr)) / (nl + nr)
        names.append(name)
        gains.append(entropy(pp) - child)
    ax.bar(names, gains, color=["#9ca3af", "#2563eb", "#059669"])
    for i, g in enumerate(gains):
        ax.text(i, g + 0.02, f"{g:.2f} bits", ha="center", fontsize=10)
    ax.set_ylabel("information gain")
    ax.set_ylim(0, 1.18)
    ax.set_title("A split is worth making only if the children are purer than the parent")
    plt.show()


def classification_versus_regression_tree():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(12)
    x = np.sort(rng.uniform(0, 10, 90))
    y = np.sin(x) + 0.28 * rng.normal(size=x.size)

    def fit(x, y, depth):
        # Piecewise-constant fit: split at the point minimising total squared error.
        # Needs 6 points: the split loop below scans range(2, len-2).
        if depth == 0 or len(x) < 6:
            return np.full_like(x, y.mean())
        best, cut = None, None
        for i in range(2, len(x) - 2):
            sse = ((y[:i] - y[:i].mean()) ** 2).sum() + ((y[i:] - y[i:].mean()) ** 2).sum()
            if best is None or sse < best:
                best, cut = sse, i
        return np.concatenate([fit(x[:cut], y[:cut], depth - 1), fit(x[cut:], y[cut:], depth - 1)])

    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1))
    axes[0].scatter(x, y, s=12, color="#9ca3af")
    axes[0].step(x, fit(x, y, 3), where="mid", color="#2563eb", lw=2.2)
    axes[0].set_title("regression tree: leaf predicts the mean", fontsize=11)

    labels = (y > 0).astype(int)
    axes[1].scatter(x, labels + 0.04 * rng.normal(size=x.size), s=12,
                    c=["#dc2626" if v == 0 else "#2563eb" for v in labels])
    axes[1].step(x, (fit(x, labels.astype(float), 3) > 0.5).astype(float), where="mid",
                 color="#059669", lw=2.2)
    axes[1].set_yticks([0, 1])
    axes[1].set_title("classification tree: leaf predicts the majority", fontsize=11)
    plt.show()


def bagging_reduces_variance():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(21)
    x = np.linspace(0, 10, 120)
    truth = np.sin(x)

    def stump_fit(xs, ys, depth=3):
        # Needs 6 points: the split loop below scans range(2, len-2).
        if depth == 0 or len(xs) < 6:
            return np.full_like(xs, ys.mean())
        best, cut = None, None
        for i in range(2, len(xs) - 2):
            sse = ((ys[:i] - ys[:i].mean()) ** 2).sum() + ((ys[i:] - ys[i:].mean()) ** 2).sum()
            if best is None or sse < best:
                best, cut = sse, i
        return np.concatenate([stump_fit(xs[:cut], ys[:cut], depth - 1),
                               stump_fit(xs[cut:], ys[cut:], depth - 1)])

    preds = []
    f, axes = plt.subplots(1, 2, figsize=(7.8, 3.1), sharey=True)
    for b in range(12):
        idx = np.unique(np.sort(rng.integers(0, len(x), len(x))))   # bootstrap sample
        xb, yb = x[idx], truth[idx] + 0.35 * rng.normal(size=len(idx))
        p = np.interp(x, xb, stump_fit(xb, yb))
        preds.append(p)
        axes[0].plot(x, p, color="#9ca3af", lw=0.9, alpha=0.8)
    axes[0].plot(x, truth, color="#111827", lw=2, ls="--")
    axes[0].set_title("12 trees on 12 bootstrap samples", fontsize=11)

    axes[1].plot(x, np.mean(preds, 0), color="#2563eb", lw=2.4, label="average of the 12")
    axes[1].plot(x, truth, color="#111827", lw=2, ls="--", label="truth")
    axes[1].set_title("their average", fontsize=11)
    axes[1].legend(fontsize=9)
    plt.show()


def random_forest_decorrelates():
    import matplotlib.pyplot as plt
    import numpy as np

    rho = np.linspace(0, 1, 200)
    B = 100
    var = rho + (1 - rho) / B                 # variance of the mean of B correlated trees

    f, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(rho, var, color="#2563eb", lw=2.6)
    for r, name, c in [(0.85, "bagging: same features every split", "#dc2626"),
                       (0.35, "random forest: random feature subset", "#059669")]:
        v = r + (1 - r) / B
        ax.plot(r, v, "o", ms=9, color=c)
        ax.annotate(name, xy=(r, v), xytext=(r - 0.32, v + 0.16),
                    arrowprops=dict(arrowstyle="->", color=c), color=c, fontsize=9.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlabel("correlation ρ between trees")
    ax.set_ylabel("variance of the ensemble average")
    ax.set_title("Averaging only helps to the extent the trees disagree")
    plt.show()


def boosting_sequential_residuals():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(5)
    x = np.linspace(0, 10, 160)
    y = np.sin(x) + 0.12 * rng.normal(size=x.size)

    def stump(xs, res):
        best = None
        for i in range(4, len(xs) - 4):
            sse = ((res[:i] - res[:i].mean()) ** 2).sum() + ((res[i:] - res[i:].mean()) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, i)
        i = best[1]
        return np.concatenate([np.full(i, res[:i].mean()), np.full(len(xs) - i, res[i:].mean())])

    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.8), sharey=True)
    pred = np.zeros_like(x)
    shots = {1: axes[0], 8: axes[1], 60: axes[2]}
    for m in range(1, 61):
        pred = pred + 0.3 * stump(x, y - pred)
        if m in shots:
            ax = shots[m]
            ax.scatter(x, y, s=7, color="#cbd5e1")
            ax.plot(x, pred, color="#2563eb", lw=2.2)
            ax.set_title(f"{m} tree{'s' if m > 1 else ''}", fontsize=11)
    f.suptitle("Each tree is fitted to what the ensemble still gets wrong",
               fontsize=12.5, fontweight="bold")
    plt.show()


def adaboost_reweights():
    import matplotlib.pyplot as plt
    import numpy as np

    rng = np.random.default_rng(8)
    n = 14
    x = np.arange(n)
    wrong = np.zeros(n, bool)
    wrong[[3, 7, 11]] = True

    w = np.full(n, 1 / n)
    f, axes = plt.subplots(1, 3, figsize=(7.8, 2.6), sharey=True)
    for r, ax in enumerate(axes):
        ax.bar(x, w, color=["#dc2626" if m else "#cbd5e1" for m in wrong])
        ax.set_title(f"round {r + 1}", fontsize=11)
        ax.set_xticks([])
        err = w[wrong].sum()
        alpha = 0.5 * np.log((1 - err) / err)
        w = w * np.exp(np.where(wrong, alpha, -alpha))
        w = w / w.sum()
    axes[0].set_ylabel("example weight")
    f.suptitle("Misclassified examples gain weight, so the next tree focuses on them",
               fontsize=12, fontweight="bold")
    plt.show()
# ---------------------------------------------------------------- registry ---
# (module file, anchor, slug, alt text, draw function)
# An anchor of ("## Section", n) puts the figure after the nth paragraph of that
# section, so it breaks up the prose instead of sitting under the heading.
FIGURES = [
    ("1.1-PY1-programming-essentials.md", ("## The array as strided memory", 4),
     "py1-broadcasting", "Broadcasting stretches a smaller array without copying it",
     broadcasting_rules),
    ("1.1-PY1-programming-essentials.md", ("## Intuition", 2),
     "py1-view-vs-copy", "A slice writes through to the original array; fancy indexing does not",
     view_versus_copy),

    ("1.2-PY2-pandas-data-cleaning-storing.md", ("## The DataFrame data model", 4),
     "py2-split-apply-combine", "Split rows by key, apply a function, combine to one row per group",
     split_apply_combine),

    ("1.3-PY3-plotting-matplotlib-seaborn.md", ("## Figure, axes, and artists", 3),
     "py3-distribution-shapes", "Mean and median agree only when the distribution is symmetric",
     distribution_shapes),

    ("1.5-PY5-scikit-learn-api-ml-pipelines.md", ("## The estimator contract", 4),
     "py5-leakage", "Fitting a transform outside the fold invents accuracy that is not there",
     leakage_vs_pipeline),

    ("1.6-PY6-pytorch-essentials.md", ("## Tensors", 4),
     "py6-autograd-loss", "The loss autograd minimises, and the fit it produces",
     autograd_graph_and_loss),

    ("1.7-MA1-linear-algebra-i-vectors-matrices-systems-transformations.md",
     ("## Vectors, Norms, and Projection", 4),
     "ma1-linear-map", "A matrix is defined by what it does to the unit square",
     linear_map_unit_square),
    ("1.7-MA1-linear-algebra-i-vectors-matrices-systems-transformations.md",
     ("## Matrices, Determinants, and Eigenvalues", 5),
     "ma1-eigenvectors", "Eigenvectors keep their direction under the map",
     eigenvectors_are_fixed_directions),

    ("1.8-MA2-linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory.md",
     ("## Linear maps and the matrix that represents them", 4),
     "ma2-eigen-directions", "The unit circle becomes an ellipse whose axes are the eigenvectors",
     eigenvectors_are_fixed_directions),
    ("1.8-MA2-linear-algebra-ii-eigenvalues-eigenvectors-spectral-theory.md",
     ("## Intuition", 2),
     "ma2-svd-lowrank", "Truncating the small singular values keeps most of the matrix",
     svd_low_rank_approximation),

    ("1.9-MA3-calculus-gradients.md", ("## Calculus for Learning", 5),
     "ma3-descent-path", "The steepest direction sets the largest step that stays stable",
     gradient_descent_path),
    ("1.9-MA3-calculus-gradients.md", ("## Matrix Calculus", 4),
     "ma3-gradient-flow", "Depth multiplies the per-layer scale, so gradients vanish or explode",
     gradient_flow_by_depth),

    ("1.10-PS1-probability-foundations.md", ("## Axioms and immediate consequences", 4),
     "ps1-bayes-update", "Each batch of evidence concentrates the posterior",
     bayes_posterior_update),

    ("1.11-PS2-expectations-multivariate-distributions.md", ("## Probability", 5),
     "ps2-lln-clt", "The running mean settles, and the distribution of means turns Gaussian",
     law_of_large_numbers),
    ("1.11-PS2-expectations-multivariate-distributions.md", ("## Statistics and Sampling", 4),
     "ps2-sampling-ci", "Sampling distribution of the mean and the coverage of 95% intervals",
     sampling_distribution_and_ci),

    ("1.12-PS3-statistics-hypothesis-testing.md", ("## Populations, samples, estimators", 4),
     "ps3-sampling-ci", "About 95% of the intervals contain the true mean",
     sampling_distribution_and_ci),

    ("1.13-ML1-data-processing-features.md", ("## Missing data", 4),
     "ml1-scaling", "Without scaling, the feature with the largest units dominates distance",
     scaling_changes_distance),

    ("1.14-ML2-supervised-learning-foundations.md", ("## Linear Regression", 5),
     "ml2-bias-variance", "Too simple misses the signal; too flexible fits the noise",
     bias_variance_tradeoff),
    ("1.14-ML2-supervised-learning-foundations.md", ("## Logistic Regression", 4),
     "ml2-logistic", "The sigmoid turns a score into a probability, and log loss punishes confident errors",
     logistic_decision_boundary),

    ("1.15-ML3-kernel-methods-svms.md", ("## Soft Margin SVM", 3),
     "ml3-margin-c", "C sets how much margin the model trades for fewer violations",
     svm_margin_and_c),

    ("1.16-ML4-tree-based-models-ensembles.md", ("## Overfitting + Regularization", 3),
     "ml4-depth", "Deeper trees reach perfect training accuracy and stop generalising",
     tree_depth_overfits),

    ("1.17-ML5-clustering.md", ("## k-Means Clustering", 4),
     "ml5-kmeans-iterations", "Assign, move, repeat until the assignment stops changing",
     kmeans_iterations),
    ("1.17-ML5-clustering.md", ("## Hierarchical Clustering", 4),
     "ml5-choosing-k", "Inertia falls with every extra cluster, so the elbow is a judgement call",
     choosing_k_silhouette),

    ("1.18-DL1-neural-networks-perceptrons-mlps-activations.md", ("## The forward pass", 4),
     "dl1-activations", "Each activation and the gradient it passes back",
     activation_functions),
    ("1.18-DL1-neural-networks-perceptrons-mlps-activations.md", ("## Intuition", 2),
     "dl1-xor", "A hidden layer bends the space until XOR becomes separable",
     xor_needs_a_hidden_layer),

    ("1.19-DL2-neural-network-regularization-dropout-batchnorm.md", ("## Regularization", 4),
     "dl2-curves", "Regularization trades training loss for a smaller train/validation gap",
     regularization_curves),

    ("1.20-DL3-model-training.md", ("## Bias, Variance, and Model Selection", 4),
     "dl3-learning-curves", "Learning curves tell high bias apart from high variance",
     learning_curves_diagnosis),
    ("1.20-DL3-model-training.md", ("## Learning-Rate Schedules", 3),
     "dl3-schedules", "Warmup protects the first few hundred steps",
     learning_rate_schedules),

    ("1.21-DL4-optimization-adam-sgd.md", ("## Gradient descent and its variants", 4),
     "dl4-optimizers", "In a narrow valley, plain SGD zig-zags where momentum and Adam do not",
     optimizer_trajectories),

    ("1.22-CV1-convolutional-neural-networks.md", ("## The convolution operation", 4),
     "cv1-convolution", "One kernel, slid everywhere, finds the same feature anywhere",
     convolution_operation),

    ("2.1-MA4-backpropagation-mechanics.md", ("## Matrix Calculus", 4),
     "ma4-gradient-flow", "The chain rule multiplies, so depth compounds the scale",
     gradient_flow_by_depth),

    ("2.2-MA5-advanced-linear-algebra-projections-svd-low-rank.md",
     ("## The Singular Value Decomposition", 4),
     "ma5-lowrank", "A few singular values carry most of the matrix",
     svd_low_rank_approximation),

    ("2.3-ML6-model-evaluation.md", ("## ROC-AUC", 3),
     "ml6-roc-pr", "The same model looks strong on ROC and weak on PR under imbalance",
     roc_and_pr_under_imbalance),
    ("2.3-ML6-model-evaluation.md", ("## Calibration", 3),
     "ml6-calibration", "A reliability diagram compares the predicted probability to reality",
     calibration_reliability),

    ("2.4-ML7-dimensionality-reduction.md", ("## Principal Component Analysis", 4),
     "ml7-pca", "A handful of components carry nearly all the variance",
     pca_variance_explained),

    ("2.5-DL5-advanced-optimization.md", ("## Optimizers", 3),
     "dl5-optimizers", "Adam adapts its step per parameter; SGD does not",
     optimizer_trajectories),
    ("2.5-DL5-advanced-optimization.md", ("## Learning-Rate Schedules", 3),
     "dl5-schedules", "Warmup then cosine decay, against a constant rate",
     learning_rate_schedules),

    ("2.6-CV2-computer-vision-classification-augmentation.md",
     ("## Convolutional Neural Networks", 4),
     "cv2-augmentation", "Augmentation asserts that the label survives each transform",
     augmentation_transforms),

    ("2.7-CV3-semantic-segmentation-unet-masks.md", ("## The task family", 3),
     "cv3-iou-dice", "IoU and Dice on the same predicted mask",
     segmentation_metrics),


    ("2.9-TF1-transformer-fundamentals.md", ("## Attention Mechanisms", 4),
     "tf1-attention-heatmap", "Attention can reach past nearby words to the right one",
     attention_heatmap),
    ("2.9-TF1-transformer-fundamentals.md", ("## The Transformer Block", 3),
     "tf1-cost", "Full attention costs O(n squared) in the sequence length",
     attention_cost_scaling),

    ("2.10-TF2-transformer-architectures-bert-mlm.md",
     ("## Transformer Architecture Families", 3),
     "tf2-masks", "BERT sees the whole sentence; GPT sees only the past",
     causal_versus_bidirectional_mask),

    ("2.11-TF3-transformer-architectures-gpt-autoregressive.md",
     ("## Language Modelling and Decoding", 4),
     "tf3-temperature", "Temperature reshapes the same logits before sampling",
     sampling_temperature),

    ("2.12-TF4-transformer-architectures-t5-encoder-decoder.md",
     ("## Sequence-to-Sequence and Attention", 4),
     "tf4-cross-attention", "Cross-attention learns the alignment between source and target",
     encoder_decoder_attention),

    ("2.13-RP1-representation-learning-autoencoders-vaes-latent-spaces.md",
     ("## Autoencoders", 4),
     "rp1-bottleneck", "The bottleneck forces the network to keep only what matters",
     autoencoder_bottleneck),

    ("2.14-RP2-contrastive-learning.md", ("## Contrastive Learning", 4),
     "rp2-embedding-space", "Contrastive training pulls positives together on the sphere",
     contrastive_embedding_space),
    ("2.14-RP2-contrastive-learning.md", ("## Non-Contrastive Methods", 3),
     "rp2-temperature", "Low temperature concentrates the loss on the hardest negatives",
     infonce_temperature),

    ("3.1-PS4-information-theory-core.md", ("## Entropy", 4),
     "ps4-entropy", "Entropy peaks when the outcome is least predictable",
     entropy_of_a_coin),
    ("3.1-PS4-information-theory-core.md", ("## Intuition", 2),
     "ps4-kl", "Forward KL punishes a model that misses a mode",
     kl_divergence_asymmetry),


    ("3.4-NL1-natural-language-processing-tokenization-embeddings.md", ("## Tokenization", 4),
     "nl1-zipf", "A few tokens cover most text and the tail is enormous",
     token_frequency_zipf),

    ("3.5-NL2-sequence-models-transfer-learning.md", ("## Recurrent Neural Networks", 4),
     "nl2-horizon", "How far back a recurrent gradient survives",
     rnn_gradient_horizon),
    ("3.5-NL2-sequence-models-transfer-learning.md",
     ("## Sequence-to-Sequence and Attention", 4),
     "nl2-cross-attention", "Attention gives the decoder a direct look at every source token",
     encoder_decoder_attention),

    ("3.6-TF5-rag-retrieval-embeddings.md", ("## The RAG pipeline", 4),
     "tf5-retrieval", "Retrieval keeps the top-k documents; the rest never reach the model",
     retrieval_similarity_ranking),

    ("3.7-TF6-relative-position-rope-positional-methods.md", ("## RoPE", 4),
     "tf6-rope", "RoPE makes the attention score depend only on the distance between tokens",
     rope_relative_decay),

    ("3.8-TF7-cross-attention-multimodal-attention.md", ("## Attention Mechanisms", 4),
     "tf7-cross-attention", "Cross-attention aligns one sequence against another",
     encoder_decoder_attention),
    ("3.8-TF7-cross-attention-multimodal-attention.md", ("## Multimodal Alignment", 3),
     "tf7-alignment", "Contrastive training trains the diagonal to win each row and column",
     multimodal_alignment_matrix),

    ("3.9-RP4-autoregressive-generative-models.md",
     ("## Language Modelling and Decoding", 4),
     "rp4-temperature", "Temperature reshapes the next-token distribution",
     sampling_temperature),


    ("3.11-GM1-generative-adversarial-networks.md",
     ("## Generative Adversarial Networks", 4),
     "gm1-dynamics", "Neither loss falls: the two networks chase each other",
     gan_training_dynamics),

    ("3.12-GM2-diffusion-models.md", ("## Diffusion Models", 4),
     "gm2-forward", "Forward diffusion dissolves structure into noise on a fixed schedule",
     diffusion_forward_process),

    ("3.13-XT1-physics-informed-neural-networks.md",
     ("## Physics-Informed Neural Networks", 4),
     "xt1-residual", "A PINN is fit to the equation itself, not to sampled data",
     pinn_residual_loss),

    ("4.1-PS5-applied-information-theory.md", ("## Information Theory", 4),
     "ps5-perplexity", "Cross-entropy in bits, and the perplexity it corresponds to",
     cross_entropy_and_perplexity),
    ("4.1-PS5-applied-information-theory.md", ("## Self-Supervised Learning", 3),
     "ps5-infonce", "Temperature controls how sharply InfoNCE weights the negatives",
     infonce_temperature),

    ("4.2-TF8-gqa-mla-attention-complexity.md", ("## KV Cache and Attention Variants", 4),
     "tf8-kv-cache", "Sharing KV heads is what makes a long context affordable",
     kv_cache_growth),

    ("4.3-TF9-efficient-attention-long-context.md",
     ("## Sparse and Long-Context Attention", 3),
     "tf9-cost", "Why long context forces a different attention pattern",
     attention_cost_scaling),
    ("4.3-TF9-efficient-attention-long-context.md", ("## FlashAttention", 4),
     "tf9-kv-cache", "The KV cache, not the weights, is what fills memory at long context",
     kv_cache_growth),

    ("4.4-RP5-multimodal-learning.md", ("## Multimodal Alignment", 4),
     "rp5-alignment", "CLIP trains the matching pair to win its row and its column",
     multimodal_alignment_matrix),

    ("4.5-RF1-reinforcement-learning.md", ("## Markov Decision Processes", 4),
     "rf1-value-iteration", "Value propagates back from the reward, one sweep at a time",
     value_iteration_convergence),
    ("4.5-RF1-reinforcement-learning.md", ("## Temporal-Difference Learning", 4),
     "rf1-exploration", "Pure greed locks onto the first arm that looks good",
     exploration_tradeoff),

    ("4.6-XT2-graph-neural-networks.md", ("## Message-Passing Neural Networks", 4),
     "xt2-receptive-field", "Each message-passing layer widens the receptive field by one hop",
     message_passing_receptive_field),
    # --- second pass: one figure per remaining wall of prose -----------------
    ("1.14-ML2-supervised-learning-foundations.md", ("## k-Nearest Neighbours", "wall"),
     "ml2-knn", "Small k memorises the noise; large k flattens the boundary",
     knn_k_and_boundary),
    ("1.17-ML5-clustering.md", ("## Density-Based Clustering", "wall"),
     "ml5-dbscan", "Density-based clustering follows shapes k-means cannot cut",
     density_clustering_shapes),
    ("1.19-DL2-neural-network-regularization-dropout-batchnorm.md", ("## Regularization", "wall"),
     "dl2-weight-decay", "Weight decay shrinks the weights, and there is a best amount of it",
     weight_decay_shrinks_weights),
    ("1.20-DL3-model-training.md", ("## Regularization", "wall"),
     "dl3-weight-decay", "The penalty pulls the weights in, and test error has a minimum",
     weight_decay_shrinks_weights),
    ("1.22-CV1-convolutional-neural-networks.md", ("## Intuition", "wall"),
     "cv1-receptive-field", "Downsampling is what buys a global receptive field cheaply",
     receptive_field_growth),
    ("2.4-ML7-dimensionality-reduction.md", ("## Kernel PCA", "wall"),
     "ml7-kernel-pca", "An RBF kernel unrolls two rings into something linearly separable",
     kernel_pca_unrolls),
    ("2.4-ML7-dimensionality-reduction.md", ("## Manifold Learning and t-SNE", "wall"),
     "ml7-tsne", "Perplexity sets how many neighbours each point is given",
     tsne_perplexity_effect),
    ("2.5-DL5-advanced-optimization.md", ("## Stochastic and Non-Convex Optimization", "wall"),
     "dl5-nonconvex", "In a non-convex landscape, the basin you land in depends on where you start",
     nonconvex_landscape),
    ("2.6-CV2-computer-vision-classification-augmentation.md",
     ("## Convolutional Neural Networks", "wall"),
     "cv2-receptive-field", "Stacking convolutions grows the receptive field slowly; pooling grows it fast",
     receptive_field_growth),
    ("2.8-CV4-modern-vision-architectures-resnet-efficientnet-transfer.md",
     ("## Hierarchical Vision Architectures", "wall"),
     "cv4-receptive-field", "How quickly each design reaches a global view",
     receptive_field_growth),
    ("2.12-TF4-transformer-architectures-t5-encoder-decoder.md",
     ("## Sequence-to-Sequence and Attention", "wall"),
     "tf4-bottleneck", "One fixed-length vector cannot hold a long sentence",
     seq2seq_bottleneck),
    ("2.14-RP2-contrastive-learning.md", ("## Contrastive Learning", "wall"),
     "rp2-mi-bound", "InfoNCE can never report more than log N bits of mutual information",
     mutual_information_bound),
    ("2.14-RP2-contrastive-learning.md", ("## Self-Supervised Learning", "wall"),
     "rp2-collapse", "Collapse is a rank problem: every input maps to the same point",
     representation_collapse),
    ("3.1-PS4-information-theory-core.md", ("## Intuition", "wall"),
     "ps4-perplexity", "Cross-entropy in bits and the perplexity it corresponds to",
     cross_entropy_and_perplexity),
    ("3.8-TF7-cross-attention-multimodal-attention.md", ("## Multimodal Alignment", "wall"),
     "tf7-collapse", "Without a guard, both towers collapse onto one point",
     representation_collapse),
    ("3.8-TF7-cross-attention-multimodal-attention.md", ("## Vision\u2013Language Models", "wall"),
     "tf7-zeroshot", "Zero-shot classification is retrieval over prompt strings",
     zero_shot_classification),
    ("3.13-XT1-physics-informed-neural-networks.md",
     ("## Physics-Informed Neural Networks", "wall"),
     "xt1-collocation", "The residual is sampled inside the domain; data only pins the boundary",
     pinn_collocation_points),
    ("4.1-PS5-applied-information-theory.md", ("## Information Theory", "wall"),
     "ps5-kl", "Forward KL punishes a model that drops a mode",
     kl_divergence_asymmetry),
    ("4.1-PS5-applied-information-theory.md", ("## Self-Supervised Learning", "wall"),
     "ps5-mi-bound", "The InfoNCE bound is capped by the batch size",
     mutual_information_bound),
    ("4.2-TF8-gqa-mla-attention-complexity.md", ("## KV Cache and Attention Variants", "wall"),
     "tf8-cost", "Attention cost against sequence length",
     attention_cost_scaling),
    ("4.3-TF9-efficient-attention-long-context.md",
     ("## Sparse and Long-Context Attention", "wall"),
     "tf9-sparse", "Sparse patterns keep a fraction of the pairs and most of the quality",
     sparse_attention_patterns),
    ("4.3-TF9-efficient-attention-long-context.md", ("## Linear Attention", "wall"),
     "tf9-linear", "Reassociating the product is the whole trick behind linear attention",
     linear_attention_associativity),
    ("4.3-TF9-efficient-attention-long-context.md", ("## State-Space Models", "wall"),
     "tf9-ssm", "An SSM's memory is a decay rate it can learn",
     state_space_scan),
    ("4.4-RP5-multimodal-learning.md", ("## Multimodal Alignment", "wall"),
     "rp5-collapse", "Contrastive training is what stops the two towers collapsing",
     representation_collapse),
    ("4.4-RP5-multimodal-learning.md", ("## Vision\u2013Language Models", "wall"),
     "rp5-zeroshot", "Zero-shot classification scores an image against prompt strings",
     zero_shot_classification),
    ("4.5-RF1-reinforcement-learning.md", ("## Temporal-Difference Learning", "wall"),
     "rf1-td-mc", "Bootstrapping trades a little bias for much less variance",
     td_versus_monte_carlo),
    ("4.5-RF1-reinforcement-learning.md", ("## Policy Gradient Methods", "wall"),
     "rf1-baseline", "Subtracting a baseline removes variance without biasing the gradient",
     policy_gradient_variance),
    ("4.5-RF1-reinforcement-learning.md", ("## RLHF and Preference Optimization", "wall"),
     "rf1-rlhf-kl", "Past a point the policy games the reward model instead of improving",
     rlhf_kl_penalty),
    ("4.6-XT2-graph-neural-networks.md", ("## Message-Passing Neural Networks", "wall"),
     "xt2-oversmoothing", "Enough layers and every node ends up looking the same",
     oversmoothing_with_depth),
    ("4.6-XT2-graph-neural-networks.md", ("## Spectral Graph Networks", "wall"),
     "xt2-spectral", "A GCN layer is a low-pass filter on the graph spectrum",
     spectral_graph_filter),
    ("4.6-XT2-graph-neural-networks.md", ("## Expressive Power and the Limits of GNNs", "wall"),
     "xt2-wl", "Two graphs message passing cannot tell apart",
     wl_test_limit),
    ("4.5-RF1-reinforcement-learning.md", ("## Temporal-Difference Learning", "wall"),
     "rf1-sarsa-qlearning", "SARSA learns the safe path while Q-learning chases the optimal one",
     q_learning_vs_sarsa),
    # --- replacing hand-drawn SVGs and notebook screenshots -----------------
    ("2.3-ML6-model-evaluation.md", ("## Threshold Tradeoffs", "wall"),
     "ml6-threshold-tradeoff-curve", "Precision rises and recall falls as the decision threshold increases",
     threshold_precision_recall),
    ("2.3-ML6-model-evaluation.md", ("## Why Accuracy Can Mislead", "wall"),
     "ml6-accuracy-hides-failure", "Predicting the majority class gives 99% accuracy and 0% recall",
     accuracy_hides_failure),
    ("1.15-ML3-kernel-methods-svms.md", ("## The Margin", "wall"),
     "ml3-max-margin", "Only the support vectors set the maximum-margin boundary",
     max_margin_hyperplane),
    ("1.15-ML3-kernel-methods-svms.md", ("## The Dual Problem", "wall"),
     "ml3-primal-dual", "The dual replaces the feature dimension by the sample count",
     primal_versus_dual),
    ("1.15-ML3-kernel-methods-svms.md", ("## Why the RBF Kernel is Infinite-Dimensional", "wall"),
     "ml3-rbf-expansion", "The RBF kernel expands into every polynomial degree",
     rbf_infinite_expansion),
    ("1.15-ML3-kernel-methods-svms.md", ("## Overfitting with RBF SVMs", "wall"),
     "ml3-rbf-gamma", "Gamma sets how far one training point's influence reaches",
     rbf_gamma_overfits),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Feature Space Partitioning", "wall"),
     "ml4-tree-partition", "A tree is a sequence of threshold questions, and a partition into rectangles",
     decision_tree_splits_space),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Information Gain", "wall"),
     "ml4-split-gain", "A split is worth making only if the children are purer than the parent",
     split_purity_gain),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Decision Trees", "wall"),
     "ml4-classification-vs-regression", "Classification leaves predict a class, regression leaves a mean",
     classification_versus_regression_tree),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Bagging", "wall"),
     "ml4-bagging", "Averaging trees fitted to bootstrap samples cancels their individual noise",
     bagging_reduces_variance),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Random Forests", "wall"),
     "ml4-forest-decorrelation", "Averaging only helps to the extent the trees disagree",
     random_forest_decorrelates),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Boosting", "wall"),
     "ml4-boosting", "Each tree is fitted to what the ensemble still gets wrong",
     boosting_sequential_residuals),
    ("1.16-ML4-tree-based-models-ensembles.md", ("## Adaboost", "wall"),
     "ml4-adaboost-weights", "Misclassified examples gain weight so the next tree focuses on them",
     adaboost_reweights)]
