"""
Part B - Stretch Problems
Week 04, Day 20 PM Session Assignment
Topics: Normal vs Standard Normal, Two-group Hypothesis Test, Standardization
"""

import random
import math
import matplotlib.pyplot as plt


# ── helper functions (same as Part A, kept self-contained) ──────────────

def generate_normal(n=1000, mu=0, sigma=1, seed=None):
    if seed is not None:
        random.seed(seed)
    data = []
    for _ in range(n // 2):
        u1 = max(random.random(), 1e-10)  # avoid log(0)
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        data.append(mu + z0 * sigma)
        data.append(mu + z1 * sigma)
    return data[:n]

def compute_mean(data):
    return sum(data) / len(data)

def compute_std(data):
    m = compute_mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

def z_score_transform(data):
    m, s = compute_mean(data), compute_std(data)
    return [(x - m) / s for x in data]


# ─────────────────────────────────────────────
# B1. Normal vs Standard Normal – plot both
# ─────────────────────────────────────────────

normal_data   = generate_normal(n=1000, mu=70, sigma=15, seed=7)  # e.g. marks
standard_data = z_score_transform(normal_data)

print("=== B1: Normal vs Standard Normal ===")
print(f"  Normal      → mean: {compute_mean(normal_data):.2f}, std: {compute_std(normal_data):.2f}")
print(f"  Standardised→ mean: {compute_mean(standard_data):.4f}, std: {compute_std(standard_data):.4f}")
print("  The shape is the same; only the scale/location changes.\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(normal_data, bins=30, color='coral', edgecolor='white')
axes[0].set_title("Normal Distribution (μ=70, σ=15)")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(standard_data, bins=30, color='mediumseagreen', edgecolor='white')
axes[1].set_title("Standard Normal Distribution (μ≈0, σ≈1)")
axes[1].set_xlabel("Z-score")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("b1_comparison.png", dpi=100)
plt.close()
print("  Plot saved → b1_comparison.png\n")

# ─────────────────────────────────────────────
# B2. Two-group hypothesis test (basic comparison)
# ─────────────────────────────────────────────

group_a = generate_normal(n=100, mu=68, sigma=10, seed=11)   # e.g. Class A scores
group_b = generate_normal(n=100, mu=75, sigma=10, seed=22)   # e.g. Class B scores

mean_a = compute_mean(group_a)
mean_b = compute_mean(group_b)
std_a  = compute_std(group_a)
std_b  = compute_std(group_b)
n_a, n_b = len(group_a), len(group_b)

# Two-sample Z-statistic
diff_means = mean_b - mean_a
se = math.sqrt((std_a ** 2 / n_a) + (std_b ** 2 / n_b))
z_stat = diff_means / se
z_crit = 1.96

print("=== B2: Two-Group Hypothesis Test ===")
print(f"  Group A → mean: {mean_a:.2f}, std: {std_a:.2f}, n: {n_a}")
print(f"  Group B → mean: {mean_b:.2f}, std: {std_b:.2f}, n: {n_b}")
print(f"  Difference in means (B - A): {diff_means:.4f}")
print(f"  Standard Error             : {se:.4f}")
print(f"  Z-statistic                : {z_stat:.4f}")
print(f"  Z-critical (α=0.05, 2-tail): ±{z_crit}")
if abs(z_stat) > z_crit:
    print("  Result: Reject H0 → there is a significant difference between the two groups.\n")
else:
    print("  Result: Fail to reject H0 → no significant difference detected.\n")

# ─────────────────────────────────────────────
# B3. Explanation – Standardisation & Z-score in ML
# ─────────────────────────────────────────────

print("=== B3: When to Standardize & Why Z-score Matters in ML ===")
print("""
  When should you standardize data?
  ----------------------------------
  - When features have different scales (e.g., age in 0-100 vs salary in 10k-100k).
  - Before using algorithms that are distance-based: KNN, SVM, K-Means.
  - Before gradient-descent based algorithms: Linear/Logistic Regression, Neural Networks.
  - When you need to compare different variables on the same scale.
  - NOT needed for tree-based models (Decision Tree, Random Forest) since they split on thresholds.

  Why is Z-score important in ML?
  --------------------------------
  - It removes the effect of scale so that no single feature dominates just because of its unit.
  - It speeds up gradient descent convergence (loss surface is more symmetric).
  - Helps in outlier detection: any point with |Z| > 3 is likely an outlier.
  - Makes model weights/coefficients directly comparable across features.
  - Required for PCA so that variance is not dominated by high-magnitude features.
""")
