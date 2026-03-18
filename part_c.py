"""
Part C - Interview Ready (Coding Question)
Week 04, Day 20 PM Session Assignment
Q2: Implement z_score(x, mean, std) and apply on a dataset
"""

import math


# ─────────────────────────────────────────────
# C-Q2: z_score function
# ─────────────────────────────────────────────

def z_score(x, mean, std):
    """
    Returns the standardised Z-score of a value x.

    Parameters:
        x    : single value or list of values
        mean : population / sample mean
        std  : population / sample standard deviation

    Returns:
        float or list of floats
    """
    if std == 0:
        raise ValueError("Standard deviation cannot be zero.")

    if isinstance(x, (list, tuple)):
        return [(val - mean) / std for val in x]
    return (x - mean) / std


# ── Answers to written interview questions ──────────────────────────────

print("=== C-Q1: Normal vs Standard Normal Distribution ===")
print("""
  Normal Distribution:
    - Characterised by any mean (μ) and standard deviation (σ).
    - Bell-shaped and symmetric.
    - Example: heights of people with μ=170cm, σ=10cm.

  Standard Normal Distribution:
    - A special case of normal distribution with μ=0 and σ=1.
    - Obtained by applying Z-score transformation to a normal distribution.
    - Allows comparison across different scales using a single Z-table.
""")

print("=== C-Q2: z_score function demo ===")

dataset = [45, 67, 72, 55, 88, 91, 34, 76, 65, 58]

# compute mean and std manually
n    = len(dataset)
mean = sum(dataset) / n
std  = math.sqrt(sum((x - mean) ** 2 for x in dataset) / n)

print(f"  Dataset : {dataset}")
print(f"  Mean    : {mean:.2f}")
print(f"  Std Dev : {std:.2f}")
print()

# Apply z_score on full dataset
z_values = z_score(dataset, mean, std)
print("  Z-scores:")
for i, (orig, z) in enumerate(zip(dataset, z_values)):
    print(f"    x={orig:3d}  →  Z = {z:+.4f}")

print()
# Apply on a single value
single_val = 91
z_single = z_score(single_val, mean, std)
print(f"  Single value: z_score({single_val}, {mean:.2f}, {std:.2f}) = {z_single:+.4f}")

print()
print("=== C-Q3: Hypothesis Testing Concepts ===")
print("""
  Hypothesis Testing:
    A statistical procedure to decide whether a claim about a population is supported by data.

  Null Hypothesis (H0):
    The default assumption we try to disprove.
    Example: "The average marks of students is 60."

  Alternative Hypothesis (H1):
    What we believe if H0 is false.
    Example: "The average marks of students is NOT 60."

  p-value:
    The probability of getting a result at least as extreme as observed, assuming H0 is true.
    - Small p-value (< α) → strong evidence against H0 → reject H0.
    - Large p-value (≥ α) → not enough evidence → fail to reject H0.

  Significance Level (α):
    The threshold we set before the test (commonly 0.05).
    If p-value < α, we reject H0.
    It represents the acceptable probability of making a Type I error (false positive).

  Decision rule (Z-test, two-tailed, α=0.05):
    - If |Z-stat| > 1.96 → Reject H0
    - If |Z-stat| ≤ 1.96 → Fail to Reject H0
""")
