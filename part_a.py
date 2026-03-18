"""
Part A - Concept Application
Week 04, Day 20 PM Session Assignment
Topics: Normal Distribution, Z-score, Hypothesis Testing
"""

import random
import math
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# A1. Generate dataset from normal distribution
# ─────────────────────────────────────────────

def generate_normal(n=1000, mu=50, sigma=10):
    """Generate normally distributed data using Box-Muller transform."""
    data = []
    for _ in range(n // 2):
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        data.append(mu + z0 * sigma)
        data.append(mu + z1 * sigma)
    return data[:n]

def compute_mean(data):
    return sum(data) / len(data)

def compute_variance(data):
    m = compute_mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def compute_std(data):
    return math.sqrt(compute_variance(data))

random.seed(42)
dataset = generate_normal(n=1000, mu=50, sigma=10)

mean_val  = compute_mean(dataset)
var_val   = compute_variance(dataset)
std_val   = compute_std(dataset)

print("=== A1: Normal Distribution ===")
print(f"  Mean     : {mean_val:.4f}")
print(f"  Variance : {var_val:.4f}")
print(f"  Std Dev  : {std_val:.4f}")

# Plot histogram
plt.figure(figsize=(7, 4))
plt.hist(dataset, bins=30, color='steelblue', edgecolor='white')
plt.title("A1 – Histogram of Generated Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("a1_histogram.png", dpi=100)
plt.close()
print("  Histogram saved → a1_histogram.png\n")

# ─────────────────────────────────────────────
# A2. Convert to Standard Normal (Z-score)
# ─────────────────────────────────────────────

def z_score_transform(data):
    m  = compute_mean(data)
    sd = compute_std(data)
    return [(x - m) / sd for x in data]

z_data = z_score_transform(dataset)

print("=== A2: Standard Normal (Z-scores) ===")
print(f"  Z-score mean : {compute_mean(z_data):.6f}  (expected ≈ 0)")
print(f"  Z-score std  : {compute_std(z_data):.6f}  (expected ≈ 1)\n")

# ─────────────────────────────────────────────
# A3. Student marks – statistics & outliers
# ─────────────────────────────────────────────

marks = [45, 67, 72, 55, 88, 91, 34, 76, 65, 58,
         100, 23, 70, 82, 49, 61, 78, 95, 54, 69]

def compute_median(data):
    s = sorted(data)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

m_mean   = compute_mean(marks)
m_median = compute_median(marks)
m_var    = compute_variance(marks)
m_std    = compute_std(marks)

print("=== A3: Student Marks Statistics ===")
print(f"  Mean   : {m_mean:.2f}")
print(f"  Median : {m_median:.2f}")
print(f"  Var    : {m_var:.2f}")
print(f"  Std    : {m_std:.2f}")

# Outliers using Z-score threshold
z_marks = [(x - m_mean) / m_std for x in marks]
outliers = [marks[i] for i, z in enumerate(z_marks) if abs(z) > 2]
print(f"  Outliers (|Z| > 2) : {outliers}\n")

# ─────────────────────────────────────────────
# A4. One-sample Z-test (manual)
# ─────────────────────────────────────────────

def one_sample_z_test(data, mu0, alpha=0.05):
    """
    H0: population mean = mu0
    Returns z-statistic and reject/fail decision.
    """
    n       = len(data)
    x_bar   = compute_mean(data)
    sigma   = compute_std(data)
    z_stat  = (x_bar - mu0) / (sigma / math.sqrt(n))

    # Critical value for two-tailed test at alpha=0.05 → z_crit ≈ 1.96
    z_crit  = 1.96
    reject  = abs(z_stat) > z_crit

    print("=== A4: One-Sample Z-Test ===")
    print(f"  H0       : μ = {mu0}")
    print(f"  n        : {n}")
    print(f"  x̄        : {x_bar:.4f}")
    print(f"  σ        : {sigma:.4f}")
    print(f"  Z-stat   : {z_stat:.4f}")
    print(f"  Z-crit   : ±{z_crit}  (α = {alpha}, two-tailed)")
    if reject:
        print(f"  Decision : Reject H0  (|Z| = {abs(z_stat):.4f} > {z_crit})\n")
    else:
        print(f"  Decision : Fail to reject H0  (|Z| = {abs(z_stat):.4f} ≤ {z_crit})\n")
    return z_stat, reject

one_sample_z_test(dataset, mu0=50)

# ─────────────────────────────────────────────
# A5. Simulate 1000 hypothesis tests (false positive rate)
# ─────────────────────────────────────────────

print("=== A5: False Positive Rate Simulation ===")

n_simulations = 1000
alpha         = 0.05
z_crit        = 1.96
false_positives = 0
true_mu       = 50   # H0 is actually true

for _ in range(n_simulations):
    sample     = generate_normal(n=100, mu=true_mu, sigma=10)
    x_bar      = compute_mean(sample)
    sigma      = compute_std(sample)
    z_stat     = (x_bar - true_mu) / (sigma / math.sqrt(len(sample)))
    if abs(z_stat) > z_crit:
        false_positives += 1

false_positive_rate = false_positives / n_simulations
print(f"  Simulations       : {n_simulations}")
print(f"  False Positives   : {false_positives}")
print(f"  False Positive Rate: {false_positive_rate:.4f}")
print(f"  Significance Level : {alpha}")
print(f"  Match?             : {'Yes – close to α' if abs(false_positive_rate - alpha) < 0.02 else 'Slight deviation (expected due to randomness)'}\n")
