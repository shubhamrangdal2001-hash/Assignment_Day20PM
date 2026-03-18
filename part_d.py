"""
Part D - AI-Augmented Task
Week 04, Day 20 PM Session Assignment

Prompt used:
    "Explain normal distribution, Z-score, and hypothesis testing
     with a simple Python example."

AI Output (from Claude / ChatGPT):
    [See below — reproduced and evaluated]
"""

import math

# ─────────────────────────────────────────────────────────────────
# AI-GENERATED CODE (from the prompt above)
# Evaluated below for correctness and runnability
# ─────────────────────────────────────────────────────────────────

# --- AI Output starts ---

def generate_normal_sample(n, mu, sigma):
    """Generate n samples from N(mu, sigma) using Box-Muller."""
    import random
    samples = []
    for _ in range(n // 2):
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        samples.extend([mu + z0 * sigma, mu + z1 * sigma])
    return samples[:n]

def compute_stats(data):
    n    = len(data)
    mean = sum(data) / n
    var  = sum((x - mean) ** 2 for x in data) / n
    std  = math.sqrt(var)
    return mean, var, std

def z_transform(data):
    mean, _, std = compute_stats(data)
    return [(x - mean) / std for x in data]

def z_test(data, mu0, alpha=0.05):
    mean, _, std = compute_stats(data)
    n      = len(data)
    z_stat = (mean - mu0) / (std / math.sqrt(n))
    z_crit = 1.96
    reject = abs(z_stat) > z_crit
    return z_stat, reject

# --- AI Output ends ---

# ─────────────────────────────────────────────────────────────────
# Running the AI-generated code
# ─────────────────────────────────────────────────────────────────

print("=== D: Running AI-Generated Code ===\n")

import random
random.seed(0)

sample = generate_normal_sample(500, mu=60, sigma=8)
mean, var, std = compute_stats(sample)
print(f"  Generated sample of 500 values")
print(f"  Mean     : {mean:.4f}")
print(f"  Variance : {var:.4f}")
print(f"  Std Dev  : {std:.4f}")

z_data = z_transform(sample)
z_mean, _, z_std = compute_stats(z_data)
print(f"\n  After Z-transform:")
print(f"  Mean ≈ {z_mean:.6f}  (expected ≈ 0)")
print(f"  Std  ≈ {z_std:.6f}  (expected ≈ 1)")

z_stat, reject = z_test(sample, mu0=60)
print(f"\n  Z-test (H0: μ = 60)")
print(f"  Z-statistic : {z_stat:.4f}")
print(f"  Decision    : {'Reject H0' if reject else 'Fail to Reject H0'}")

# ─────────────────────────────────────────────────────────────────
# Evaluation of AI Output
# ─────────────────────────────────────────────────────────────────

print("""
=== D: Evaluation of AI Output ===

  Is the explanation correct?
    Yes. The AI correctly explained:
    - Normal distribution as a bell curve described by mean and std.
    - Z-score as a way to standardise values relative to the mean.
    - Hypothesis testing as a framework to evaluate claims about population parameters.

  Is the code logically correct and runnable?
    Yes. All three functions (generate_normal_sample, compute_stats, z_transform, z_test)
    run without errors. The logic matches standard statistical definitions:
    - Box-Muller transform is a valid method to generate normal samples.
    - Z-score formula: (x - mean) / std is correct.
    - Z-test uses the correct formula: (x̄ - μ0) / (σ / √n).
    - Critical value 1.96 is correct for α=0.05 two-tailed test.

  Any issues noticed?
    - The AI did not handle edge cases like std=0 (division by zero).
    - It didn't verify normality visually (no histogram).
    These are minor gaps that a student would normally catch during testing.

  Overall: The AI output is correct and useful as a starting reference.
           It should always be verified before submitting as final work.
""")
