# Week 04 · Day 20 (PM Session) Assignment
**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**
Gitlink:https://github.com/shubhamrangdal2001-hash/Assignment_Day20PM.git
Topics: Normal Distribution · Standard Normal Distribution · Hypothesis Testing · Descriptive Statistics

---

## Folder Structure

```
week04-day20-pm/
├── part_a.py      # Part A – Concept Application (A1 to A5)
├── part_b.py      # Part B – Stretch Problems (B1 to B3)
├── part_c.py      # Part C – Interview Ready (Q1, Q2, Q3)
├── part_d.py      # Part D – AI-Augmented Task
├── README.md
└── outputs/
    ├── a1_histogram.png
    └── b1_comparison.png
```

---

## How to Run

> No external libraries required. Everything is implemented using Python's standard library (`math`, `random`, `matplotlib` for plots).

```bash
# Run each part independently
python part_a.py
python part_b.py
python part_c.py
python part_d.py
```

Make sure `matplotlib` is installed for plots:
```bash
pip install matplotlib
```

---

## What Each File Does

| File | Contents |
|------|----------|
| `part_a.py` | Generates a 1000-point normal dataset, computes statistics manually, Z-score transform, outlier detection, one-sample Z-test, and false positive rate simulation |
| `part_b.py` | Compares normal vs standard normal visually, runs a two-group Z-test, explains when/why to standardize data |
| `part_c.py` | Interview answers + `z_score(x, mean, std)` function applied on a dataset |
| `part_d.py` | AI prompt output (normal distribution + Z-score + hypothesis testing), runnable code, and student evaluation |

---

## Key Concepts Covered

- **Normal Distribution** – generated using the Box-Muller transform (no scipy)
- **Z-score** – computed manually as `(x - mean) / std`
- **One-sample Z-test** – compares sample mean to a hypothesized population mean
- **False Positive Rate** – simulated over 1000 trials; should approximate α = 0.05
- **Two-sample Z-test** – compares means of two independent groups

---

## Notes

- All statistical formulas are implemented manually without using `scipy.stats` or `numpy`.
- Random seed is fixed for reproducibility.
- Plots are saved as PNG files in the working directory.
