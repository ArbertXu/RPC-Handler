import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.style.use("seaborn-v0_8-darkgrid")

# Common tick values
tick_vals = [1, 2, 5, 10, 20, 50, 100, 200]

# Helper to apply log scale with readable tick labels
def set_log_ticks(ax):
    ax.set_yscale("log")
    ax.set_yticks(tick_vals)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())  # show numbers, not scientific notation

# --- 1. Load-Latency Curve ---
df = pd.read_csv("load_latency_curve.csv")

plt.figure(figsize=(8, 6))
plt.plot(df["Target_RPS"], df["Avg_ms"], marker='o', label="Average (ms)")
plt.plot(df["Target_RPS"], df["p50_ms"], marker='x', label="p50 (ms)")
plt.plot(df["Target_RPS"], df["p95_ms"], marker='^', label="p95 (ms)")
plt.plot(df["Target_RPS"], df["p99_ms"], marker='s', label="p99 (ms)")
plt.xlabel("Target RPS")
plt.ylabel("Latency (ms, log scale)")
set_log_ticks(plt.gca())
plt.title("Load–Latency Curve")
plt.legend()
plt.tight_layout()
plt.show()

# --- 2. Operation Comparison ---
df = pd.read_csv("operation_comparison.csv")

plt.figure(figsize=(8, 6))
for op, group in df.groupby("Operation"):
    plt.plot(group["Target_RPS"], group["Avg_ms"], marker='o', label=f"{op} avg (ms)")
plt.xlabel("Target RPS")
plt.ylabel("Average Latency (ms, log scale)")
set_log_ticks(plt.gca())
plt.title("Operation Comparison: Avg Latency vs Load")
plt.legend()
plt.tight_layout()
plt.show()

# --- 3. Sync vs Async Comparison ---
df = pd.read_csv("sync_vs_async_comparison.csv")

plt.figure(figsize=(8, 6))
for mode, group in df.groupby("Mode"):
    plt.plot(group["Target_RPS"], group["Avg_ms"], marker='o', label=f"{mode} avg (ms)")
plt.xlabel("Target RPS")
plt.ylabel("Average Latency (ms, log scale)")
set_log_ticks(plt.gca())
plt.title("Sync vs Async Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# --- 4. Throughput Analysis (simplified) ---
df = pd.read_csv("throughput_analysis.csv")

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(df["Target_RPS"], df["Actual_RPS"], marker='o', color='tab:blue', label="Actual RPS")
ax1.set_xlabel("Target RPS")
ax1.set_ylabel("Actual RPS", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Secondary y-axis for latency only
ax2 = ax1.twinx()
ax2.plot(df["Target_RPS"], df["p99_ms"], marker='^', color='tab:red', label="p99 (ms)")
set_log_ticks(ax2)
ax2.set_ylabel("Latency (ms, log scale)", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.title("Throughput Analysis (RPS vs p99 Latency)")
plt.tight_layout()
plt.show()
