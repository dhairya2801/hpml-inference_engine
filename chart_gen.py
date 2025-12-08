import matplotlib.pyplot as plt
import numpy as np

phases = ['Phase 1\n(Naive)', 'Phase 2\n(Static Batching)', 'Phase 3\n(NanoVLLM)']
throughput = [4.37, 82.30, 176.14] # From your screenshots
colors = ['#ff9999', '#66b3ff', '#99ff99']

plt.figure(figsize=(10, 6))
bars = plt.bar(phases, throughput, color=colors, edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Throughput Comparison: From Naive to Optimized', fontsize=16, fontweight='bold')
plt.ylabel('Throughput (tokens/sec)', fontsize=12)
plt.ylim(0, 200)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.annotate('40x Speedup!', xy=(2, 176), xytext=(1.5, 120),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.savefig('throughput_comparison.png', dpi=300)
print("Generated throughput_comparison.png")

labels = ['Cold Request\n(First time)', 'Warm Request\n(Cached)']
times = [2.19, 0.30]
colors_cache = ['#ffcc99', '#99ff99']

plt.figure(figsize=(8, 5))
bars2 = plt.bar(labels, times, color=colors_cache, edgecolor='black', width=0.6)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Impact of Prefix Caching on Latency', fontsize=16, fontweight='bold')
plt.ylabel('Time to Complete (seconds)', fontsize=12)
plt.ylim(0, 2.5)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.annotate('7x Faster', xy=(1, 0.30), xytext=(0.5, 1.0),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5), fontsize=12)

plt.savefig('prefix_caching.png', dpi=300)
print("Generated prefix_caching.png")
