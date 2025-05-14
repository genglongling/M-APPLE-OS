import matplotlib.pyplot as plt
import numpy as np

# Median gap values (from your figure, approximate)
labels = [
    "MAPLE (Dynamic)", "SeEvo(GLM3)", "SeEvo(GPT3.5)", "GEP", "GP", "SPT", "TWKR", "SRM", "SSO", "LPT",
    "SPT/TWK", "SPT*TWK", "SPT+SSO", "SPT/LSO"
]
medians = [
    0.5,  # MAPLE (Dynamic)
    0.3,  # SeEvo(GLM3)
    #0.2,  # SeEvo(GPT3.5)
    2.2,  # GEP
    2.0,  # GP
    2.0,  # SPT
    2.2,  # TWKR
    2.2,  # SRM
    2.2,  # SSO
    2.2,  # LPT
    2.0,  # SPT/TWK
    2.0,  # SPT*TWK
    1.8,  # SPT+SSO
    2.0   # SPT/LSO
]

# Multiply medians by 10 to convert to percentage (0-100%)
medians = [m * 10 for m in medians]

# Simulate boxplot data (for illustration, use normal distribution around median)
data = [np.random.normal(loc=med, scale=5, size=20) for med in medians]

fig, ax = plt.subplots(figsize=(14, 6))
box = ax.boxplot(data, patch_artist=True, labels=labels, showmeans=False)

# Set colors: MAPLE (Dynamic) blue, others as in your figure
colors = ['blue', 'yellow', 'green', 'blue', 'pink', 'lightcoral', 'orange', 'gray', 'brown', 'purple', 'cyan', 'lightblue', 'lightgray', 'gray']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel("Gap Ratio (%)")
ax.set_ylim(0, 100)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("summary_dynamic_boxplot.png")
plt.show() 