import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("02-gen-data/test.csv")
dur = df["dur_s"]
dur_0_80 = dur.iloc[0:81]   # 81 because upper bound is exclusive

mean = np.mean(dur_0_80)
std = np.std(dur_0_80)
print(std)
plt.figure()
plt.plot(dur_0_80, marker='o', linestyle='None')
plt.ylabel("Duration (s)")
plt.xlabel("Index")
plt.title("Duration (index 0–80)")
plt.axhline(mean, linestyle='--', label=f"Mean = {mean:.3f}s")
plt.show()



