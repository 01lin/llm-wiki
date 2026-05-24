# Enhanced 3D plot with contour projection (no colors specified explicitly)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Grid
accept_len = np.linspace(1, 20, 80)
perf = np.linspace(0.1, 1.0, 80)
A, P = np.meshgrid(accept_len, perf)

# More realistic speedup model (non-linear)
# diminishing returns on acceptance + performance interaction
S = (A * P) / (1 + 0.05 * A)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Surface
surf = ax.plot_surface(A, P, S)

# Contour projection on bottom plane
ax.contour(A, P, S, zdir='z', offset=np.min(S))

# Labels
ax.set_xlabel('Acceptance Length')
ax.set_ylabel('Inference Performance')
ax.set_zlabel('Speedup')

# Key annotation points
ax.scatter([18], [0.9], [(18*0.9)/(1+0.05*18)])  # high-high
ax.text(18, 0.9, (18*0.9)/(1+0.05*18), 'High Speedup Region')

ax.scatter([5], [0.9], [(5*0.9)/(1+0.05*5)])  # high perf, low accept
ax.text(5, 0.9, (5*0.9)/(1+0.05*5), 'Low Accept')

ax.scatter([18], [0.2], [(18*0.2)/(1+0.05*18)])  # low perf
ax.text(18, 0.2, (18*0.2)/(1+0.05*18), 'Low Perf')

plt.title("Speculative Decoding Tradeoff Surface")

plt.show()