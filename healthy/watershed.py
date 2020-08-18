# Implement the watershed algorithm to strip the skull (brutal)
# Lucas McCullum
# August 16, 2020

# Determine the correct imports
import matplotlib.pyplot as plt
import pydicom
import collections
import numpy as np
import pdb

# Get the data
data = pydicom.dcmread('DICOM/ST000000/SE000000/MR000000')
normal_im = data.pixel_array

plot_hist_vals = normal_im.flatten()
hist_vals = collections.Counter(plot_hist_vals)
itter_hist_vals = [hist_vals[key] for key in sorted(hist_vals)]
cumul_hist_vals = np.cumsum(itter_hist_vals)

# Equalize the histogram
cumul_hist_min = min(cumul_hist_vals)
hist_eq_denom = normal_im.shape[0] * normal_im.shape[1] - cumul_hist_min
hist_eq = [round(((v - cumul_hist_min) / hist_eq_denom) * 255) for v in cumul_hist_vals] 
cumul_hist_eq = np.cumsum(hist_eq)

# Determine the equalized image
value_map = dict(zip(set(plot_hist_vals),hist_eq))
eq_im = np.array([value_map[v] for v in plot_hist_vals]).reshape(normal_im.shape)

# Plot the data
# Gray-Level Histogram
plt.subplot(421)
plt.hist(plot_hist_vals, bins=256)
# Cumulative Histogram
plt.subplot(423)
plt.bar(range(len(cumul_hist_vals)), cumul_hist_vals)
# Gray-Level Equalized Histogram
plt.subplot(425)
plt.bar(hist_eq, [v[1] for v in sorted(hist_vals.items())])
# Cumulative Equalized Histogram
plt.subplot(427)
plt.bar(range(len(cumul_hist_eq)), cumul_hist_eq)
# Show the normal image
plt.subplot(4,2,(2,4))
plt.imshow(normal_im, cmap=plt.cm.bone)
# Show the equalized image
plt.subplot(4,2,(6,8))
plt.imshow(eq_im, cmap=plt.cm.bone)
# Display the plot
plt.show()
