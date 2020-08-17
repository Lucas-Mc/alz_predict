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
pixel_vals = data.pixel_array
plot_hist_vals = pixel_vals.flatten()
hist_vals = collections.Counter(plot_hist_vals)
itter_hist_vals = [hist_vals[key] for key in sorted(hist_vals)]
cumul_hist_vals = np.cumsum(itter_hist_vals)

# Equalize the histogram
cumul_hist_min = min(cumul_hist_vals)
hist_eq_denom = pixel_vals.shape[0] * pixel_vals.shape[1] - cumul_hist_min
hist_eq = [round(((v - cumul_hist_min) / hist_eq_denom) * 255) for v in cumul_hist_vals] 
cumul_hist_eq = np.cumsum(hist_eq)

# Plot the data
# Gray-Level Histogram
plt.subplot(411)
plt.hist(plot_hist_vals, bins=256)
# Cumulative Histogram
plt.subplot(412)
plt.bar(range(len(cumul_hist_vals)), cumul_hist_vals)
# Gray-Level Equalized Histogram
plt.subplot(413)
pdb.set_trace()
plt.bar(hist_eq, [v[1] for v in sorted(hist_vals.items())])
# Cumulative Equalized Histogram
plt.subplot(414)
plt.bar(range(len(cumul_hist_eq)), cumul_hist_eq)
# Display the plot
plt.show()
