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
itter_hist_vals = [hist_vals[v] for v in sorted(hist_vals)]
cumul_hist_vals = np.cumsum(itter_hist_vals)

# Plot the data (Gray-Level Histogram)
plt.subplot(211)
plt.hist(plot_hist_vals, bins=256)
plt.subplot(212)
plt.bar(range(len(cumul_hist_vals)), cumul_hist_vals)
plt.show()
