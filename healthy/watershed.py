# Implement the watershed algorithm to strip the skull (brutal)
# Lucas McCullum
# August 16, 2020

# Determine the correct imports
import matplotlib.pyplot as plt
import pydicom
import pdb

# Get the data
data = pydicom.dcmread('DICOM/ST000000/SE000000/MR000000')
pixel_vals = data.pixel_array

# Plot the data (Gray-Level Histogram)
plt.hist(pixel_vals.flatten(), bins=256)
plt.show()
