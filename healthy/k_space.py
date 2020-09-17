# Lucas McCullum
# September 10th, 2020
# Used to convert MRI images to k-Space

# Outside packages
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import cmath
import pdb

data = pydicom.dcmread('DICOM/ST000000/SE000000/MR000000')
normal_im = data.pixel_array

# Perform the 2-Dimensional Discrete Fourier Transform (DFT) to get the k-Space
# F[k,l] = \sum_{m=0}^{N-1}\sum_{n=0}^{N-1}f[m,n]e^{(-i)\frac{2\pi km}{N}}e^{(-i)\frac{2\pi ln}{N}}
# N = normal_im.shape[0]

# VERYYYY slow way
# fourier_im = np.empty((N,N), dtype=np.complex_)
# exp_mat = np.empty((N,N), dtype=np.complex_)
#
# for i in range(N):
#     for j in range(N):
#         exp_mat[i,j] = cmath.exp((-1j) * ((2 * cmath.pi * i * j) / N))
#
# for k in range(N):
#     for l in range(N):
#         temp_val = 0
#         for m in range(N):
#             for n in range(N):
#                 temp_val += normal_im[m,n] * exp_mat[k,m] * exp_mat[l,n]
#         fourier_im[k,l] = temp_val
#         print('Done: ({},{})'.format(k,l))

# Much faster way
fourier_im = np.fft.fft2(normal_im)
inverse_fourier_im = np.fft.ifft2(fourier_im)

# Plot the Fourier Transformation
plt.figure(figsize=(10,4))
plt.suptitle('Fourier Tranformation of Original Image')
# Plot the real components
plt.subplot(1,3,1)
plt.title('Real')
plt.imshow(fourier_im.real  > 0, cmap='Greys')
# Plot the imaginary components
plt.subplot(1,3,2)
plt.title('Imaginary')
plt.imshow(fourier_im.imag  > 0, cmap='Greys')
# Plot the absolute value
plt.subplot(1,3,3)
plt.title('Absolute Value')
plt.imshow(abs(fourier_im > 0), cmap='Greys')
# Show the plot
plt.show()

# Plot the Inverse Fourier Transformation
plt.figure(figsize=(10,4))
plt.suptitle('Inverse Fourier Tranformation of Fourier Transform of Original Image')
# Plot the real components
plt.subplot(1,3,1)
plt.title('Real')
plt.imshow(inverse_fourier_im.real, cmap='Greys')
# Plot the imaginary components
plt.subplot(1,3,2)
plt.title('Imaginary')
plt.imshow(inverse_fourier_im.imag, cmap='Greys')
# Plot the absolute value
plt.subplot(1,3,3)
plt.title('Absolute Value')
plt.imshow(abs(inverse_fourier_im), cmap='Greys')
# Show the plot
plt.show()
