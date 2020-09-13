# Lucas McCullum
# September 10th, 2020
# Used to convert MRI images to k-Space

# Outside packages
import pydicom
import numpy as np
import cmath
import pdb

data = pydicom.dcmread('DICOM/ST000000/SE000000/MR000000')
normal_im = data.pixel_array

# Perform the 2-Dimensional Discrete Fourier Transform (DFT) to get the k-Space
# F[k,l] = \sum_{m=0}^{N-1}\sum_{n=0}^{N-1}f[m,n]e^{(-i)\frac{2\pi km}{N}}e^{(-i)\frac{2\pi ln}{N}}
N = normal_im.shape[0]

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
pdb.set_trace()
