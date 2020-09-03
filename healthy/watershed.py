# Implement the watershed algorithm to strip the skull (brutal)
# Lucas McCullum
# August 16, 2020

# These techniques assume the image is 256x256

# Determine the correct imports
import matplotlib.pyplot as plt
import pydicom
import collections
import numpy as np
import pdb


def equalize_image(normal_image, adaptive=False, tile_size=(8,8), contrast_limit=None):
    """
    Parameters
    ----------
    normal_image : ndarray
        The image desired to be equalized.
    adaptive : bool, optional
        Determines whether (True) or not (False) the adaptive histogram
        equalization should be used.
    tile_size : tuple, optional
        Must be defined if adaptive is `True`, determines the sub-blocks
        used to create the local histograms. These values should be divisible
        by the dimensions of the image. Default is `(8,8)`.
    contrast_limit : int, optional
        Defines the maximum allowed contrast for the input image histogram.
        Leaving the value as `None` will not apply a contrast limit.

    Returns
    -------
    equalized_image : ndarray
        The image after equalization.

    """
    # Check for valid inputs
    if adaptive:
        if (normal_image.shape[0] % tile_size[0] != 0) or (normal_image.shape[1] % tile_size[1] != 0):
            raise Exception('The tile size must be divisible by the image size: {}'.format(normal_image.shape))

    if adaptive:
        # From: https://www.academia.edu/26306360/Realization_of_the_Contrast_Limited_Adaptive_Histogram_Equalization_CLAHE_for_Real_Time_Image_Enhancement
        # Get the dimensions of each tile based on the number of sections
        tile_width = normal_image.shape[1] // tile_size[1]
        tile_height = normal_image.shape[0] // tile_size[0]

        # Get the reference histogram values for the algorithm
        hist_grid = [[[] for i in range(tile_size[0])] for j in range(tile_size[1])]
        for i in range(tile_size[1]):
            for j in range(tile_size[0]):
                # Divide the histogram into sections
                x_start = i * tile_width
                y_start = j * tile_height
                x_stop = x_start + tile_width
                y_stop = y_start + tile_height

                # Determing the histogram
                sub_image = normal_image[x_start:x_stop, y_start:y_stop]
                plot_hist_vals = sub_image.flatten()
                hist_dict = collections.Counter(plot_hist_vals)
                hist_keys = [v[0] for v in sorted(hist_dict.items())]
                hist_values = [v[1] for v in sorted(hist_dict.items())]
                cumul_hist_vals = np.cumsum(hist_values)

                # Determine the equalized image
                N = len(hist_values)
                M = len(sub_image.flatten())
                hist_grid[i][j] = ((N - 1) / M) * cumul_hist_vals

        # Use the equalized values to run the adaptive algorithm
        output_hist_grid = np.empty((normal_image.shape))
        for i in range(tile_size[1]):
            for j in range(tile_size[0]):
                # Determine the indices of the histogram section
                x_start = i * tile_width
                y_start = j * tile_height
                x_stop = x_start + tile_width
                y_stop = y_start + tile_height

                if ((i == 0) and (j == 0)):
                    # Top left corner
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if ((ii < tile_width/2) and (jj < tile_height/2)):
                                # Corner
                                output_hist_grid[x_start+ii, y_start+jj] = hist_grid[i][j]
                            elif ((ii > tile_width/2) and (jj < tile_height/2)):
                                # Top edge
                                s = ii - ((tile_width - 1) / 2)
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j+1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif ((ii < tile_width/2) and (jj > tile_height/2)):
                                # Left edge
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i+1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            else:
                                # Center
                                y = ii - ((tile_width - 1) / 2)
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j+1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j+1] + (x/(x+y)) * hist_grid[i][j])

                elif ((i == 0) and (j == 4*tile_size[0]-1)):
                    # Bottom left corner
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if ((ii < tile_width/2) and (jj > tile_height/2)):
                                # Corner
                                output_hist_grid[x_start+ii, y_start+jj] = hist_grid[i][j]
                            elif ((ii > tile_width/2) and (jj > tile_height/2)):
                                # Bottom edge
                                s = ii - ((tile_width - 1) / 2)
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j+1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif ((ii < tile_width/2) and (jj < tile_height/2)):
                                # Left edge
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i-1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            else:
                                # Center
                                y = ii - ((tile_width - 1) / 2)
                                x = tile_width - y
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i-1][j+1] + (x/(x+y)) * hist_grid[i-1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j+1] + (x/(x+y)) * hist_grid[i][j])

                elif ((i == 4*tile_size[1]-1) and (j == 0)):
                    # Top right corner
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if ((ii > tile_width/2) and (jj < tile_height/2)):
                                # Corner
                                output_hist_grid[x_start+ii, y_start+jj] = hist_grid[i][j]
                            elif ((ii < tile_width/2) and (jj < tile_height/2)):
                                # Top edge
                                s = ((tile_width - 1) / 2) - ii
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j-1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif ((ii > tile_width/2) and (jj > tile_height/2)):
                                # Right edge
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i+1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            else:
                                # Center
                                y = ((tile_width - 1) / 2) - ii
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j-1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j-1] + (x/(x+y)) * hist_grid[i][j])

                elif ((i == 4*tile_size[1]-1) and (j == 4*tile_size[0]-1)):
                    # Bottom right corner
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if ((ii > tile_width/2) and (jj > tile_height/2)):
                                # Corner
                                output_hist_grid[x_start+ii, y_start+jj] = hist_grid[i][j][x_start+ii, y_start+jj]
                            elif ((ii > tile_width/2) and (jj < tile_height/2)):
                                # Bottom edge
                                s = ((tile_width - 1) / 2) - ii
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j-1][ii][jj]) + ((r/(r+s)) * hist_grid[i][j][ii][jj])
                            elif ((ii < tile_width/2) and (jj > tile_height/2)):
                                # Right edge
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i-1][j][ii][jj]) + ((r/(r+s)) * hist_grid[i][j][ii][jj])
                            else:
                                # Center
                                y = ((tile_width - 1) / 2) - ii
                                x = tile_width - y
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i-1][j-1][ii][jj] + (x/(x+y)) * hist_grid[i-1][j][ii][jj]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j-1][ii][jj] + (x/(x+y)) * hist_grid[i][j][ii][jj])               

                elif (i == 0):
                    # Left edges
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if (ii < tile_width/2) and (jj < tile_height/2):
                                # Top edge
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i-1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii < tile_width/2) and (jj > tile_height/2):
                                # Bottom edge
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i+1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii > tile_width/2) and (jj < tile_height/2):
                                # Top center
                                y = ii - ((tile_width - 1) / 2)
                                x = tile_width - y
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i-1][j+1] + (x/(x+y)) * hist_grid[i-1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j+1] + (x/(x+y)) * hist_grid[i][j])
                            else:
                                # Bottom center
                                y = ii - ((tile_width - 1) / 2)
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j+1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j+1] + (x/(x+y)) * hist_grid[i][j])

                elif (i == tile_size[1]-1):
                    # Right edges
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if (ii < tile_width/2) and (jj > tile_height/2):
                                # Top edge
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i-1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii > tile_width/2) and (jj > tile_height/2):
                                # Bottom edge
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i+1][j]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii < tile_width/2) and (jj < tile_height/2):
                                # Top center
                                y = ((tile_width - 1) / 2) - ii
                                x = tile_width - y
                                s = ((tile_height - 1) / 2) - jj
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i-1][j-1][ii][jj] + (x/(x+y)) * hist_grid[i-1][j][ii][jj]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j-1][ii][jj] + (x/(x+y)) * hist_grid[i][j][ii][jj])   
                            else:
                                # Bottom center
                                y = ((tile_width - 1) / 2) - ii
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j-1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j-1] + (x/(x+y)) * hist_grid[i][j])

                elif (j == 0):
                    # Top edges
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if (ii < tile_width/2) and (jj < tile_height/2):
                                # Left edge
                                s = ((tile_width - 1) / 2) - ii
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j-1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii > tile_width/2) and (jj < tile_height/2):
                                # Right edge
                                s = ii - ((tile_width - 1) / 2)
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j+1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii < tile_width/2) and (jj > tile_height/2):
                                # Left center
                                y = ((tile_width - 1) / 2) - ii
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j-1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j-1] + (x/(x+y)) * hist_grid[i][j])
                            else:
                                # Right center
                                y = ii - ((tile_width - 1) / 2)
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j+1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j+1] + (x/(x+y)) * hist_grid[i][j])

                elif (j == tile_size[0]-1):
                    # Bottom edges
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if (ii < tile_width/2) and (jj < tile_height/2):
                                # Left edge
                                s = ((tile_width - 1) / 2) - ii
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j-1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii > tile_width/2) and (jj < tile_height/2):
                                # Right edge
                                s = ii - ((tile_width - 1) / 2)
                                r = tile_width - s
                                output_hist_grid[x_start+ii, y_start+jj] = ((s/(r+s)) * hist_grid[i][j+1]) + ((r/(r+s)) * hist_grid[i][j])
                            elif (ii < tile_width/2) and (jj > tile_height/2):
                                # Left center
                                y = ((tile_width - 1) / 2) - ii
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j-1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j-1] + (x/(x+y)) * hist_grid[i][j])
                            else:
                                # Right center
                                y = ii - ((tile_width - 1) / 2)
                                x = tile_width - y
                                s = jj - ((tile_height - 1) / 2)
                                r = tile_height - s
                                output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i+1][j+1] + (x/(x+y)) * hist_grid[i+1][j]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i][j+1] + (x/(x+y)) * hist_grid[i][j])

                else:
                    # Middle
                    for ii in range(tile_width):
                        for jj in range(tile_height):
                            if (0):
                                # Top right center
                                pass
                            elif (1):
                                # Bottom right center
                                pass
                            elif (2):
                                # Bottom left center
                                pass
                            else:
                                # Top left center
                                pass
                            r = jj
                            s = tile_height - jj
                            x = ii
                            y = tile_width - ii
                            output_hist_grid[x_start+ii, y_start+jj] = (s/(r+s)) * ((y/(x+y)) * hist_grid[i-1][j-1][ii][jj] + (x/(x+y)) * hist_grid[i][j-1][ii][jj]) + \
                                    (r/(r+s)) * ((y/(x+y)) * hist_grid[i-1][j][ii][jj] + (x/(x+y)) * hist_grid[i][j][ii][jj])

        pdb.set_trace()
        equalized_image = output_hist_grid

    else:
        # Determing the histogram
        plot_hist_vals = normal_image.flatten()
        hist_dict = collections.Counter(plot_hist_vals)
        hist_keys = [v[0] for v in sorted(hist_dict.items())]
        hist_values = [v[1] for v in sorted(hist_dict.items())]
        cumul_hist_vals = np.cumsum(hist_values)

        # Equalize the histogram
        cumul_hist_min = min(cumul_hist_vals)
        hist_eq_denom = normal_image.shape[0] * normal_image.shape[1] - cumul_hist_min
        hist_eq = [round(((v - cumul_hist_min) / hist_eq_denom) * 255) for v in cumul_hist_vals]
        cumul_hist_eq = np.cumsum(hist_eq)

        # Determine the equalized image
        value_map = dict(zip(hist_keys, hist_eq))
        equalized_image = np.array([value_map[v] for v in plot_hist_vals]).reshape(normal_im.shape)

    if contrast_limit:
        pass

    return equalized_image


# Get the data
data = pydicom.dcmread('DICOM/ST000000/SE000000/MR000000')
normal_im = data.pixel_array
eq_im = equalize_image(normal_im, adaptive=True)
pdb.set_trace()

# Determing the histogram
plot_hist_vals = normal_im.flatten()
hist_dict = collections.Counter(plot_hist_vals)
hist_keys = [v[0] for v in sorted(hist_dict.items())]
hist_values = [v[1] for v in sorted(hist_dict.items())]
cumul_hist_vals = np.cumsum(hist_values)

# Equalize the histogram
cumul_hist_min = min(cumul_hist_vals)
hist_eq_denom = normal_im.shape[0] * normal_im.shape[1] - cumul_hist_min
hist_eq = [round(((v - cumul_hist_min) / hist_eq_denom) * 255) for v in cumul_hist_vals]
cumul_hist_eq = np.cumsum(hist_eq)

# Determine the equalized image
value_map = dict(zip(hist_keys, hist_eq))
eq_im = np.array([value_map[v] for v in plot_hist_vals]).reshape(normal_im.shape)

# Plot the data
# Gray-level histogram
plt.subplot(421)
plt.hist(plot_hist_vals, bins=256)
# Cumulative histogram
plt.subplot(423)
plt.bar(range(len(cumul_hist_vals)), cumul_hist_vals)
# Gray-Level equalized histogram
plt.subplot(425)
plt.bar(hist_eq, hist_values)
# Cumulative equalized histogram
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
