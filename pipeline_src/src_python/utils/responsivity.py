"""
Created on Sat May 17 16:29:13 2025

@author: jctourtellotte
"""
# outside packages

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import scipy

# bright_img_path - path to bright image (string)
# dark_img_path - path to dark image (string)
def detector_responsivity_determ(bright_img_path, dark_img_path):
    bright_image = tifffile.imread(bright_img_path)
    dark_image = tifffile.imread(dark_img_path)
    bg_var = np.var(dark_image)
    # create frame represenative of the average at each x,y position
    dark_mean = np.mean(dark_image, 0) 
    # from the bright image, subtract the overall mean value of the
    #   represenative dark frame
    bg_corrected = bright_image[:, :, : ] - np.mean(dark_mean)
    
    #QUESTION: why do this correction instead of
    #   subtracting the rep frame from the brightfield frame? (can do because CCD)
    
    # Normalization of data (bleaching correction)
    #   such that the average intensity per plane remains constant over time
    firstplane_avg = np.mean(bg_corrected[0, :, :])
    mean_array = np.mean(bg_corrected, (1, 2))
    
    dim_array = np.ones((1, bg_corrected.ndim), int).ravel()
    dim_array[0] = -1
    b_reshaped = (firstplane_avg / mean_array).reshape(dim_array)
    bg_corrected = bg_corrected * b_reshaped
    
    mean = np.mean(bg_corrected, 0)
    variance = np.var(bg_corrected, 0)
    
    # calculating gain
    meanvar_binned_stats = scipy.stats.binned_statistic(mean.flatten(),
                                               variance.flatten(),
                                               bins = 100, # QUESTION: should this be hard coded?
                                               statistic = 'mean')
    weights, _ = np.histogram(mean.flatten(),
                              bins = meanvar_binned_stats.bin_edges)
    weights = weights / np.sum(weights)
    center = (meanvar_binned_stats.bin_edges[1:] + meanvar_binned_stats.bin_edges[:-1]) / 2
    
    # remove nans and fit
    nanvalues = np.isnan(meanvar_binned_stats.statistic)
    fit = np.polyfit(center[~nanvalues], 
                     meanvar_binned_stats.statistic[~nanvalues], 
                     1, 
                     w=weights[~nanvalues])
    
    # data needed to plot mean ADU vs variance ADU
    meanvar_plot_data = {"center": center,
                         "statistic": meanvar_binned_stats.statistic,
                         "fit": fit,
                         "weights": weights}

    gain = 1/fit[0]
    
    calibration_stats = {"gain": gain,
                         "offset": np.mean(dark_mean),
                         "dk_noise": gain * np.sqrt(bg_var)}
    
    print(f"\n---Detector Responsivity Determination---\ngain = {calibration_stats['gain']} \noffset = {calibration_stats['offset']}")
    
    return calibration_stats, meanvar_plot_data


# def plot_meanvar(meanvarplot):
#     center = meanvarplot['center']
#     fit = meanvarplot['fit']
#     meanvar_statistic = meanvarplot['statistic']
#     weights = meanvarplot['weights']
    
    
#     fig, ax = plt.subplots()
#     ax.scatter(center,
#                meanvar_statistic,
#                marker='x',
#                label='mean variance in bin'
#                )
#     ax.plot(center, 
#             np.polyval(fit, center), 
#             color='red', 
#             label='fit = ' + str(np.around(fit[0], 3)) + 'x + ' + str(np.around(fit[1], 3)))
#     ax.set_xlabel('mean ADU')
#     ax.set_ylabel('Variance ADU')
#     ax2 = ax.twinx()
#     ax2.plot(center, weights, label='weights')
#     ax2.set_ylabel("Weights (prop to pixels)", color="blue")
#     ax2.set_yscale('log')
#     fig.legend()
    
#     return fig
#     # fig.savefig(self.resultprefix + 'calibration.png',
#     #             format='png',
#     #             dpi=100,
#     #             bbox_inches='tight')
#     # plt.close('all')