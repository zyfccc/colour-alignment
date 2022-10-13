import numpy as np
import scipy.stats as ss

ITEMS = []

def normalise_colour(color):
    return color * 255.0 / np.max(color)

def linear_intensity_aligning(colours1_intensity, colours2_intensity):
    points1 = np.array([colours1_intensity[0], colours1_intensity[-1]])
    points2 = np.array([colours2_intensity[0], colours2_intensity[-1]])
    slope, intercept, r_value, p_value, std_err = ss.linregress(
        points2, points1)
    colours2_intensity =  colours2_intensity * slope + intercept
    return colours2_intensity


def linear_intensity_matching(colours1, colours2, items=ITEMS, optimal=True):
    '''
    optimal: whether to use the proposed optimal intensity and chromaticity offset and scaling for demonstration and comparison.
    '''
    tmp1 = []
    tmp2 = []
    if len(items) != 0:
        for i in range(len(colours1)):
            if i in items:
                tmp1.append(colours1[i])
                tmp2.append(colours2[i])
        tmp1 = np.array(tmp1)
        tmp2 = np.array(tmp2)
    else:
        tmp1 = colours1
        tmp2 = colours2
    # match colors2 to colors1 
    colours1_intensity = np.sum(tmp1, axis=1)
    colours2_intensity = np.sum(tmp2, axis=1)
    slope, intercept, r_value, p_value, std_err = ss.linregress(
        colours2_intensity, colours1_intensity)
    if optimal:
        colours2 = intensity_scaling(colours2, slope)
        colours2 = intensity_offset(colours2, intercept)
    else:
        colours2 = colours2 * slope
        colours2 = colours2 + intercept / 3
    return colours2


def intensity_scaling(colours, scale):
    return colours * scale


def intensity_offset(colours, offset):
    old_B = colours[:,0]
    old_G = colours[:,1]
    old_R = colours[:,2]
 
    dividend = (old_B+old_G+old_R+offset)*old_B
    divisor = (old_B+old_G+old_R)
    B = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)

    dividend = (old_B+old_G+old_R+offset)*old_R
    divisor = (old_B+old_G+old_R)
    R = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor!=0)
    G = old_B+old_G+old_R+offset-B-R
    new_colours = np.dstack((B, G, R))[0]
    return new_colours


def chromaticity_scaling(colours, scale_r, scale_b):
    intensity = np.sum(colours, axis=1)
    old_B = colours[:,0]
    old_R = colours[:,2]
    B = np.multiply(scale_b, old_B)
    R = np.multiply(scale_r, old_R)
    G = intensity - B - R
    colours = np.dstack((B, G, R))[0]
    return colours


def chromaticity_offset(colours, offset_r, offset_b):
    intensity = np.sum(colours, axis=1)
    old_B = colours[:,0]
    old_G = colours[:,1]
    old_R = colours[:,2]
    B = offset_b*(old_B+old_G+old_R)+old_B
    R = offset_r*(old_B+old_G+old_R)+old_R
    G = intensity - B - R
    colours = np.dstack((B, G, R))[0]
    return colours


def linear_chromaticity_matching(colours1, colours2, items=ITEMS):
    tmp1 = []
    tmp2 = []
    if len(items) != 0:
        for i in range(len(colours1)):
            if i in items:
                tmp1.append(colours1[i])
                tmp2.append(colours2[i])
        tmp1 = np.array(tmp1)
        tmp2 = np.array(tmp2)
    else:
        tmp1 = colours1
        tmp2 = colours2

    # match colors2 to colors1
    intensity_1 = tmp1[:, 0] + tmp1[:, 1] + tmp1[:, 2]
    intensity_2 = tmp2[:, 0] + tmp2[:, 1] + tmp2[:, 2]
    a = tmp1[:, 2]
    b = tmp2[:, 2]
    c = tmp1[:, 0]
    d = tmp2[:, 0]
    ratio_r_1 =  np.divide(a, intensity_1, out=np.zeros_like(a), where=intensity_1!=0)
    ratio_r_2 =  np.divide(b, intensity_2, out=np.zeros_like(b), where=intensity_2!=0)
    ratio_b_1 =  np.divide(c, intensity_1, out=np.zeros_like(c), where=intensity_1!=0)
    ratio_b_2 =  np.divide(d, intensity_2, out=np.zeros_like(d), where=intensity_2!=0)

    idx1 = intensity_1 > 0.2

    ratio_r_1 = ratio_r_1[idx1]
    ratio_r_2 = ratio_r_2[idx1]
    ratio_b_1 = ratio_b_1[idx1]
    ratio_b_2 = ratio_b_2[idx1]

    slope_blue, offset_blue, r_value, p_value, std_err1 = ss.linregress(
        ratio_b_2, ratio_b_1)
    slope_red, offset_red, r_value, p_value, std_err1 = ss.linregress(
        ratio_r_2, ratio_r_1)

    colours2 =  chromaticity_scaling(colours2, slope_red, slope_blue)
    colours2 =  chromaticity_offset(colours2, offset_red, offset_blue)

    return colours2

