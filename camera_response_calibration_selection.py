import cv2
import json
import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import libs.QcImage as QcImage
from libs.TrainingSet import TrainingSet
import libs.ColourMatching as ColourMatching


VIS = True


def read_DoRF():
    curves = []
    with open('datasets/dorf/dorfCurves.txt') as f:
        while f.readline() is not '':
            scale = f.readline()
            f.readline()
            f.readline()
            f.readline()
            line = f.readline()
            # if 'lin-lin' not in scale:
            #     continue
            strings = line.split('   ')
            nums = []
            for string in strings:
                nums.append(float(string))
            curves.append(nums)
    return np.array(curves)


def get_colours(path, ts, start=None, end=None):

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    colours = []

    for i in range(len(ts.references)):
        anno = ts.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            image, anno.position, anno.rect)
        sample_bgr = QcImage.get_average_rgb(colour_area)

        colours.append([sample_bgr[0], sample_bgr[1], sample_bgr[2]])

    return np.array(colours)[start:end]


def select_crf(np_cali_colors_b, np_cali_colors_g, np_cali_colors_r, crfs):
    min_error = sys.maxsize
    best_crf = None
    raw_intensity = (np_cali_colors_b + np_cali_colors_g +
                     np_cali_colors_r) / 3

    # sort arrays
    mean_intensity = np.mean(raw_intensity, axis=0)
    idx = mean_intensity.argsort()
    mean_intensity = mean_intensity[idx]
    sorted_intensity = raw_intensity[:, idx]
    xx, yy = sorted_intensity.shape


    for n in range(len(crfs)):
        crf = crfs[n]
        x = np.linspace(0, 1, crf.size)
        crf = np.interp(x, crf, x)
        corrected_intensity = np.interp(sorted_intensity, x, crf)

        aligned_corrected_items = []
        corrected_intensities = []
        for i in range(xx):
            for j in range(i, xx):
                # align intensities to the mean
                if i == j:
                    continue
                item = corrected_intensity[i]
                reference = corrected_intensity[j]
                aligned_item = ColourMatching.linear_intensity_aligning([
                                                                        0, 1], item)
                aligned_reference = ColourMatching.linear_intensity_aligning([
                                                                             0, 1], reference)
                corrected_intensities.append(item)
                distance = np.abs(aligned_item - aligned_reference)
                aligned_corrected_items.append(distance)
        corrected_intensities = np.array(corrected_intensities)
        aligned_corrected_items = np.array(aligned_corrected_items)


        interpreted = []
        start = np.min(corrected_intensities)
        end = np.max(corrected_intensities)
        sampling_x = np.linspace(start, end, 100)
        for i in range(aligned_corrected_items.shape[0]):
            item = aligned_corrected_items[i]
            corrected_intensity = corrected_intensities[i]
            sampled_corrected_items = np.interp(sampling_x, corrected_intensity, item)
            interpreted.append(sampled_corrected_items)

        interpreted = np.array(interpreted)

        mean_interpreted = np.mean(interpreted, axis=0)

        # # area under curve
        g = asymmetry_coefficient(mean_interpreted)
        auc = np.sum(mean_interpreted - np.min(mean_interpreted))
        offset = (end + start - 1)
        cost = np.mean(np.square([
            np.abs(g - 3*offset), 
            auc]))

        if cost < min_error:
            min_error = cost
            best_crf = crf

    print(min_error)

    return best_crf

def asymmetry_coefficient(x):
    steps = np.linspace(0,0.99,100)
    n = np.sum(x)
    mean = np.sum(np.multiply(steps, x)) / n
    K2 = np.multiply(x, np.square(steps - mean))
    K3 = np.multiply(x, np.power(steps - mean, 3))
    K2 = np.sum(K2)/100
    K3 = np.sum(K3)/100
    return K3 / (K2 * np.sqrt(K2))


def calibrate_crf(img_path, tag_path, crf_path):
    with open(tag_path) as json_data:
        obj = json.load(json_data)[0]
    ts = TrainingSet(obj)

    cali_colours_b = []
    cali_colours_g = []
    cali_colours_r = []

    image_names = os.listdir(img_path)
    for name in image_names:
        image_path = img_path + name
        colours = get_colours(image_path, ts,start=0, end=24)
        colours /= 255.0
        cali_colours_b.append(colours[:, 0])
        cali_colours_g.append(colours[:, 1])
        cali_colours_r.append(colours[:, 2])

    cali_colours_b = np.array(cali_colours_b)
    cali_colours_g = np.array(cali_colours_g)
    cali_colours_r = np.array(cali_colours_r)

    # camera calibration using the Selection approach
    start = time.time()
    crfs = read_DoRF()
    crf = select_crf(cali_colours_b, cali_colours_g, cali_colours_r, crfs)
    np.savetxt(crf_path, crf, fmt='%f')
    end = time.time()
    print("time: " + str(end - start))

    if VIS:
        visualise_crf(crf)

    return crf



def visualise_crf(crf):
    x = np.linspace(0, 1, crf.size)
    plt.plot(x, crf, color='g')
    plt.plot(x, x, color='y')
    plt.show()



CRF = "crf.txt"

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./datasets/modified_Middlebury_calibration/', help='path to the image dataset')
    parser.add_argument('--tag_path', type=str, default='./datasets/modified_Middlebury_calibration/tags.json', help='path to the label json file')
    parser.add_argument('--crf_save_path', type=str, default='./results/crfs/', help='name of the tags json file')
    parser.add_argument('--vis', type=int, default=True, help='whether to visualise the generated CRF')
    args = parser.parse_args()
    img_path = args.img_path
    tag_path = args.tag_path
    crf_save_path = args.crf_save_path
    VIS = args.vis

    folders = []
    files = os.listdir(img_path)
    for file in files:
        if os.path.isdir(os.path.join(img_path, file)):
            folders.append(file)

    # calibrate CRF of each camera
    for folder in folders:
        img_path_camera = os.path.join(img_path, folder + '/')
        crf_path_camera = os.path.join(crf_save_path, folder + '/crf.txt')
        os.makedirs(os.path.join(crf_save_path, folder), exist_ok=True)
        calibrate_crf(img_path_camera, tag_path, crf_path_camera)
    
