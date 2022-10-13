import cv2
import time
import os
import numpy as np
import argparse


CRF = "crf.txt"

def intensity_scaling(colours, scale):
    b = colours[:, 0]
    g = colours[:, 1]
    r = colours[:, 2]
    b = np.multiply(b, scale)
    g = np.multiply(g, scale)
    r = np.multiply(r, scale)
    return np.dstack((b, g, r))[0]


def correct_intensity(colours, crf):
    intensity = np.sum(colours, axis=1) / 3
    x = np.linspace(0, 1, crf.size)
    new_intensity = np.interp(intensity, x, crf)
    intensity[intensity == 0] = 1E-6
    ratios = np.divide(new_intensity, intensity)
    return intensity_scaling(colours, ratios)


def linearise_img(crf_path, img_path, output_path):
    predictions = np.loadtxt(crf_path + CRF, dtype=np.float)

    start = time.time()

    image_names = os.listdir(img_path)
    for name in image_names:
        image = cv2.imread(img_path + '/' + name, cv2.IMREAD_COLOR)

        x, y, c = image.shape
        colours = np.reshape(image, (x * y, 3))

        colours_norm = colours / 255.0
        colours_norm = correct_intensity(colours_norm, predictions)

        reshaped_back = np.reshape(colours_norm, (x, y, 3))
        reshaped_back = reshaped_back * 255.0

        if not cv2.imwrite(output_path + name, reshaped_back):
            raise Exception("Could not write image")

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./datasets/modified_Middlebury_test/', help='path to the image dataset')
    parser.add_argument('--crf_save_path', type=str, default='./results/crfs/', help='name of the tags json file')
    parser.add_argument('--output_path', type=str, default='./results/images/', help='path to store the response lienarised images')
    args = parser.parse_args()
    img_path = args.img_path
    crf_save_path = args.crf_save_path
    output_path = args.output_path

    folders = []
    files = os.listdir(img_path)
    for file in files:
        if os.path.isdir(os.path.join(img_path, file)):
            folders.append(file)

    for folder in folders:
        img_path_camera = os.path.join(img_path, folder + '/')
        crf_path_camera = os.path.join(crf_save_path, folder + '/')
        output_path_camera = os.path.join(output_path, folder + '/')
        linearise_img(crf_path_camera, img_path_camera, output_path_camera)
