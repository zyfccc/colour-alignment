import cv2
import os
import json
import argparse
import numpy as np
import libs.MathUtil as util
import libs.QcImage as QcImage
import libs.ColourMatching as ColourMatching
from libs.TrainingSet import TrainingSet

INTENSITY_MATCHING = True
CHROMATICITY_MATCHING = True

# items = [14, 3, 19]
# items = [14, 3, 19, 11]
# items = [14, 3, 15, 11]
# items = [19, 20, 21, 22]
# items = [19, 3]
# items = [19, 17]
# items = [19, 14]
# items = linspace(0,23,24,dtype=np.int32)

def get_colours(path, ts, start=0, end=24):

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    colours = []

    for i in range(len(ts.references)):
        anno = ts.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            image, anno.position, anno.rect)
        sample_bgr = QcImage.get_average_rgb(colour_area)

        colours.append([sample_bgr[0], sample_bgr[1], sample_bgr[2]])

    sub_array = np.array(colours)[start:end]

    return sub_array

def full_colour_linear_matching(colours1, colours2, items):
    new_items = np.array(items)

    if INTENSITY_MATCHING:
        colours2 = ColourMatching.linear_intensity_matching(colours1, colours2, items=new_items, optimal=True)

    if CHROMATICITY_MATCHING:
        colours2 = ColourMatching.linear_chromaticity_matching(colours1, colours2, items=new_items)

    return colours2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./results/images/', help='path to the images to be matched')
    parser.add_argument('--tag_path', type=str, default='./datasets/modified_Middlebury_test/tags.json', help='path to the label json file')
    parser.add_argument('--items', type=str, default='19,14', help='patch indexes for matching')
    parser.add_argument('--intensity', type=str, default=True, help='whether to perform intensity matching')
    parser.add_argument('--chromaticity', type=str, default=True, help='whether to perform chromaticity matching')
    args = parser.parse_args()
    img_path = args.img_path
    tag_path = args.tag_path
    items = [int(x) for x in str(args.items).split(',')]
    INTENSITY_MATCHING = args.intensity
    CHROMATICITY_MATCHING = args.chromaticity

    print('Intensity matching: ' + str(INTENSITY_MATCHING))
    print('Chromaticity matching: ' + str(CHROMATICITY_MATCHING))
    print("items: ", items)

    folders = os.listdir(img_path)   

    with open(tag_path) as json_data:
        obj = json.load(json_data)[0]
    ts = TrainingSet(obj)
        
    angular_errors = []
    distance_errors = []
    colours_A = []
    colours_B = []
    deltaE00_errors = []

    for dirA in folders:
        if not os.path.isdir(os.path.join(img_path, dirA)):
            continue
        image_namesA = os.listdir(os.path.join(img_path, dirA))


        for nameA in image_namesA:

            pathA = os.path.join(img_path, dirA + '/' + nameA)
            coloursA = get_colours(pathA, ts)

            for nameB in image_namesA:

                if nameA == nameB:
                    continue

                pathB = os.path.join(img_path, dirA + '/' + nameB)
                coloursB = get_colours(pathB, ts)

                coloursA = ColourMatching.normalise_colour(coloursA)
                coloursB = ColourMatching.normalise_colour(coloursB)

                coloursB = full_colour_linear_matching(coloursA, coloursB, items)

                colours_A.append(coloursA)
                colours_B.append(coloursB)

                RAE = util.angle(coloursA, coloursB)
                RMSE = util.rmse(coloursA, coloursB)
                DeltaE00 = util.deltaE2000(coloursA, coloursB)
                angular_errors.append(RAE)
                distance_errors.append(RMSE)
                deltaE00_errors.append(DeltaE00)

    angular_errors = np.array(angular_errors)
    distance_errors = np.array(distance_errors)
    deltaE00_errors = np.array(deltaE00_errors)
    colours_A_np = np.array(colours_A)
    colours_B_np = np.array(colours_B)

    rmses_A = []
    angulars_A = []
    deltaE00_A = []

    for idx in range(colours_A_np.shape[1]):
        array1 = colours_A_np[:, idx, :]
        array2 = colours_B_np[:, idx, :]
        rmses_A.append(util.rmse(array1, array2))
        angulars_A.append(util.angle(array1, array2))
        deltaE00_A.append(util.deltaE2000(array1, array2))

    print('single camera============')

    angular_errors = angular_errors * 180 / np.pi
    angulars_A = np.array(angulars_A) * 180 / np.pi
    len25 = int(angular_errors.size * 0.25)

    print("Number of comparisons: " + str(angular_errors.size))
    print("Angle error 1 is: " + str(angulars_A))
    print("Mean angle error 1 is: " + str(np.mean(angular_errors)))
    print("Median angle error 1 is: " + str(np.median(angular_errors)))
    angular_errors = np.sort(angular_errors)
    print("Best 25P angle error 1 is: " +
        str(np.mean(angular_errors[0:len25])))
    print("Worst 25P angle error 1 is: " +
        str(np.mean(angular_errors[-len25:])))
    print("95 Percentile angle error 1 is: " +
        str(np.percentile(angular_errors, 95)))

    print("RMS error 1 is: " + str(rmses_A))
    print("Mean RMS error 1 is: " + str(np.mean(distance_errors)))
    print("Median RMS error 1 is: " + str(np.median(distance_errors)))
    distance_errors = np.sort(distance_errors)
    print("Best 25P RMS error 1 is: " +
        str(np.mean(distance_errors[0:len25])))
    print("Worst 25P RMS error 1 is: " +
        str(np.mean(distance_errors[-len25:])))
    print("95 Percentile RMS error 1 is: " +
        str(np.percentile(distance_errors, 95)))

    print("DeltaE00 is: " + str(deltaE00_A))
    print("Mean DeltaE00 is: " + str(np.mean(deltaE00_errors)))
    print("Median DeltaE00 is: " + str(np.median(deltaE00_errors)))

    angular_errors = []
    distance_errors = []
    deltaE00_errors = []
    colours_A = []
    colours_B = []

    for dirA in folders:
        if not os.path.isdir(os.path.join(img_path, dirA)):
            continue
        image_namesA = os.listdir(os.path.join(img_path, dirA))

        for dirB in folders:
            if not os.path.isdir(os.path.join(img_path, dirB)):
                continue
            image_namesB = os.listdir(os.path.join(img_path, dirB))

            for nameA in image_namesA:

                pathA = os.path.join(img_path, dirA + '/' + nameA)
                coloursA = get_colours(pathA, ts, 0, 24)

                for nameB in image_namesB:

                    pathB = os.path.join(img_path, dirB + '/' + nameB)
                    coloursB = get_colours(pathB, ts, 0, 24)

                    coloursA = ColourMatching.normalise_colour(coloursA)
                    coloursB = ColourMatching.normalise_colour(coloursB)

                    coloursB = full_colour_linear_matching(coloursA, coloursB, items)

                    colours_A.append(coloursA)
                    colours_B.append(coloursB)

                    RAE = util.angle(coloursA, coloursB)
                    RMSE = util.rmse(coloursA, coloursB)
                    DeltaE00 = util.deltaE2000(coloursA, coloursB)
                    angular_errors.append(RAE)
                    distance_errors.append(RMSE)
                    deltaE00_errors.append(DeltaE00)

    angular_errors = np.array(angular_errors)
    distance_errors = np.array(distance_errors)
    deltaE00_errors = np.array(deltaE00_errors)
    colours_A_np = np.array(colours_A)
    colours_B_np = np.array(colours_B)

    rmses_A = []
    angulars_A = []
    deltaE00_A = []

    for idx in range(colours_A_np.shape[1]):
        array1 = colours_A_np[:, idx, :]
        array2 = colours_B_np[:, idx, :]
        rmses_A.append(util.rmse(array1, array2))
        angulars_A.append(util.angle(array1, array2))
        deltaE00_A.append(util.deltaE2000(array1, array2))

    print('Across camera============')

    angular_errors = angular_errors * 180 / np.pi
    angulars_A = np.array(angulars_A) * 180 / np.pi
    len25 = int(angular_errors.size * 0.25)

    print("Number of comparisons: " + str(angular_errors.size))
    print("Angle error 2 is: " + str(angulars_A))
    print("Mean angle error 2 is: " + str(np.mean(angular_errors)))
    print("Median angle error 2 is: " + str(np.median(angular_errors)))
    angular_errors = np.sort(angular_errors)
    print("Best 25P angle error 2 is: " +
        str(np.mean(angular_errors[0:len25])))
    print("Worst 25P angle error 2 is: " +
        str(np.mean(angular_errors[-len25:])))
    print("95 Percentile angle error 2 is: " +
        str(np.percentile(angular_errors, 95)))

    print("RMS error 2 is: " + str(rmses_A))
    print("Mean RMS error 2 is: " + str(np.mean(distance_errors)))
    print("Median RMS error 2 is: " + str(np.median(distance_errors)))
    distance_errors = np.sort(distance_errors)
    print("Best 25P RMS error 2 is: " +
        str(np.mean(distance_errors[0:len25])))
    print("Worst 25P RMS error 2 is: " +
        str(np.mean(distance_errors[-len25:])))
    print("95 Percentile RMS error 2 is: " +
        str(np.percentile(distance_errors, 95)))

    print("DeltaE00 is: " + str(deltaE00_A))
    print("Mean DeltaE00 is: " + str(np.mean(deltaE00_errors)))
    print("Median DeltaE00 is: " + str(np.median(deltaE00_errors)))