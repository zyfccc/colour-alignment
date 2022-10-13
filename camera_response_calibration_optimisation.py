import os
import cv2
import json
import sys
import math
import argparse
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import libs.QcImage as QcImage
from libs.TrainingSet import TrainingSet
import libs.ColourMatching as ColourMatching


VIS = True

random_init_num_crf = 50


def get_colours(path, ts, start=0, end=24):

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    colours = []

    for i in range(len(ts.references)):
        anno = ts.references[i]
        colour_area = QcImage.crop_image_by_position_and_rect(
            image, anno.position, anno.rect)
        sample_bgr = QcImage.get_average_rgb(colour_area)

        colours.append([sample_bgr[0], sample_bgr[1], sample_bgr[2]])

    return np.array(colours)[start:end]


def train_crf(np_cali_intensities, H):

    #############
    # input
    #############
    xx, yy = np_cali_intensities.shape
    input_cali_intensities = tf.placeholder(tf.float64, shape=[xx, yy])

    colours_intensity = input_cali_intensities

    #############
    # crf
    #############
    params = tf.Variable(tf.random.normal(
        [H.shape[0] - 2, 1], mean=0.0, stddev=10.0, dtype=tf.float64), name='params')
    x_axis = tf.constant(H[0, :], dtype=tf.float64)
    H_mean = tf.constant(H[1, :], dtype=tf.float64)
    H_variance = tf.constant(H[2:, :], dtype=tf.float64)
    predicted_crf = tf.add(H_mean,
                           tf.matmul(tf.transpose(params), H_variance))
    x = tf.reshape([[x_axis]], [1, -1, 1])
    y = tf.reshape([[predicted_crf]], [1, -1, 1])

    colours_intensity_resized = tf.reshape(
        [[colours_intensity]], [1, -1, 1])
    colours_intensity_resized_corrected = tf.contrib.image.interpolate_spline(
        x, y, colours_intensity_resized, order=1)
    colours_intensity = tf.reshape(
        colours_intensity_resized_corrected, [xx, yy])

    # micro-smoothness of crf
    predicted_crf = tf.reshape(predicted_crf, [-1])
    size2 = tf.size(predicted_crf)
    s1 = tf.slice(predicted_crf, [1], [size2 - 1])
    s2 = tf.slice(predicted_crf, [0], [size2 - 1])
    d1 = tf.subtract(s1, s2)
    d1_abs = d1 - tf.abs(d1)
    size3 = tf.size(d1)
    s3 = tf.slice(d1, [1], [size3 - 1])
    s4 = tf.slice(d1, [0], [size3 - 1])
    d2 = tf.subtract(s3, s4)
    micro_smooth_crf = tf.reduce_max(d1) - tf.reduce_mean(d1)
    micro_smooth_crf2 = tf.reduce_mean(tf.abs(d2))

    # macro-smoothness of crf
    samples = tf.gather(
        predicted_crf, [0, 102, 204, 306, 408, 510, 612, 714, 816, 918, 1023])
    sample_size = tf.size(samples)
    s1_samples = tf.slice(samples, [1], [sample_size - 1])
    s2_samples = tf.slice(samples, [0], [sample_size - 1])
    d1_samples = tf.subtract(s1_samples, s2_samples)
    s3_samples = tf.slice(d1_samples, [1], [sample_size - 2])
    s4_samples = tf.slice(d1_samples, [0], [sample_size - 2])
    d2_samples = tf.subtract(s3_samples, s4_samples)
    macro_smooth_crf = tf.reduce_max(d1_samples) - tf.reduce_mean(d1_samples)
    macro_smooth_crf2 = tf.reduce_mean(tf.abs(d2_samples))

    cost_items = []
    intensity_items = []
    aligned_intensity_items = []

    #############
    # Linear matchings
    ############
    for idx_i in range(xx):
        for idx_j in range(idx_i, xx):
            if idx_i == idx_j:
                continue
            reference_intensities = tf.reshape(
                tf.gather(colours_intensity, [idx_i]), shape=[yy])
            colour_intensities = tf.reshape(
                tf.gather(colours_intensity, [idx_j]), shape=[yy])

            gathered_cols = tf.gather(colour_intensities, [0, yy-2, yy-1])
            reference_cols = tf.gather(reference_intensities, [0, yy-2, yy-1])

            ref = tf.constant([0.0, 1.0, 1.0], dtype=tf.float64)

            W_intensity, b_intensity = estimate_coefficients1d(
                gathered_cols, ref)

            aligned_colour_intensities = colour_intensities * W_intensity + b_intensity

            W_intensity2, b_intensity2 = estimate_coefficients1d(
                reference_cols, ref)
            reference_intensities = reference_intensities * W_intensity2 + b_intensity2

            #############
            # Cost
            #############
            distance = tf.abs(aligned_colour_intensities - reference_intensities)

            cost_items.append(distance)
            intensity_items.append(colour_intensities)
            aligned_intensity_items.append(aligned_colour_intensities)

    interpreted = []
    start = tf.reduce_min(intensity_items)
    end = tf.reduce_max(intensity_items)
    sampling_x = tf.linspace(start, end, 100)
    for i in range(len(cost_items)):
        item = cost_items[i]
        intensity = intensity_items[i]

        x = tf.reshape([[intensity]], [1, -1, 1])
        y = tf.reshape([[item]], [1, -1, 1])

        sampling_x_resized = tf.reshape(
            [[sampling_x]], [1, -1, 1])
        sampling_x_resized_corrected = tf.contrib.image.interpolate_spline(
            x, y, sampling_x_resized, order=1)
        sampling_x = tf.reshape(
            sampling_x_resized_corrected, [100])

        interpreted.append(sampling_x)

    mean_interpreted = tf.reduce_mean(interpreted, axis=0)

    g = asymmetry_coefficient_tf(mean_interpreted)
    offset = (end + start - 1)
    auc = tf.reduce_mean(mean_interpreted - np.min(mean_interpreted))

    loss = tf.global_norm([
        1E2 * tf.abs(g - 3*offset), 
        1E2 * auc,
        1E2 * micro_smooth_crf,
        1E5 * micro_smooth_crf2,
        1E1 * macro_smooth_crf,
        1E2 * macro_smooth_crf2,
        1E10 * d1_abs,  # monotony
        1E10 * (predicted_crf - tf.clip_by_value(predicted_crf, 0.0, 1.0))
        ])

    # training algorithm
    random_init_num = random_init_num_crf
    escape_rate_crf = 1E-3
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        0.5, global_step, 1000, 0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    res_params = None
    min_error = sys.maxsize

    for j in range(random_init_num):

        # initializing the variables
        init = tf.global_variables_initializer()

        # starting the session session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # optimise GPU memory
        sess = tf.Session(config=config)
        sess.run(init)

        epoch = 2000
        prev_training_cost = sys.maxsize

        for step in range(epoch):
            _, training_cost = sess.run(
                [optimizer, loss], feed_dict={
                    input_cali_intensities: np_cali_intensities})

            if math.isinf(training_cost) or math.isnan(training_cost):
                break

            if np.abs(prev_training_cost - training_cost) <= escape_rate_crf:
                break

            if step % 10 == 0:
                print(training_cost)

            prev_training_cost = training_cost

        if min_error > prev_training_cost:
            min_error = prev_training_cost

            tmp_params = params.eval(session=sess)
            res_params = tmp_params
            res_params = res_params.reshape((res_params.size))

            micro1 = micro_smooth_crf.eval(session=sess)
            micro2 = micro_smooth_crf2.eval(session=sess)
            macro1 = macro_smooth_crf.eval(session=sess)
            macro2 = macro_smooth_crf2.eval(session=sess)

            print('crf=========')
            print(res_params)
            print(min_error)
            print(micro1)
            print(micro2)
            print(macro1)
            print(macro2)
            print('=========')

    return res_params

def estimate_coefficients1d(x, y):
    n = tf.cast(tf.shape(x), tf.float64)
    mean_x = tf.reduce_mean(x)
    mean_y = tf.reduce_mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = tf.reduce_sum(y * x) - mean_y * mean_x * n
    SS_xx = tf.reduce_sum(x * x) - mean_x * mean_x * n

    # calculating regression coefficients
    # SS_xx = tf.Print(SS_xx, [SS_xx], 'SS_xx', summarize=50)
    b_1 = tf.div_no_nan(SS_xy, SS_xx)
    b_0 = mean_y - b_1 * mean_x
    return b_1, b_0


def asymmetry_coefficient_tf(x):
    steps = tf.cast(tf.linspace(0.0, 0.99, 100), tf.float64)
    n = tf.reduce_sum(x)
    mean = tf.reduce_sum(tf.multiply(steps, x)) / n
    K2 = tf.multiply(x, tf.square(steps - mean))
    K3 = tf.multiply(x, tf.pow(steps - mean, 3))
    K2 = tf.reduce_sum(K2)/100
    K3 = tf.reduce_sum(K3)/100
    return K3 / (K2 * tf.sqrt(K2))


def calibrate_crf(img_path, tag_path, crf_path):
    with open(tag_path) as json_data:
        obj = json.load(json_data)[0]
    ts = TrainingSet(obj)

    cali_intensities = []

    image_names = os.listdir(img_path)
    for name in image_names:
        # Using colour patches on the colour chart
        colours = get_colours(os.path.join(img_path, name), ts)
        colours /= 255.0
        cali_intensities.append((colours[:, 0]+colours[:, 1]+colours[:, 2]) / 3)

    cali_intensities = np.array(cali_intensities)

    cali_intensities_mean = np.mean(cali_intensities, axis=0)

    sort_idx = cali_intensities_mean.argsort()
    cali_intensities_mean = cali_intensities_mean[sort_idx]
    cali_intensities = cali_intensities[:, sort_idx]

    xx, yy = cali_intensities.shape

    aligned_intensities = []
    aligned_items = []
    for i in range(xx):
        for j in range(xx):
            if i==j:
                continue
            # align intensities to the mean
            item = cali_intensities[i]
            reference = cali_intensities[j]
            aligned_item = ColourMatching.linear_intensity_aligning(
                [0.0, 1.0], item)
            aligned_reference = ColourMatching.linear_intensity_aligning(
                [0.0, 1.0], reference)
            aligned_intensities.append(aligned_item)
            aligned_items.append(np.abs(aligned_item - aligned_reference))
    aligned_intensities = np.array(aligned_intensities)
    aligned_items = np.array(aligned_items)

    # normalisation
    normalised_aligned_intensities = aligned_items[:, 1:-1] / aligned_intensities[:, 1:-1]

    ori_mean = np.mean(normalised_aligned_intensities, axis=0)
    selected_idxs = ori_mean <= np.percentile(ori_mean, 90)
    idxs = np.append(np.append(np.array([True]), selected_idxs), np.array([True]))
    cali_intensities = cali_intensities[:, idxs]

    # Training
    try:
        # check if already exist
        crf = np.loadtxt(crf_path, dtype=np.float)
    except:
        start = time.time()
        H = QcImage.read_matrix("datasets/dorf/invemor.txt", 6)
        params = train_crf(cali_intensities, H)
        crf = QcImage.predict(H, params)
        np.savetxt(crf_path, crf, fmt='%f')
        end = time.time()
        print("time: " + str(end - start))
    if VIS:
        visualise_crf(crf)

    return crf

def visualise_crf(crf):
    x = np.linspace(0, 1, 1024)
    plt.plot(x, crf, color='g')
    plt.plot(x, x, color='y')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./datasets/modified_Middlebury_calibration/', help='path to the image dataset')
    parser.add_argument('--tag_path', type=str, default='./datasets/modified_Middlebury_calibration/tags.json', help='path to the label json file')
    parser.add_argument('--crf_save_path', type=str, default='./results/crfs/', help='name of the tags json file')
    parser.add_argument('--vis', type=bool, default=True, help='whether to visualise the generated CRF')
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
