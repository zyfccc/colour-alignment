import numpy as np



RECT_SCALE = 1000


def get_average_rgb(image_data):
    return np.average(image_data, axis=(0, 1))


def crop_image_by_position_and_rect(cv_image, position, rect):
    # img[y: y + h, x: x + w]
    height = cv_image.shape[0]
    width = cv_image.shape[1]
    position_x = position.x * width
    position_y = position.y * height
    rect_x = width * rect.x / RECT_SCALE
    rect_y = height * rect.y / RECT_SCALE
    return cv_image[int(position_y):int(position_y) + int(rect_y),
                    int(position_x):int(position_x) + int(rect_x)]


def read_matrix(path, n_params):
    H = None
    line_arr = np.array([])
    count = 0
    with open(path) as f:
        f.readline()
        for line in f:
            if "=" in line:
                count += 1
                if H is None:
                    H = line_arr
                else:
                    H = np.vstack((H, line_arr))
                line_arr = np.array([])
                continue
            if count >= n_params:
                break
            line_arr = np.hstack((line_arr, line.split()))
    return H.astype(np.float64)


def predict(H, params):
    return np.sort(np.abs(H[1, :] +
                          np.matmul(params.transpose(), H[2:, :])))

