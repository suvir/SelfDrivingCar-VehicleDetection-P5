import cv2
import numpy as np

from feature_extraction import extract_features


def increase_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def draw_sliding_windows(image, windows, color=(197, 27, 138), thick=3):
    for window in windows:
        cv2.rectangle(image, window[0], window[1], color, thick)
    return image


def apply_heatmap_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bounding_boxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (197, 27, 138), 3)
    return img


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list


def search_windows(img, windows, clf, scaler, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                   cell_per_block, hog_channel, spatial_switch, hist_switch, hog_switch):
    selected_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        extracted_features = extract_features([test_img], cspace=color_space, orient=orient, spatial_size=spatial_size,
                                              hist_bins=hist_bins, pix_per_cell=pix_per_cell,
                                              cell_per_block=cell_per_block,
                                              spatial_switch=spatial_switch, hist_switch=hist_switch, hog_switch=hog_switch,
                                              hog_channel=hog_channel)

        X_features = scaler.transform(np.array(extracted_features).reshape(1, -1))
        if clf.predict(X_features) is 1:
            selected_windows.append(window)
    return selected_windows
