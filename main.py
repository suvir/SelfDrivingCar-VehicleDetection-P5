import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import DetectionManager
import data_exploration
import feature_extraction

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_switch = True  # Spatial features on or off
hist_switch = True  # Histogram features on or off
hog_switch = True  # HOG features on or off
x_start_stop = [None, None]
y_start_stop = [400, 600]
xy_window = (95, 85)
xy_overlap = (0.75, 0.75)


def get_images():
    vehicle_files_dir = './data/vehicles/'
    non_vehicle_files_dir = './data/non-vehicles/'

    vehicle_files = data_exploration.extract_files(vehicle_files_dir)
    non_vehicle_files = data_exploration.extract_files(non_vehicle_files_dir)

    vehicle_images = [mpimg.imread(file) for file in vehicle_files]
    non_vehicle_images = [mpimg.imread(file) for file in non_vehicle_files]

    print('Number of vehicle files: {}'.format(len(vehicle_files)))
    print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))

    return vehicle_images, non_vehicle_images


def extract_features(vehicle_images, non_vehicle_images):
    vehicle_features = feature_extraction.extract_features(vehicle_images, color_space, orient, spatial_size, hist_bins,
                                                           pix_per_cell, cell_per_block, spatial_switch, hist_switch,
                                                           hog_switch,
                                                           hog_channel)

    non_vehicle_features = feature_extraction.extract_features(non_vehicle_images, color_space, orient, spatial_size,
                                                               hist_bins, pix_per_cell, cell_per_block, spatial_switch,
                                                               hist_switch, hog_switch, hog_channel)
    print(
        'Shape of vehicle and non-vehicle features: {}, {}'.format(vehicle_features.shape, non_vehicle_features.shape))
    return vehicle_features, non_vehicle_features


def standardize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


vehicle_images, non_vehicle_images = get_images()
vehicle_features, non_vehicle_features = extract_features(vehicle_images, non_vehicle_images)

X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
Y = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))))

chosen_c, chosen_loss, chosen_penalty = feature_extraction.find_best_hyperparams(X, Y)

X, scaler = standardize(X)
svc = LinearSVC(C=chosen_c, penalty=chosen_penalty, loss=chosen_loss).fit(X, Y)

vehicle_detector = DetectionManager.DetectionManager(color_space=color_space, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=hog_channel,
                                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                                     spatial_feat=spatial_switch,
                                                     hist_feat=hist_switch, hog_feat=hog_switch,
                                                     y_start_stop=y_start_stop,
                                                     x_start_stop=x_start_stop, xy_window=xy_window,
                                                     xy_overlap=xy_overlap,
                                                     heat_threshold=15, scaler=scaler, classifier=svc)

output_file = './output_project_video.mp4'
input_file = './project_video.mp4'

clip = VideoFileClip(input_file)
out_clip = clip.fl_image(vehicle_detector.detect_vehicle)
out_clip.write_videofile(output_file, audio=False)
