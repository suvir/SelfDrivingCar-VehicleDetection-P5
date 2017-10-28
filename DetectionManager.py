import numpy as np
from scipy.ndimage import label

import vehicle_frame_detection
from FrameManager import FrameManager
from vehicle_frame_detection import slide_window, search_windows


class DetectionManager:
    """
    This class drives all the code. It includes:
    1. Sliding windows algorithm
    2. Feature generation
    3. Training SVM
    4. Post-processing filtering
    """

    def __init__(self, color_space, orient, pix_per_cell, cell_per_block,
                 hog_channel, spatial_size, hist_bins, spatial_feat,
                 hist_feat, hog_feat, y_start_stop, x_start_stop, xy_window,
                 xy_overlap, heat_threshold, scaler, classifier):
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.y_start_stop = y_start_stop
        self.x_start_stop = x_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap
        self.heat_threshold = heat_threshold
        self.scaler = scaler
        self.classifier = classifier

        self.frame_manager = FrameManager(25)

    def detect_vehicle(self, input_image):
        original = np.copy(input_image)
        original = original.astype(np.float32) / 255.0

        slided_windows = slide_window(original, x_start_stop=self.x_start_stop,
                                      y_start_stop=self.y_start_stop,
                                      xy_window=self.xy_window, xy_overlap=self.xy_overlap)

        selected_windows = search_windows(original, slided_windows, self.classifier, self.scaler,
                                          color_space=self.color_space, spatial_size=self.spatial_size,
                                          hist_bins=self.hist_bins, orient=self.orient,
                                          pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                          hog_channel=self.hog_channel, spatial_switch=self.spatial_feat,
                                          hist_switch=self.hist_feat, hog_switch=self.hog_feat)

        heat_map = np.zeros_like(original)
        heat_map = vehicle_frame_detection.increase_heat(heat_map, selected_windows)
        self.frame_manager.pushFrame(heat_map)

        all_frames = self.frame_manager.sum_frames()
        heat_map = vehicle_frame_detection.apply_heatmap_threshold(all_frames, self.heat_threshold)

        labels = label(heat_map)

        return vehicle_frame_detection.draw_labeled_bounding_boxes(input_image, labels)