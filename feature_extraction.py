import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def find_best_hyperparams(X, Y):
    chosen_accuracy = 0.0
    C_candidates = [0.07, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.6]
    #C_candidates = [0.07]
    penalty_candidates = ['l2']
    loss_fn_candidates = ['hinge', 'squared_hinge']
    for c in C_candidates:
        for p in penalty_candidates:
            for l in loss_fn_candidates:
                # Split into training and test set
                train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

                # Standardize
                scaler = StandardScaler().fit(train_X)
                train_X = scaler.transform(train_X)
                test_X = scaler.transform(test_X)

                # Train SVM
                svc = LinearSVC(C=c, penalty=p, loss=l).fit(train_X, train_Y)
                iter_accuracy = svc.score(test_X, test_Y)
                print('Accuracy: {:.5f}. C: {}, penalty function: {}, loss: {}'.format(iter_accuracy, c, p, l))

                if chosen_accuracy < iter_accuracy:
                    chosen_accuracy = iter_accuracy
                    chosen_c = c
                    chosen_loss = l
                    chosen_penalty = p

    print('Top accuracy: {:.4f}'.format(chosen_accuracy))
    print('Chosen parameters ... C: {}, penalty: {}, loss: {}'.format(chosen_c, chosen_penalty, chosen_loss))
    return chosen_c, chosen_loss, chosen_penalty


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), \
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
                                  visualise=True, feature_vector=False)
        return features.ravel(), hog_image
    else:

        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), \
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
                       visualise=False, feature_vector=feature_vec)
        return features.ravel()


def bin_spatial(image, size=(32, 32)):
    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    color_1_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_2_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_3_hist = np.histogram(img[:, :, 0], bins=nbins)
    return np.concatenate((color_1_hist[0], color_2_hist[0], color_3_hist[0]))


def extract_features(images, cspace='RGB', orient=9, spatial_size=(32, 32), hist_bins=32,
                     pix_per_cell=8, cell_per_block=2,
                     spatial_switch=True, hist_switch=True, hog_switch=True, hog_channel=0):
    features = []
    for image in images:
        file_features = []
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_switch:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_switch:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
            file_features.append(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))
    return np.array(features)


def image_features(img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True, hist_switch=True, hog_switch=True, vis=True):
    image_features = []
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        image_features.append(spatial_features)
    if hist_switch:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        image_features.append(hist_features)
    if hog_switch:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            if vis:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                           pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            image_features.append(hog_features)
    return np.concatenate(image_features)
    # if hog_switch and vis:
    #     return np.concatenate(image_features), hog_image
    # else:
    #     return np.concatenate(image_features)
