import os

import numpy as np
from matplotlib import image as img, gridspec as gridspec, pyplot as plt


def extract_files(parent, extension='.png'):
    file_container = []
    for root, dirs, files in os.walk(parent):
        for file in files:
            if file.endswith(extension):
                file_container.append(os.path.join(root, file))
    return file_container


def display_random_images(image_files, num_images=20, images_per_row=5, main_title=None):
    random_files = np.random.choice(image_files, num_images)
    images = []
    for random_file in random_files:
        images.append(img.imread(random_file))

    grid_space = gridspec.GridSpec(num_images // images_per_row + 1, images_per_row)
    grid_space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(images_per_row, num_images // images_per_row + 1))

    for index in range(0, num_images):
        axis_1 = plt.subplot(grid_space[index])
        axis_1.axis('off')
        axis_1.imshow(images[index])

    if main_title is not None:
        plt.suptitle(main_title)
    plt.show()


def visualize_hog_features(hog_features, images, cmap=None, title=None):
    num_images = len(images)
    space = gridspec.GridSpec(num_images, 2)
    space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(4, 2 * (num_images // 2 + 1)))

    for index in range(0, num_images*2):
        if index % 2 == 0:
            axis_1 = plt.subplot(space[index])
            axis_1.axis('off')
            axis_1.imshow(images[index // 2], cmap=cmap)
        else:
            axis_2 = plt.subplot(space[index])
            axis_2.axis('off')
            axis_2.imshow(hog_features[index // 2], cmap=cmap)

    if title is not None:
        plt.suptitle(title)
    plt.show()