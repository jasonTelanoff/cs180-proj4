import matplotlib.pyplot as plt
import os
import json
import numpy as np
import cv2
from scipy.interpolate import griddata
from matplotlib.widgets import Button

import utils

img1 = plt.imread("chinese_food.jpg")
img2 = plt.imread("chinese_food.jpg")

points_img1, points_img2 = utils.load_points(img1, img2, "chinese_points.json")

# # Load the images
# image1_path = "soda-1.jpg"  # Update the path to your first image
# image2_path = "soda-2.jpg"  # Update the path to your second image
# points_file = "selected_points.json"
# save_file = "soda.jpg"

# # Set the scale down factor to reduce image resolution for processing
# scale_factor = 1

# # Load full resolution images for point selection


# # Convert points to numpy arrays
# points_img1_np = np.array(points_img1, dtype=np.float32)
# points_img2_np = np.array(points_img2, dtype=np.float32)


# # Function to compute homography matrix


# # Compute the homography matrix
# homography_matrix = computeH(points_img1_np, points_img2_np)
# print("Computed Homography Matrix:")
# print(homography_matrix)


# # Warp the first image using the computed homography matrix
# warped_image, alpha_channel, translate = warpImage(
#     img1, homography_matrix, crop_to_original=False
# )

# translate = (-int(translate[0]), -int(translate[1]))

# print(translate)


# plt.imshow(combined_image)
# plt.title("Combined Image")
# plt.axis("off")
# plt.show()

# # save the combined image
# cv2.imwrite(save_file, combined_image * 255)


# # plt.figure(figsize=(10, 8))
# # plt.imshow(alpha_channel)
# # plt.title("Warped Image 1")
# # plt.axis("off")
# # plt.show()

# # plt.figure(figsize=(10, 8))
# # plt.imshow(warped_image)
# # plt.title("Warped Image 1")
# # plt.axis("off")
# # plt.show()
