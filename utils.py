import matplotlib.pyplot as plt
import os
import json
import numpy as np
import cv2
from scipy.interpolate import griddata
from matplotlib.widgets import Button
from scipy.ndimage import distance_transform_edt


def load_points(img1, img2, points_file):
    if os.path.exists(points_file):
        with open(points_file, "r") as file:
            data = json.load(file)
            points_img1 = data["points_img1"]
            points_img2 = data["points_img2"]
        print("Loaded points from file:")
        print("Points from Image 1:", points_img1)
        print("Points from Image 2:", points_img2)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        ax1.imshow(img1)
        ax1.set_title("Image 1 - Full Resolution")
        ax1.axis("off")
        ax2.imshow(img2)
        ax2.set_title("Image 2 - Full Resolution")
        ax2.axis("off")

        plt.subplots_adjust(wspace=0.1)

        points_img1 = []
        points_img2 = []

        def onclick(event):
            if event.button == 3:
                if event.inaxes == ax1 and len(points_img1) < 10:
                    points_img1.append((event.xdata, event.ydata))
                    ax1.plot(event.xdata, event.ydata, "ro")
                    fig.canvas.draw()
                elif event.inaxes == ax2 and len(points_img2) < 10:
                    points_img2.append((event.xdata, event.ydata))
                    ax2.plot(event.xdata, event.ydata, "ro")
                    fig.canvas.draw()

        def finalize(event):
            if len(points_img1) > 0 and len(points_img2) > 0:
                plt.close()
            else:
                print(
                    "Please select at least one point from each image before finalizing."
                )

        axbutton = plt.axes([0.45, 0.01, 0.1, 0.05])
        button = Button(axbutton, "Finalize")
        button.on_clicked(finalize)

        fig.canvas.mpl_connect("button_press_event", onclick)

        plt.show()

        data = {"points_img1": points_img1, "points_img2": points_img2}
        with open(points_file, "w") as file:
            json.dump(data, file)

        print("Points from Image 1:", points_img1)
        print("Points from Image 2:", points_img2)

    return points_img1, points_img2


def recale(img, points, factor):
    img = cv2.resize(
        img,
        (
            int(img.shape[1] * factor),
            int(img.shape[0] * factor),
        ),
    )

    points = [(x * factor, y * factor) for x, y in points]

    return img, points


def computeH(im1_pts, im2_pts):
    big_matrix = []
    little_matrix = []
    for i in range(im1_pts.shape[0]):
        x, y = im1_pts[i]
        u, v = im2_pts[i]

        big_matrix.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        big_matrix.append([0, 0, 0, x, y, 1, -x * v, -y * v])

        little_matrix.append(u)
        little_matrix.append(v)

    big_matrix = np.array(big_matrix)
    little_matrix = np.array(little_matrix).T

    homography_vector = np.linalg.lstsq(big_matrix, little_matrix, rcond=None)[0]

    homography_matrix = np.array(
        [
            [homography_vector[0], homography_vector[1], homography_vector[2]],
            [homography_vector[3], homography_vector[4], homography_vector[5]],
            [homography_vector[6], homography_vector[7], 1],
        ]
    )
    return homography_matrix


def warpImage(im1, H, crop_to_original=False):
    h1, w1, c1 = im1.shape

    corners = np.array(
        [[0, 0, 1], [w1 - 1, 0, 1], [w1 - 1, h1 - 1, 1], [0, h1 - 1, 1]],
        dtype=np.float32,
    ).T

    warped_corners = H @ corners
    warped_corners /= warped_corners[2, :]
    min_x, min_y = np.min(warped_corners[:2, :], axis=1)
    max_x, max_y = np.max(warped_corners[:2, :], axis=1)

    if crop_to_original:
        out_w = w1
        out_h = h1
        min_x, min_y = 0, 0
    else:
        out_w = int(np.ceil(max_x - min(min_x, 0)))
        out_h = int(np.ceil(max_y - min(min_y, 0)))

    print(f"Output image dimensions: width={out_w}, height={out_h}")

    stitched_image = np.zeros((out_h, out_w, c1), dtype=np.float32)
    alpha_channel = np.zeros((out_h, out_w), dtype=np.float32)

    translate_x = -min_x if not crop_to_original else 0
    translate_y = -min_y if not crop_to_original else 0
    print(f"Translation offsets: translate_x={translate_x}, translate_y={translate_y}")

    translated_corners = warped_corners.copy()
    translated_corners[0, :] += translate_x
    translated_corners[1, :] += translate_y

    cv2.fillConvexPoly(alpha_channel, translated_corners[:2, :].T.astype(np.int32), 1)

    warped_points = []
    pixel_values = []
    print("Starting forward mapping of pixels...")
    # TODO: Get rid of for loops
    for y in range(h1):
        for x in range(w1):
            original_coords = np.array([x, y, 1]).T

            warped_coords = H @ original_coords
            warped_coords /= warped_coords[2]
            warped_x, warped_y = warped_coords[:2]

            warped_x += translate_x
            warped_y += translate_y

            if 0 <= warped_x < out_w and 0 <= warped_y < out_h:
                warped_points.append([warped_y, warped_x])
                pixel_values.append(im1[y, x, :])

    print(f"Number of points to interpolate: {len(warped_points)}")

    # Interpolate the pixel values using griddata for smoother transformation
    warped_points = np.array(warped_points)
    pixel_values = np.array(pixel_values)
    grid_y, grid_x = np.mgrid[0:out_h, 0:out_w]
    print("Interpolating pixel values...")
    for i in range(c1):
        stitched_image[:, :, i] = griddata(
            warped_points,
            pixel_values[:, i],
            (grid_y, grid_x),
            method="linear",
            fill_value=0,
        )

    return (
        np.clip(stitched_image * 255, 0, 255).astype(np.uint8),
        alpha_channel,
        (-int(translate_x), -int(translate_y)),
    )


def combine_images(warped_image, img2, alpha_channel, translate):
    im2_pos = (
        0 if translate[0] >= 0 else -translate[0],
        0 if translate[1] >= 0 else -translate[1],
    )
    im1_pos = (
        0 if translate[0] <= 0 else translate[0],
        0 if translate[1] <= 0 else translate[1],
    )

    combined_w = max(warped_image.shape[1] + im1_pos[0], img2.shape[1] + im2_pos[0])
    combined_h = max(warped_image.shape[0] + im1_pos[1], img2.shape[0] + im2_pos[1])

    alpha_channel_large = np.zeros((combined_h, combined_w), dtype=np.float32)
    alpha_channel_large[
        im1_pos[1] : im1_pos[1] + warped_image.shape[0],
        im1_pos[0] : im1_pos[0] + warped_image.shape[1],
    ] = alpha_channel

    im2_mask = np.zeros((combined_h, combined_w), dtype=np.float32)
    im2_mask[
        im2_pos[1] : im2_pos[1] + img2.shape[0],
        im2_pos[0] : im2_pos[0] + img2.shape[1],
    ] = 1

    overlap = (alpha_channel_large > 0) & (im2_mask > 0)

    just_warped = np.ones((combined_h, combined_w), dtype=np.int32)
    just_warped[alpha_channel_large > 0] = 0
    just_warped[overlap] = 1

    just_im2 = np.ones((combined_h, combined_w), dtype=np.int32)
    just_im2[im2_mask > 0] = 0
    just_im2[overlap] = 1

    distance_map_warped = distance_transform_edt(just_warped)
    distance_map_im2 = distance_transform_edt(just_im2)

    distance_map_warped[overlap == 0] = 0
    distance_map_im2[overlap == 0] = 0

    epsilon = 1e-9
    denominator = distance_map_warped + distance_map_im2 + epsilon

    combined_distance_map = distance_map_im2 / denominator

    combined_distance_map[just_warped == 0] = 1
    combined_distance_map[just_im2 == 0] = 0

    combined_image = np.zeros((combined_h, combined_w, 3), dtype=np.float32)

    warped_image_large = np.zeros((combined_h, combined_w, 3), dtype=np.float32)
    warped_image_large[
        im1_pos[1] : im1_pos[1] + warped_image.shape[0],
        im1_pos[0] : im1_pos[0] + warped_image.shape[1],
    ] = (
        warped_image / 255.0
    )

    img2_large = np.zeros((combined_h, combined_w, 3), dtype=np.float32)
    img2_large[
        im2_pos[1] : im2_pos[1] + img2.shape[0],
        im2_pos[0] : im2_pos[0] + img2.shape[1],
    ] = img2

    combined_image[just_warped == 0] = warped_image_large[just_warped == 0]
    combined_image[just_im2 == 0] = img2_large[just_im2 == 0]

    expanded_combined_distance_map = np.expand_dims(combined_distance_map, axis=-1)

    combined_image[overlap] = (
        expanded_combined_distance_map * warped_image_large
        + (1 - expanded_combined_distance_map) * img2_large
    )[overlap]

    return combined_image
