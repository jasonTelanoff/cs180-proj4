import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, peak_local_max
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import os
import json
from matplotlib.widgets import Button


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


def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method="eps", sigma=1)
    coords = peak_local_max(h, min_distance=1, threshold_abs=10)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )
    coords = coords[mask]
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Data dimension does not match dimension of centers"

    return (
        (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c**2).T, axis=0)
        - 2 * np.inner(x, c)
    )


def anms(h, coords, max_points=500):
    n = len(coords)
    c_robust = 0.9

    # sort coords by h value first
    coords = coords[np.argsort(h[coords[:, 0], coords[:, 1]])]

    D = dist2(coords, coords)

    h_values = np.array([h[int(coord[0]), int(coord[1])] for coord in coords])

    r = np.full(n, np.inf)

    for i in tqdm(range(n), desc="ANMS"):
        valid_mask = h_values[i] < c_robust * h_values
        valid_mask[i] = False

        if np.any(valid_mask):
            r[i] = np.min(D[i, valid_mask])

    max_points = min(max_points, n)
    selected_indices = np.argsort(r)[-max_points:]

    return coords[selected_indices]


def get_gaussian_stack(image, N, sigma):
    stack = [image]
    for _ in range(N):
        stack.append(gaussian_filter(stack[-1], sigma))
    return stack


def get_features(coords, stack):
    features = []

    for i in range(len(coords)):
        y, x = coords[i]
        feature = np.zeros((8, 8))

        for j in range(8):
            for k in range(8):
                _j = 5 * (j - 4) // 4
                _k = 5 * (k - 4) // 4

                feature[j, k] = stack[2][y + _j, x + _k]

        feature = feature.flatten()
        feature = (feature - np.mean(feature)) / np.std(feature)
        features.append(feature)

    return np.array(features)


def compare_features(features1, features2):
    D = dist2(features1, features2)
    N1, _ = D.shape

    sorted_indices = np.argsort(D, axis=1)
    nearest_indices = sorted_indices[:, 0]
    second_nearest_indices = sorted_indices[:, 1]

    nearest_distances = D[np.arange(N1), nearest_indices]
    second_nearest_distances = D[np.arange(N1), second_nearest_indices]

    ratios = nearest_distances / second_nearest_distances

    ratio_threshold = 0.8
    mask = ratios < ratio_threshold

    comparison_map = np.zeros_like(D, dtype=bool)
    comparison_map[np.arange(N1), nearest_indices] = mask

    matched_indices = np.argwhere(comparison_map)

    return comparison_map, matched_indices


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


def RANSAC(pairs, _map, coords1, coords2, n=1000, t=1):
    best_count = 0
    best_H = None
    best_coords = None
    best_distance = float("inf")

    print(coords1.shape)

    for _ in tqdm(range(n), desc="RANSAC"):
        while True:
            special_pairs = pairs[np.random.choice(pairs.shape[0], 4, replace=False)]
            if len(np.unique(special_pairs.flatten())) == 8:
                break

        special_coords1 = coords1[special_pairs[:, 0]]
        special_coords2 = coords2[special_pairs[:, 1]]

        H = computeH(special_coords1, special_coords2)

        transformed_coords1 = H @ np.vstack([coords1.T, np.ones(coords1.shape[0])])

        transformed_coords1 /= transformed_coords1[2, :]
        transformed_coords1 = transformed_coords1[:2].T

        D = dist2(transformed_coords1, coords2)
        mask = D < t

        result = np.logical_and(mask, _map)
        count = np.sum(result)

        dist = np.sum(D[result])

        if count > best_count or (count == best_count and dist < best_distance):
            best_count = count
            best_H = H
            best_coords = transformed_coords1
            best_distance = dist

    return best_H, best_coords, best_count, best_distance


def warpImage(im, H, crop_to_original=False):
    h, w, _ = im.shape

    corners = np.array(
        [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]],
        dtype=np.float32,
    ).T

    warped_corners = H @ corners
    warped_corners /= warped_corners[2, :]

    min_x, min_y = np.min(warped_corners[:2, :], axis=1)
    max_x, max_y = np.max(warped_corners[:2, :], axis=1)

    if crop_to_original:
        out_w = w
        out_h = h
        min_x, min_y = 0, 0
    else:
        out_w = int(np.ceil(max_x - min(min_x, 0)))
        out_h = int(np.ceil(max_y - min(min_y, 0)))

    print(f"Output image dimensions: width={out_w}, height={out_h}")

    alpha_channel = np.zeros((out_h, out_w), dtype=np.float32)

    translate_x = -min_x if not crop_to_original else 0
    translate_y = -min_y if not crop_to_original else 0
    print(f"Translation offsets: translate_x={translate_x}, translate_y={translate_y}")

    translated_corners = warped_corners.copy()
    translated_corners[0, :] += translate_x
    translated_corners[1, :] += translate_y

    cv2.fillConvexPoly(alpha_channel, translated_corners[:2, :].T.astype(np.int32), 1)

    H_inv = np.linalg.inv(H)

    print("Starting inverse mapping of pixels...")

    x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))
    warped_coords = np.stack(
        [x - translate_x, y - translate_y, np.ones_like(x)], axis=-1
    )

    original_coords = (H_inv @ warped_coords.reshape(-1, 3).T).T
    original_coords /= original_coords[:, [2]]

    original_x = original_coords[:, 0].reshape(out_h, out_w).astype(np.float32)
    original_y = original_coords[:, 1].reshape(out_h, out_w).astype(np.float32)

    original_x = np.clip(original_x, 0, w - 1)
    original_y = np.clip(original_y, 0, h - 1)

    remapped = cv2.remap(
        im,
        original_x,
        original_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return remapped.astype(int), alpha_channel, (-int(translate_x), -int(translate_y))


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
    ] = warped_image

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


def visualize_feature_pairs(image1, image2, coords1, coords2, pairs):
    combined_image = np.hstack((image1, image2))

    offset = image1.shape[1]
    coords2_offset = coords2 + [0, offset]

    plt.figure(figsize=(10, 5))
    plt.imshow(combined_image, cmap="gray")

    plt.scatter(
        coords1[:, 1], coords1[:, 0], color="blue", s=10, label="Image 1 Features"
    )
    plt.scatter(
        coords2_offset[:, 1],
        coords2_offset[:, 0],
        color="blue",
        s=10,
        label="Image 2 Features",
    )

    for idx1, idx2 in pairs:
        point1 = coords1[idx1]
        point2 = coords2_offset[idx2]
        plt.plot(
            [point1[1], point2[1]], [point1[0], point2[0]], "yellow", linewidth=0.5
        )

    plt.legend()
    plt.axis("off")
    plt.title("Feature Matches Between Images")
    plt.show()
