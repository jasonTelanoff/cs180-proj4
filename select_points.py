import matplotlib.pyplot as plt
import utils

img1 = plt.imread("data/chinese_food.jpg")
img2 = plt.imread("data/chinese_food.jpg")

points_img1, points_img2 = utils.load_points(img1, img2, "data/chinese_points.json")
