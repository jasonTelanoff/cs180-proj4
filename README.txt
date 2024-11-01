Hello there.

To manually align images:
Put the images you want to align where you want them, and then use select_points.py to manually select your points. If you are recitifying then select where you want the shape to end up in the second image (it will be cropped to the size of the original image). If you want to be more precise you can change the json file. Next, create a cell in main to run one do_everything for a panorama and rectify to rectify the image.

To automatically align images:
Put the images somewhere in the code folder, and then call create_panorama in main.ipynb.
