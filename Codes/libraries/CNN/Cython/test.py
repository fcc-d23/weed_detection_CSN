from frame_creation import slide_window_creator
import cv2
import numpy as np
import time
img_prueba = cv2.imread('C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/IMG_20190818_124524.jpg').astype(np.uint8)
win_size = np.array([256, 256]).astype(np.int16)
new_win_shape = np.array([64, 64]).astype(np.int16)
sld_step = int(192)
data_type = 'CSN'

tic = time.time()

frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array, frame_array_tot = slide_window_creator(img_prueba, win_size, new_win_shape, sld_step, data_type)
toc = time.time()
print(toc-tic)

print(frame_array_in.shape)