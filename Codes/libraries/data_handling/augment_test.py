from img_processing import translate_pad, translate_augment, random_augment, rotation_augment, bright_augment, color_augment, zoomout_augment, region_tagging, noise_augment, multiclass_tagging
import cv2
import numpy as np

img_rgb = cv2.imread('C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce/4.jpg')
pad_factor = 0.25
from PyQt5.QtWidgets import QApplication, QLabel, QDialog, QPushButton, QDialogButtonBox, QLineEdit, QFileDialog, QAction, QSpinBox, QDoubleSpinBox, QCheckBox, QVBoxLayout,\
    QFrame, QWidget

app =QApplication([])
origin_folder = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol'
destinies_folder = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl'
#multiclass_tagging(origin_folder, destinies_folder, frame_size = (256, 256, 3), new_shape = (256, 256, 3), ovrwrite = True)
#region_tagging(origin_folder, destinies_folder, frame_size = (1800, 1800, 3), new_shape = (256, 256, 3), overwrite = False, black_fill= False)
#pad_img = np.zeros(img_rgb.shape, dtype=np.uint8)
'''
translate_augment('C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/pasto',\
    'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/pasto',pad_factor, pad_methods = 1)
translate_augment('C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/trebol',\
    'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/trebol',pad_factor, pad_methods = 2)
translate_augment('C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce',\
    'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce',pad_factor, pad_methods = 7)

frame_size = [1440, 1440]
sld_step = [360, 360]
pad = [0,0]
img = np.ones((3280, 4280, 3), dtype = np.uint8)
img_size = img.shape

pad[0] = frame_size[0] - img_size[0] % sld_step[0] if img_size[0] % sld_step[0] != 0 else 0
pad[1] = frame_size[1] - img_size[1] % sld_step[1] if img_size[1] % sld_step[1] != 0 else 0
print(pad) 
padded_img = (np.pad(img, ((0,pad[0]), (0,pad[1]), (0,0)), 'constant'))*255
cv2.imshow('', cv2.resize(padded_img, (800,800)))
cv2.waitKey(0)
print(padded_img.shape)
'''
img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl/pasto'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl_aug/pasto'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'color', 'zoom', 'noise' ], bright_change = 10, color_variation= 5, keep_orig = False, overwrite = True)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl/trebol'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl_aug/trebol'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'color', 'zoom', 'noise' ], bright_change = 10, color_variation= 5, keep_orig = True, overwrite = True)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl/miniwl'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl_aug/miniwl'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'color', 'zoom', 'noise' ], bright_change = 10, color_variation= 5, keep_orig = False, overwrite = True, double_aug = True)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl/miniwl'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/extract pasto_trebol/con_wl_aug/miniwl'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'color', 'zoom', 'noise' ], bright_change = 10, color_variation= 5, keep_orig = True, overwrite = False, double_aug = False)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'pad', 'color', 'zoom' ], bright_change = 10, color_variation= 5, keep_orig = True, overwrite = True, double_aug = 'Yes')

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip' ], bright_change = 20, keep_orig = True, overwrite = False, double_aug = 'No')

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce'
rotation_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, repeats= 10)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce'
bright_augment(img_rgb_source, img_aug_dest, bright_change = 20, keep_orig = False, overwrite = False, repeats=10)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce'
color_augment(img_rgb_source, img_aug_dest, color_variation = 15, keep_orig = False, overwrite = False, repeats=10)


img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo - respaldo/wild lettuce'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass_felo_aug/wild lettuce'
#random_augment(img_rgb_source, img_aug_dest, augment_list = ['pad' ], bright_change = 20, keep_orig = False, overwrite = False)

translate_augment(img_rgb_source, img_aug_dest + 'translate', pad_factor, pad_methods = 7)

zoomout_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, zoom_factor = 0.75, rl_pad = 0.75, tb_pad = 0.5, repeats = 5, color_rand = False, pad_rand = True)



#####################AUGMENT PARA REGIONES

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/pasto'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/pasto'
random_augment(img_rgb_source, img_aug_dest, augment_list = [ 'bright', 'color', 'noise', 'zoom' ], bright_change = 20, color_variation= 5, keep_orig = False, overwrite = True, double_aug='rand')
#random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip' ], bright_change = 20, keep_orig = True, overwrite = False, double_aug = 'No')
#rotation_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, repeats= 1)
#bright_augment(img_rgb_source, img_aug_dest, bright_change = 20, keep_orig = False, overwrite = False, repeats=1)
#color_augment(img_rgb_source, img_aug_dest, color_variation = 15, keep_orig = False, overwrite = False, repeats=1)
#zoomout_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, zoom_factor = 0.75, rl_pad = 0.75, tb_pad = 0.5, repeats = 2, color_rand = False, pad_rand = True)
#noise_augment(img_rgb_source, img_aug_dest, noise_var = 35, keep_orig = False, overwrite = False, repeats=1)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/trebol'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/trebol'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['bright', 'color',  'noise', 'zoom' ], bright_change = 20, color_variation= 5, keep_orig = False, overwrite = True, double_aug='rand')
#random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip' ], bright_change = 20, keep_orig = True, overwrite = False, double_aug = 'No')
#rotation_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, repeats= 1)
#bright_augment(img_rgb_source, img_aug_dest, bright_change = 20, keep_orig = False, overwrite = False, repeats=1)
#color_augment(img_rgb_source, img_aug_dest, color_variation = 15, keep_orig = False, overwrite = False, repeats=1)
#zoomout_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, zoom_factor = 0.75, rl_pad = 0.75, tb_pad = 0.5, repeats = 2, color_rand = False, pad_rand = True)
#noise_augment(img_rgb_source, img_aug_dest, noise_var = 35, keep_orig = False, overwrite = False, repeats=1)
'''
img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 1'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 1'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'pad', 'color', 'zoom' ], bright_change = 20, color_variation= 15, keep_orig = True, overwrite = True, double_aug = 'Yes')

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 1'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 1'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip' ], bright_change = 20, keep_orig = True, overwrite = False, double_aug = 'No')

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 1'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 1'
rotation_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, repeats= 5)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 1'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 1'
bright_augment(img_rgb_source, img_aug_dest, bright_change = 20, keep_orig = False, overwrite = False, repeats=5)

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 1'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 1'
color_augment(img_rgb_source, img_aug_dest, color_variation = 5, keep_orig = False, overwrite = False, repeats=5)


img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 1'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 1'
translate_augment(img_rgb_source, img_aug_dest + 'translate', pad_factor, pad_methods = 4)
zoomout_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, zoom_factor = 0.75, rl_pad = 0.75, tb_pad = 0.5, repeats = 5, color_rand = False, pad_rand = True)
'''

#WL 2

img_rgb_source = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region/wild lettuce 2'
img_aug_dest = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/region_aug/wild lettuce 2'
random_augment(img_rgb_source, img_aug_dest, augment_list = ['bright',  'color', 'noise' ], bright_change = 20, color_variation= 15, keep_orig = True, overwrite = True, double_aug = 'Yes')
#random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip' ], bright_change = 20, keep_orig = True, overwrite = False, double_aug = 'No')
#rotation_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, repeats= 10)
bright_augment(img_rgb_source, img_aug_dest, bright_change = 20, keep_orig = False, overwrite = False, repeats=1)
color_augment(img_rgb_source, img_aug_dest, color_variation = 5, keep_orig = False, overwrite = False, repeats=1)
#translate_augment(img_rgb_source, img_aug_dest + 'translate', pad_factor, pad_methods = 7)
zoomout_augment(img_rgb_source, img_aug_dest, keep_orig = False, overwrite = False, zoom_factor = 0.75, rl_pad = 0.75, tb_pad = 0.5, repeats = 1, color_rand = False, pad_rand = True)
noise_augment(img_rgb_source, img_aug_dest, noise_var = 35, keep_orig = False, overwrite = False, repeats=1)