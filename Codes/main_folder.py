#Example file to run the weed detection method
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2023

##
#External library importing
#from keras.backend.cntk_backend import learning_phase
import numpy as np, platform
from matplotlib import pyplot as plt
import cv2
import os, sys

from PyQt5.QtWidgets import QApplication, QLabel, QDialog, QPushButton, QDialogButtonBox, QLineEdit, QFileDialog, QAction, QSpinBox, QDoubleSpinBox, QCheckBox, QVBoxLayout,\
    QFrame, QWidget
from sys import getsizeof
##KERAS##
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Flatten, Lambda


##
##Importing of developed libraries
#Cambio de cd
file_path = os.path.abspath(__file__)
file_path = file_path[::-1]
#Para ver si es windows o linux 
oper_sys = platform.system()
if oper_sys == "Linux":
    file_path = file_path.split("/", 1)[1][::-1]
elif oper_sys == "Windows":
    file_path = file_path.split("\\", 1)[1][::-1]
os.chdir(file_path)
cwd = os.getcwd()

code_path = os.path.dirname(os.path.realpath(__file__)) + '/libraries'
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(code_path)

from data_handling.img_processing import  slide_window_creator
from CNN.cnn_database import  multiclassparser, multiclass_preprocessing, multiclass_set_creator
from CNN.cnn_configuration import CNN_params, CNN_train_params
from CNN.csn_functions import CSN_centroid_finder
from CNN.cnn_training_testing import CNN_train_iterator, best_model_finder, conf_matrix, crossval_stat_calc
from CNN.seg_functions import CSN_region_seg, pipeline, NMS_bb, folder_pipeline

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Main pipeline begins...
#Loading of trained models
CNN_folder = cwd + '/CNN_data/CNN/CNN_models'
CSN_folder = cwd + '/CNN_data/CSN/CSN_models'
#Crossval data is loaded to guarantee the best performance in crossvalidation
crossval_CNN_dir =  CNN_folder + '/crossval_stats'
CNN_model = crossval_stat_calc(CNN_folder, crossval_CNN_dir, CNN_class_name = 'CNN')
crossval_CSN_dir = CSN_folder + '/crossval_stats'
CSN_model = crossval_stat_calc(CSN_folder, crossval_CSN_dir, CNN_class_name = 'CSN')

#The centroid is obtained for CSN detection
sources_list, class_list = multiclassparser(cwd + '/../Database/Frames/Regions')
sources_list = [s + "/train" for s in sources_list]
X, Y, class_dict_reg = multiclass_preprocessing(sources_list, class_list, image_size = (128,128,3))
print(X.shape)
CNN_train_test_ratio = 1
X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict_reg = multiclass_set_creator(X, Y, class_dict_reg, CSN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    
feat_mean_reg, class_dict_reg, semantic_net_sld = CSN_centroid_finder(X_train, Y_train, CSN_model, class_dict_reg, save_data = True, save_dir = cwd + '/CNN_data/CSN' )
class_dict_sld = class_dict_reg
class_dict_sld['class_name_list'] = ['Grass', 'Structured', 'Unstructured']
#The actual pipeline is used on the folder data
print('####PIPELINE CSN######')
multires_win_size = [n/16 for n in range(6, 11)]
img_folder = cwd + '/../Database/Full Images'
folder_pipeline(img_folder, CSN_model, CNN_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = feat_mean_reg, feat_mean_second= feat_mean_reg, overlap_factor_first = .85,\
    overlap_factor_second = 0.75, multi_res_win_size = multires_win_size, multi_res_name = 'Structured', IOU_multires = .25, IOU_hm = .75, heat_map_class = 'Unstructured', heat_map_display = True, bg_class = 'Grass',\
        class_mask_display = True, method = 'region_sld', model_type = ['CSN', 'CNN'], savedir = img_folder + '/seg_imgs', pred_batch = [64, 128], r_neighbours = 0, imsave = True,\
            hm_thresh = 6e-1, region_wl_thresh = 7e-1, region_hm_thresh = 2e-1)
