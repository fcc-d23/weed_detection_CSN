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
from CNN.seg_functions import CSN_region_seg, pipeline, NMS_bb

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Main pipeline begins...
#Loading of trained models
CNN_folder = cwd + '/CNN_data/CNN/CNN_models'
CSN_folder = cwd + '/Pasto/extract/data/CSN'
#Crossval data is loaded to guarantee the best performance in crossvalidation
crossval_CNN_dir =  CNN_folder + '/crossval_stats'
CNN_sld_model = crossval_stat_calc(CNN_folder, crossval_CNN_dir, CNN_class_name = 'CNN')

raise ValueError("todo bien hasta l 56")
##DATA TAGGING##
app =QApplication([])
cwd = os.getcwd()
im_path = cwd + '/Pasto/extract'

#Creación de la base de datos a partir de imágenes completas
destinies_folder_felo = cwd + '/Pasto/extract/multiclass_felo'
destinies_folder_ral = cwd + '/Pasto/extract/multiclass_ral'
destinies_folder_ej = cwd + '/Pasto/extract/multiclass_ej'
#multiclass_tagging(im_path, destinies_folder_felo, frame_size = (256, 256, 3), new_shape = (64, 64, 3), ovrwrite=True)
#multiclass_tagging(im_path, destinies_folder_ral, frame_size = (256, 256, 3), new_shape = (128, 128, 3), ovrwrite=True)
#multiclass_tagging(im_path, destinies_folder_ej, frame_size = (256, 256, 3), new_shape = (64, 64, 3), ovrwrite=True)

#Conversión de imágenes a matriz y creación de los conjuntos de entrenamiento, validación y prueba
img_shape = (64, 64, 3)
CNN_folder = cwd + '/Pasto/extract/data/CNN_felo'
CSN_folder = cwd + '/Pasto/extract/data/CSN_felo'
TL_folder = cwd + '/Pasto/extract/data/TL_felo'

#sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/extract pasto_trebol/con_wl')


#X, Y, class_dict = multiclass_preprocessing(sources_list, class_list, image_size = img_shape)
class_n = 3#len(class_dict['class_n_list'])
#print(getsizeof(X))

#CNN_train_test_ratio = 0.75
#X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, CNN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    

#CSN_train_test_ratio = 0.75
#CSN_X_train, CSN_X_val, CSN_X_test, CSN_Y_train, CSN_Y_val, CSN_Y_test, class_dict = \
#    multiclass_set_creator(X, Y, class_dict, CSN_folder, train_test_rate = CSN_train_test_ratio, test_val_rate = 0.5, CNN_class = 'CSN')


#TL_X_train, TL_X_val, TL_X_test, TL_Y_train, TL_Y_val, TL_Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, TL_folder, train_test_rate = 0.75, test_val_rate = 0.5, CNN_class='CNN', TL = True, TL_model_name = 'VGG')    
#Creación de una red neuronal de tipo CNN con arquitectura encoder-decoder
CNN_params_obj = CNN_params('CNN', class_n = class_n, learning_rate=0.001)
CNN_params_obj.conv2dlayer_shape_adder([50, 50, 4])
CNN_params_obj.conv2dlayer_shape_adder([20, 20, 8])
CNN_params_obj.conv2dlayer_shape_adder([8, 8, 10])
CNN_params_obj.conv2dlayer_shape_adder([3, 3, 15])

#CNN_params_obj.conv2dlayer_shape_adder([8, 8, 10])
##CNN_params_obj.conv2dlayer_shape_adder([20, 20, 8])
#CNN_params_obj.conv2dlayer_shape_adder([50, 50, 4])

#Creación del objeto que contiene las opciones para la sesión de entrenamiento. En orden: batch size, epochs totales, verbose (0 ó 1), shuffle (True o False) y epochs mínimas de entrenamiento
CNN_train_params_obj = CNN_train_params(32, 450, 0, True, 300)
#Iteración sobre un rango de capas de clasificación, incluyendo la cantidad de capas como entrada. En orden, las entradas son:

#CNN_params_obj: Objeto con la clase de red (CNN, TL o CSN) y la arquitectura de las capas convolucionales
#CNN__train_params_obj: Objeto con las opciones de entrenamiento
#img_shape: La forma de las imágenes de entrada, dada por el preprocesamiento anterior
#CNN_folder: La carpeta donde se guardarán las redes, coincide con la carpeta donde se crearon los conjuntos
#destiny: El nombre de la sub-carpeta donde se guardará cada red, para esta parte es irrelevante
#model_name: Por ahora no importa    
#class_layers_range: Cantidad de capas de clasificación previas a la capa de etiquetas
#class_neuron_range: Rango de neuronas de clasificación sobre las que se iterará. Si la lista tiene 2 números se revisan uno a uno, si son 3 se revisan la cantidad de neuronas del tercer valor dentro
#del rango especificado
#last_check: Revisa el último estado de iteración (si se había partido antes y quedó a la mitad)
#ntimes: Número de veces que
# se reinicializa la red, para disminuir el sesgo por mala inicialización (VER COMO SE PODRÍA EVITAR ESTO, COSTOSO EN TIEMPO)
#tf_seed: Semilla para los procesos randomizados de TF y KERAS

img_shape = (64, 64, 3)
#Descomentar para entrenar la red
###############################################################################################################################
#CNN_train_iterator(CNN_params_obj, CNN_train_params_obj,  img_shape, CNN_folder, destiny = '0.75_aug_CNN4conv', model_name = 'generico',\
#class_layers_range = 1, class_neuron_range = [128, 256, 9], last_check = True,  ntimes = 12, tf_seed = 1,acc_vec=True)  
img = cv2.imread(cwd + '/Pasto/IMG_20190818_124619.jpg')
#box_model, session = best_model_finder(CNN_folder + '/0.75_region_aug', CNN_class_name = 'CNN')
#box_seg(img, box_model, class_dict, win_size = (1800, 1800), overlap_factor = 0.75, pred_batch_size = 1, multiresolution = [1800, 1800], multires_name = 'wild lettuce')
'''
reg_model, session = best_model_finder(CNN_folder + '/0.75_region_aug', CNN_class_name = 'CNN')
sld_model, session = best_model_finder(CNN_folder + '/0.75_aug', CNN_class_name = 'CNN')
#sld_model = 'a'
#Método de multi-resolución
img = cv2.imread(cwd + '/Pasto/IMG_20190818_124525.jpg')
import time
tic = time.time()
#pipeline(img, reg_model, sld_model, class_dict, frame_size = (256, 256, 3), overlap_factor_first = 0.25, overlap_factor_second = 0.75, multi_res_win_size = [1800, 1800], multi_res_name = 'wild lettuce', \
#   heat_map_display = True, bg_class = 'pasto', class_mask_display = True, method = 'multires_sld', model_type = ['CNN', 'CNN'] )
tictoc_multires_sld = time.time()-tic

img = cv2.imread(cwd + '/Pasto/IMG_20190818_124624.jpg')
#pipeline(img, reg_model, sld_model, class_dict, frame_size = (256, 256, 3), overlap_factor_first = 0.75, multi_res_win_size = [1800, 1800], multi_res_name = 'wild lettuce', \
#   heat_map_display = True, class_mask_display = True, method = 'multires_sld', model_type = ['CNN', 'CNN'] )
#Método de regiones
img = cv2.imread(cwd + '/Pasto/IMG_20190818_124629.jpg')
tic = time.time()
pipeline(img, reg_model, sld_model, class_dict, frame_size = (256, 256, 3), overlap_factor_first = 0.25, overlap_factor_second = 0.75, multi_res_win_size = [1800, 1800], multi_res_name = 'wild lettuce', \
    heat_map_display = True, class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'] )
tictoc_region_sld = time.time()-tic

img = cv2.imread(cwd + '/Pasto/IMG_20190818_124627.jpg')
#pipeline(img, reg_model, sld_model, class_dict, frame_size = (256, 256, 3), overlap_factor = 0.85, multi_res_win_size = [1800, 1800], multi_res_name = 'wild lettuce', \
#   heat_map_display = True, class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'] )

print('tiempo de cálculo del método multires_sld: ' + str(tictoc_multires_sld) + 's' )
print('tiempo de cálculo del método region_sld: ' + str(tictoc_region_sld) + 's' )
'''
#Ver como se desarrollo el entrenamiento de un modelo específico
train_vector_folder = CSN_folder + '/vector_folder'
plot_destiny = CSN_folder + '/vector_folder/accuracy plots/(16,)'
#train_plot(train_vector_folder, plot_destiny)
#######################################################################

#PARA PROBAR PIPELINE CON CSN

CNN_folder = cwd + '/Pasto/extract/data/CNN_ral'
CSN_folder = cwd + '/Pasto/extract/data/CSN_ral'

CSN_reg_model, session = best_model_finder(CSN_folder + '/0.75_region_aug_128_CNN4convgrande_mkiii', CNN_class_name = 'CSN')
CSN_sld_model, session = best_model_finder(CSN_folder + '/0.75_extractpastotrebol_conwl_aug_noise_64_CNN4conv_mini_deep', CNN_class_name = 'CSN')
'''
#sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/extract pasto_trebol/con_wl_aug')
sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/region_aug')
X, Y, class_dict = multiclass_preprocessing(sources_list, class_list, image_size = (128,128,3))
CNN_train_test_ratio = 0.75
X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, CSN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    
feat_mean_reg, class_dict, semantic_net_sld = CSN_centroid_finder(X_train, Y_train, CSN_reg_model, class_dict, save_data = True, save_dir = CSN_folder + '/0.75_region_aug_128_CNN4convgrande_mkiii')

sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/extract pasto_trebol/con_wl_aug')
#sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/region_aug')
X, Y, class_dict = multiclass_preprocessing(sources_list, class_list, image_size = (64,64,3))
CNN_train_test_ratio = 0.75
X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, CSN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    
feat_mean_sld, class_dict, semantic_net_sld = CSN_centroid_finder(X_train, Y_train, CSN_sld_model, class_dict, save_data = True, save_dir = CSN_folder + '/0.75_extractpastotrebol_conwl_aug_noise_64_CNN4conv_mini_deep')
'''
with open(CSN_folder + '/0.75_region_aug_128_CNN4convgrande_mkiii' + '/feat_dict.pickle', 'rb') as handle: feat_mean_reg_dict = pickle_load( handle) 

with open(CSN_folder + '/0.75_extractpastotrebol_conwl_aug_noise_64_CNN4conv_mini_deep' + '/feat_dict.pickle', 'rb') as handle: feat_mean_sld_dict = pickle_load( handle) 

feat_mean_reg = feat_mean_reg_dict['feat_mean']
feat_mean_sld = feat_mean_sld_dict['feat_mean']
class_dict_reg = feat_mean_reg_dict['class_dict']
class_dict_sld = feat_mean_sld_dict['class_dict']

#sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/multiclass_felo_aug')
#X, Y, class_dict = multiclass_preprocessing(sources_list, class_list, image_size = img_shape)
class_n = 3#len(class_dict['class_n_list'])
#print(getsizeof(X))

CNN_train_test_ratio = 0.75
#X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, CNN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    

#feat_mean_sld, _, semantic_net_sld = CSN_centroid_finder(X_train, Y_train, CSN_sld_model, class_dict)

#semantic_X = semantic_net.predict(X_train)
img = cv2.imread(cwd + '/Pasto/IMG_20190818_124623.jpg')
'''
h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
v = v.astype(np.int) + 10 
v[v>255] = 255
img_new = cv2.cvtColor(cv2.merge([h.astype(np.uint8),s.astype(np.uint8),v.astype(np.uint8)]), cv2.COLOR_HSV2BGR)
'''
CSN_sld_model, session = best_model_finder(CSN_folder + '/0.75_extractpastotrebol_conwl_aug_noise_64_CNN4conv_mini_deep', CNN_class_name = 'CSN')
CNN_reg_model,_ = best_model_finder(CNN_folder + '/0.75_region_aug_128_CNN4convgrande_mkii', CNN_class_name = 'CNN') 
CNN_sld_model,_ = best_model_finder(CNN_folder + '/0.75_extractpastotrebol_conwl_aug_64_CNN4conv_mini_deep', CNN_class_name = 'CNN') 

#CSN_region_seg(img, CSN_reg_model, class_dict, feat_mean = feat_mean_reg, method = 'sld_win_CSN', win_size = (1800, 1800), min_frames_region = 2, overlap_factor = .25, thresh = 'mean_half', \
#    multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce' , IOU = .25 )
'''
pipeline(img, CNN_reg_model, CNN_sld_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = feat_mean_reg, feat_mean_second= feat_mean_sld, overlap_factor_first = 0.85,\
    overlap_factor_second = 0.85, multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce', IOU = .25, heat_map_class = 'trebol', heat_map_display = True, bg_class = 'pasto',\
        class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'], savedir = cwd + '/seg_imgs' )

pipeline(img, CSN_reg_model, CNN_sld_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = feat_mean_reg, feat_mean_second= feat_mean_sld, overlap_factor_first = 0.85,\
    overlap_factor_second = 0.85, multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce', IOU = .25, heat_map_class = 'trebol', heat_map_display = True, bg_class = 'pasto',\
        class_mask_display = True, method = 'region_sld', model_type = ['CSN', 'CNN'] )
pipeline(img, CSN_reg_model, CSN_sld_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = feat_mean_reg, feat_mean_second= feat_mean_sld, overlap_factor_first = 0.85,\
    overlap_factor_second = 0.85, multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce', IOU = .25, heat_map_class = 'trebol', heat_map_display = True, bg_class = 'pasto',\
        class_mask_display = True, method = 'region_sld', model_type = ['CSN', 'CSN'] )
'''
CSN_region_seg(img, CSN_reg_model, class_dict_reg, feat_mean = feat_mean_reg, method = 'sld_win_CSN', win_size = (1800, 1800), min_frames_region = 2, overlap_factor = .75, thresh = 'mean_half', \
    multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce' )
kll

img = cv2.imread(cwd + '/Pasto/IMG_20190818_124623.jpg')
CSN_region_seg(img, CSN_reg_model, class_dict, feat_mean = feat_mean_reg, method = 'box_region', win_size = (1800, 1800), min_frames_region = 2, overlap_factor = .75, thresh = 'mean_half', \
    multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce' , IOU = .25 )

img = cv2.imread(cwd + '/Pasto/IMG_20190818_124625.jpg')
CSN_region_seg(img, CSN_reg_model, class_dict, feat_mean = feat_mean_reg, method = 'box_region', win_size = (1800, 1800), min_frames_region = 2, overlap_factor = .75, thresh = 'mean_half', \
    multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce' , IOU = .25 )

img = cv2.imread(cwd + '/Pasto/IMG_20190818_124626.jpg')
CSN_region_seg(img, CSN_reg_model, class_dict, feat_mean = feat_mean_reg, method = 'box_region', win_size = (1800, 1800), min_frames_region = 2, overlap_factor = .75, thresh = 'mean_half', \
    multi_res_win_size = (1800, 1800), multi_res_name = 'wild lettuce' , IOU = .25 )

raise ValueError('Adieu')
'''
'''
frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = slide_window_creator(img, win_size = (1800, 1800), new_win_shape = (64,64), overlap_factor = 0.25, data_type = 'CNN')
semantic_X = np.repeat(semantic_net_reg.predict(frame_array_in)[:,np.newaxis,:], len(class_list), axis = 1)
d_mat = np.sqrt( np.sum(np.square(semantic_X-feat_mean_reg), axis = 2) )
CSN_pred = 1-d_mat
print(semantic_X.shape)
print(CSN_pred.shape)
class_pred = np.argmax(CSN_pred, axis = 1)
print(class_pred.shape)
print(np.max(CSN_pred))

#semantic_X = np.repeat(semantic_net_sld.predict(frame_array_in)[:,np.newaxis,:], len(class_list), axis = 1)
#d_mat = np.sqrt( np.sum(np.square(semantic_X-feat_mean_sld), axis = 2) )
#CSN_pred = 1-d_mat
#print(semantic_X.shape)
#print(CSN_pred.shape)
#class_pred_sld = np.argmax(CSN_pred, axis = 1)
#print(class_pred.shape)
#print(np.max(CSN_pred))

for class_idx in range(CSN_pred.shape[1]):
    pred_vector_coordinates = (class_pred == class_idx) & ( CSN_pred[:, class_idx] > 0.5) 
    pred_vector = CSN_pred[pred_vector_coordinates ,class_idx]
    class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
        frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
    
    frame_coordinates_in[(class_pred == class_idx) & ( CSN_pred[:, class_idx] > 0.5),:]
    print(pred_vector)
    print(class_points)
    bounding_box_center, bounding_box_wh = NMS_bb(class_points, pred_vector, 1800, IOU = .25)
    print(bounding_box_center.shape[0])
    print(pred_vector.shape[0])
    
    for i in range(bounding_box_center.shape[0]):
        bb = bounding_box_center[i]
        wh = bounding_box_wh[i]
        cv2.imshow(class_dict['class_name_list'][class_idx],cv2.resize(img[np.max([bb[0] - int(wh[0]), 0]):bb[0] , np.max([bb[1] - int(wh[1]), 0]):bb[1] ], (480,480)))
        cv2.waitKey(0)
raise ValueError('Voila!')

###############################################################################################################################

##############################################################################################
##PARA VER QUE "VE" LA RED NEURONAL CSN
cv2.imwrite(cwd + '/Pasto/weed.jpg', cv2.resize( cv2.imread(cwd + '/Pasto/IMG_20190818_124629.jpg'), (500, 334) ) )
#img_shape = (128,128,3)
img_shape = (64,64,3)
sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/extract pasto_trebol/con_wl_aug')
X, Y, class_dict = multiclass_preprocessing(sources_list, class_list, image_size = img_shape)
class_n = len(class_dict['class_n_list'])
class_name_list = class_dict['class_name_list']
print(getsizeof(X))

CNN_train_test_ratio = 0.75
X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, CNN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    
CSN_folder = cwd + '/Pasto/extract/data/CSN_ral'
CNN_folder = cwd + '/Pasto/extract/data/CNN_ral'
model_sld, session = best_model_finder(CNN_folder + '/0.75_extractpastotrebol_conwl_aug_64_CNN4conv_mini_deep', CNN_class_name = 'CNN')


feat_mean, _, _ = CSN_centroid_finder(X_train, Y_train, model_sld, class_dict, save_data = True, save_dir = CSN_folder + '/0.75_extractpastotrebol_conwl_aug_noise_64_CNN4conv_mini_deep')
conf_matrix(model_sld, 'CNN', X_test, Y_test, feat_mean = feat_mean, class_names = class_dict['class_name_list'], destiny = CNN_folder + '/conf_matrix_test_sld')
conf_matrix(model_sld, 'CNN', X_val, Y_val, feat_mean = feat_mean, class_names = class_dict['class_name_list'], destiny = CNN_folder + '/conf_matrix_val_sld')
conf_matrix(model_sld, 'CNN', X, Y, feat_mean = feat_mean, class_names = class_dict['class_name_list'], destiny = CNN_folder + '/conf_matrix_tot_sld')
#REGION
img_shape = (128,128,3)
#img_shape = (64,64,3)
sources_list, class_list = multiclassparser(cwd + '/Pasto/extract/region_aug')
X, Y, class_dict = multiclass_preprocessing(sources_list, class_list, image_size = img_shape)
class_n = len(class_dict['class_n_list'])
class_name_list = class_dict['class_name_list']
print(getsizeof(X))

CNN_train_test_ratio = 0.75
X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_set_creator(X, Y, class_dict, CNN_folder, train_test_rate = CNN_train_test_ratio, test_val_rate = 0.5, CNN_class='CNN')    
CSN_folder = cwd + '/Pasto/extract/data/CSN_ral'
CNN_folder = cwd + '/Pasto/extract/data/CNN_ral'
model_reg, session = best_model_finder(CNN_folder + '/0.75_region_aug_128_CNN4convgrande_mkii', CNN_class_name = 'CNN')


feat_mean, _, _ = CSN_centroid_finder(X_train, Y_train, model_reg, class_dict, save_data = True, save_dir = CSN_folder + '/0.75_region_aug_128_CNN4convgrande_mkiii')
conf_matrix(model_reg, 'CNN', X_test, Y_test, feat_mean = feat_mean, class_names = class_dict['class_name_list'], destiny = CNN_folder + '/conf_matrix_test_reg')
conf_matrix(model_reg, 'CNN', X_val, Y_val, feat_mean = feat_mean, class_names = class_dict['class_name_list'], destiny = CNN_folder + '/conf_matrix_val_reg')
conf_matrix(model_reg, 'CNN', X, Y, feat_mean = feat_mean, class_names = class_dict['class_name_list'], destiny = CNN_folder + '/conf_matrix_tot_reg')

raise ValueError('Listo')

sem_model = Sequential()
feat_model = Sequential()
CNN_class_name = 'CSN'
autoencoder = True
for idx, layer in enumerate(model.layers):
    if CNN_class_name == 'CSN':
        if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
            for in_layer in layer.layers:
                if isinstance(in_layer, Model):print(in_layer.summary())
                if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                    sem_model.add(in_layer)
                    feat_model.add(in_layer)
                elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model.add(in_layer)
    else:
        if not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense): 
            sem_model.add(layer)
            feat_model.add(layer)
        elif ( isinstance(layer, Dense) and idx < len(model.layers) -1 ) or isinstance(layer, Flatten):
            feat_model.add(layer)
            
feat_fig, feat_ax = plt.subplots()  
linefmt = ['r:', 'c:', 'k:']    
markerfmt_train = ['rx', 'cx', 'kx'] 
markerfmt_val = ['rX', 'cX', 'kX'] 
for class_num in range(class_n):
    
    X_n_train = np.reshape( X_train[(Y_train == class_num)[:,0],:,:,:], (np.where( ( Y_train == class_num)[:,0] == True)[0].shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3] ) )
    X_n_train_mean = np.reshape ( np.sum(sem_model.predict(X_n_train, batch_size=1), axis = 0) / X_n_train.shape[0], (16,15) )
    feat_n_mean_train = np.sum(feat_model.predict(X_train[(Y_train == class_num)[:,0],:,:,:], batch_size=1), axis = 0) /  X_n_train.shape[0]
    print(feat_n_mean_train)
    feat_n_mean_train = feat_n_mean_train / ( np.max(feat_n_mean_train) +1e-6) 
    
    X_n_val = np.reshape( X_val[(Y_val == class_num)[:,0],:,:,:], (np.where( ( Y_val == class_num)[:,0] == True)[0].shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3] ) )
    X_n_val_mean = np.sum(sem_model.predict(X_n_val, batch_size=1), axis = 0) / X_n_val.shape[0]
    feat_n_mean_val = np.sum(feat_model.predict(X_val[(Y_val == class_num)[:,0],:,:,:], batch_size=1), axis = 0) /  X_n_val.shape[0]
    print(feat_n_mean_val)
    feat_n_mean_val = feat_n_mean_val / ( np.max(feat_n_mean_val) + 1e-6)
    
    feat_xaxis = np.linspace(0, feat_n_mean_train.shape[0]-1, num = feat_n_mean_train.shape[0])
    feat_ax.stem(feat_xaxis, feat_n_mean_train, linefmt = linefmt[class_num], markerfmt = markerfmt_train[class_num], label = 'train' + class_name_list[class_num])
    
    feat_xaxis = np.linspace(0, feat_n_mean_val.shape[0]-1, num = feat_n_mean_val.shape[0])
    feat_ax.stem(feat_xaxis, feat_n_mean_val, linefmt = linefmt[class_num], markerfmt = markerfmt_val[class_num], label = 'val' + class_name_list[class_num])
    
    print(feat_n_mean_train.shape)
    cv2.imshow('Clase promedio: ' + str(class_num) , cv2.resize(X_n_train_mean, (950,950)))
    cv2.waitKey(0)
    
plt.legend()
plt.show(feat_fig)
raise ValueError('Termina demo')
for idx, pred in enumerate(sem_model.predict(X_train)):
    print(pred.shape)
    cv2.imshow('Predicción de clase: ' + class_dict['class_name_list'][int(Y_train[idx])],cv2.resize(pred/np.max(pred), (480,480) ) )
    cv2.waitKey(0)


cv2.destroyAllWindows()
model, session = best_model_finder(CNN_folder + '/0.75_region_aug', CNN_class_name = 'CNN')
sem_model = Sequential()
for layer in model.layers:
    if not isinstance(layer, Flatten) and not isinstance(layer, Dense): sem_model.add(layer)
for idx, pred in enumerate(sem_model.predict(X_train, batch_size= 1)):
    cv2.imshow(str(Y_train[idx]),cv2.resize(pred, (480,480) ) )
    cv2.waitKey(0)

#######################################################################################################


#Creación de una red neuronal de tipo CSN con arquitectura encoder-decoder
CNN_params_obj = CNN_params('CSN', class_n = 2, learning_rate = 0.001)
CNN_params_obj.conv2dlayer_shape_adder([50, 50, 4])
CNN_params_obj.conv2dlayer_shape_adder([20, 20, 8])
CNN_params_obj.conv2dlayer_shape_adder([8, 8, 10])
CNN_params_obj.conv2dlayer_shape_adder([4, 4, 15])

#CNN_params_obj.conv2dlayer_shape_adder([2, 2, 15])

#CNN_params_obj.conv2dlayer_shape_adder([8, 8, 10])
#CNN_params_obj.conv2dlayer_shape_adder([20, 20, 8])
#CNN_params_obj.conv2dlayer_shape_adder([50, 50, 4])

#Creación del objeto que contiene las opciones para la sesión de entrenamiento. En orden: batch size, epochs totales, verbose (0 ó 1), shuffle (True o False) y epochs mínimas de entrenamiento
CNN_train_params_obj = CNN_train_params(128, 225, 0, True, 450)

#Iteración sobre un rango de capas de clasificación, incluyendo la cantidad de capas como entrada. En orden, las entradas son:

#CNN_params_obj: Objeto con la clase de red (CNN, TL o CSN) y la arquitectura de las capas convolucionales
#CNN__train_params_obj: Objeto con las opciones de entrenamiento
#img_shape: La forma de las imágenes de entrada, dada por el preprocesamiento anterior
#CSN_folder: La carpeta donde se guardarán las redes, coincide con la carpeta donde se crearon los conjuntos
#destiny: El nombre de la sub-carpeta donde se guardará cada red, para esta parte es irrelevante
#model_name: Por ahora no importa    
#class_layers_range: Cantidad de capas de clasificación previas a la capa de dos etiquetas
#class_neuron_range: Rango de neuronas de clasificación sobre las que se iterará. Si la lista tiene 2 números se revisan uno a uno, si son 3 se revisan la cantidad de neuronas del tercer valor dentro
#del rango especificado
#last_check: Revisa el último estado de iteración (si se había partido antes y quedó a la mitad)
#ntimes: Número de veces que se reinicializa la red, para disminuir el sesgo por mala inicialización (VER COMO SE PODRÍA EVITAR ESTO, COSTOSO EN TIEMPO)
#tf_seed: Semilla para los procesos randomizados de TF y KERAS

img_shape = (64, 64, 3)

#Descomentar para entrenar la red
###############################################################################################################################
CSN_train_test_ratio = 0.75
CSN_folder = CSN_folder = cwd + '/Pasto/extract/data/CSN_felo'
CNN_train_iterator(CNN_params_obj, CNN_train_params_obj,  img_shape, CSN_folder, destiny = str(CSN_train_test_ratio) + '_conwlaug', model_name = 'generico',\
class_layers_range = 1, class_neuron_range = [32, 64, 3], last_check = True,  ntimes = 8, tf_seed = 1, acc_vec=True)

img = cv2.imread(cwd + '/Pasto/IMG_20190818_124625.jpg')
#cv2.imshow('Imagen a segmentar CSN', cv2.resize( img, None, fx = 0.25, fy = 0.25 ) )
#cv2.waitKey(0)
model, session = best_model_finder(CSN_folder + '/0.75_aug', CNN_class_name = 'CSN')
pipeline(img, model, class_dict, frame_size = (256, 256, 3), X_train = X_train, Y_train = Y_train, overlap_factor = 0.75 , model_type='CSN', multi_res_name= 'wild lettuce')
feat_mean, class_dict, semantic_model = CSN_centroid_finder(X_train, Y_train, model, class_dict)
CSN_region_seg(img, model, X_train, Y_train, class_dict, method = 'sld_win_feat', win_size = (256, 256), min_frames_region = 32, overlap_factor = 0.5, thresh = 'mean_half', multi_res_name = 'wild lettuce')
#CSN_centroid_pred(img, model, X_train, Y_train, class_dict, win_size = (256, 256), new_win_shape = (64, 64), overlap_factor=0.75)
#CSN_region_seg(img, model, session, X_train, Y_train, class_dict, method = 'sld_win_CSN', win_size = (256, 256), min_frames_region = 32, overlap_factor=0.25, thresh = 'mean_half')
kldskla
###############################################################################################################################

#Creación de una red neuronal de tipo TL con arquitectura encoder-decoder
CNN_params_obj = CNN_params('TL', class_n = class_n)
CNN_params_obj.conv2dlayer_shape_adder([50, 50, 4])
CNN_params_obj.conv2dlayer_shape_adder([20, 20, 8])
CNN_params_obj.conv2dlayer_shape_adder([8, 8, 10])

CNN_params_obj.conv2dlayer_shape_adder([8, 8, 10])
CNN_params_obj.conv2dlayer_shape_adder([20, 20, 8])
CNN_params_obj.conv2dlayer_shape_adder([50, 50, 4])

#Creación del objeto que contiene las opciones para la sesión de entrenamiento. En orden: batch size, epochs totales, verbose (0 ó 1), shuffle (True o False) y epochs mínimas de entrenamiento
CNN_train_params_obj = CNN_train_params(16, 1000, 0, True, 200)

#Iteración sobre un rango de capas de clasificación, incluyendo la cantidad de capas como entrada. En orden, las entradas son:

#CNN_params_obj: Objeto con la clase de red (CNN, TL o CSN) y la arquitectura de las capas convolucionales
#CNN__train_params_obj: Objeto con las opciones de entrenamiento
#img_shape: La forma de las imágenes de entrada, dada por el preprocesamiento anterior
#CNN_folder: La carpeta donde se guardarán las redes, coincide con la carpeta donde se crearon los conjuntos
#destiny: El nombre de la sub-carpeta donde se guardará cada red, para esta parte es irrelevante
#model_name: Por ahora no importa    
#class_layers_range: Cantidad de capas de clasificación previas a la capa de dos etiquetas
#class_neuron_range: Rango de neuronas de clasificación sobre las que se iterará. Si la lista tiene 2 números se revisan uno a uno, si son 3 se revisan la cantidad de neuronas del tercer valor dentro
#del rango especificado
#last_check: Revisa el último estado de iteración (si se había partido antes y quedó a la mitad)
#ntimes: Número de veces que se reinicializa la red, para disminuir el sesgo por mala inicialización (VER COMO SE PODRÍA EVITAR ESTO, COSTOSO EN TIEMPO)
#tf_seed: Semilla para los procesos randomizados de TF y KERAS

img_shape = (64, 64, 3)

#Descomentar para entrenar la red
###############################################################################################################################
CNN_train_iterator(CNN_params_obj, CNN_train_params_obj,  img_shape, TL_folder, destiny = '0.75', model_name = 'generico',\
class_layers_range = 2, class_neuron_range = [128, 256, 9], last_check = True,  ntimes = 10, tf_seed = 1)  
###############################################################################################################################