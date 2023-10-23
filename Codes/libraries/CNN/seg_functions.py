#LIBRARY FOR SEGMENTATION OF OBJECTS WITHIN AN IMAGE BASED ON NEURAL NETWORKS/SVM 
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Import of external libraries
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from pickle import  dump as pickle_dump, HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL
import cv2
from sklearn.cluster import DBSCAN
import time
from shutil import rmtree

##KERAS##
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Lambda, GlobalAveragePooling2D
from keras.utils import to_categorical

##
##Import of own libraries
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from data_handling.img_processing import slide_window_creator
from CNN.csn_functions import CSN_centroid_pred
from SVM.feature_functions import lbp_feats_calc, hog_feats_calc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Implementation of own functions

#Function that performs a simple NMS or calculates the average bounding box
def NMS_bb(class_points, pred_vector, win_size, IOU = 0.5, method = 'average_box'):
    #The maximum value within the prediction set is found
    pred_vector = list(pred_vector)
    class_points = list(class_points)
    bounding_box_center = np.zeros((0))
    bounding_box_wh = np.zeros((1))
    bounding_box_score = np.zeros(0)
    i = 0
    #As long as all points are not discarded, iteration continues
    while len(pred_vector) > 0:
        #If the bounding box method is chosen as local 'absolute' maximum
        if method == 'absolute_max':
            i += 1
            #The maximum point is found with its corresponding coordinate and score
            max_pred_idx = np.argmax(pred_vector)
            max_bb_pred = pred_vector.pop(max_pred_idx)
            max_bb_point = class_points.pop(max_pred_idx)
            other_pred = []
            #For the remaining maximum points, the IOU test is performed
            for idx, point in enumerate(class_points):
                if pred_vector[idx] < max_bb_pred  and\
                ( win_size[idx] - abs(point[0] - max_bb_point[0]) )>0 and ( win_size[idx] - abs(point[1] -max_bb_point[1]) ) > 0:
                    if ( win_size[idx] - abs(point[0] - max_bb_point[0] ) ) *  (win_size[idx] - abs(point[1] - max_bb_point[1] ) ) > (IOU * win_size[idx]**2): other_pred.append(idx)
            #All points within the IOU region are discarded
            other_pred.reverse()
            for pop_idx in other_pred:
                class_points.pop(pop_idx)
                pred_vector.pop(pop_idx)
            #The data of the central coordinate of each BB and its width and height are attached
            if bounding_box_center.shape[0] == 0:
                bounding_box_center = np.reshape( np.array( [max_bb_point[0],max_bb_point[1]] ), (1, 2) ).astype(int)
                bounding_box_wh = np.reshape( np.array( [win_size[max_pred_idx], win_size[max_pred_idx]] ), (1,2) ).astype(int)
                bounding_box_score = np.reshape( np.array( [max_bb_pred] ), (1, 1) )
            else:
                bounding_box_center = np.concatenate ( (bounding_box_center,  np.reshape( np.array( [max_bb_point[0], max_bb_point[1]] ), (1, 2) ) ), axis = 0 ).astype(int)
                bounding_box_wh = np.concatenate ( ( bounding_box_wh, np.reshape( np.array( [win_size[max_pred_idx], win_size[max_pred_idx]] ), (1,2) ) ), axis = 0).astype(int)
                bounding_box_score = np.concatenate ( ( bounding_box_score, np.reshape( np.array( [max_bb_pred] ), (1,1) ) ), axis = 0)
        #If the average bounding box method is chosen
        elif method == 'average_box':
            pred_vector.reverse(), class_points.reverse()
            win_size =  np.flip(win_size)
            #The maximum point is found with its corresponding coordinate and score
            max_pred_idx = np.argmax(pred_vector)
            max_bb_pred = pred_vector.pop(max_pred_idx)
            max_bb_point = class_points.pop(max_pred_idx)
            other_pred = []
            older_pred = [max_bb_point]
            #Average values for new bounding boxes
            avg_point = max_bb_point * max_bb_pred
            avg_size = win_size[max_pred_idx] * max_bb_pred
            avg_ct = max_bb_pred
            #For the remaining points at the maximum, the IOU test is performed, and if it is estimated that they correspond to the same object, their statistics
            #are added together for the final detection
            for idx, point in enumerate(class_points):
                for old_point_idx, old_point in enumerate(older_pred):
                    if ( win_size[idx]*.5 + win_size[old_point_idx]*.5  - abs(point[0] - old_point[0]) ) * ( win_size[idx]*.5 + win_size[old_point_idx]*.5 - abs(point[1] -old_point[1]) ) > (IOU * ( (win_size[idx]*.5 + win_size[old_point_idx]*.5)**2)) and \
                        ( win_size[idx]*.5 + win_size[old_point_idx]*.5 - abs(point[0] - old_point[0]) )>0 and ( win_size[idx]*.5 + win_size[old_point_idx]*.5 - abs(point[1] -old_point[1]) ) > 0:
                        avg_point[0], avg_point[1] = avg_point[0] + point[0] * pred_vector[idx] , avg_point[1] + point[1] * pred_vector[idx]
                        avg_size += win_size[idx] * pred_vector[idx]
                        other_pred.append(idx)
                        older_pred.append(point)
                        avg_ct += pred_vector[idx]
                        break
            #If the number of predictions within a neighbourhood is below (X), the point is considered a misdetection
            if len(other_pred)<3: print('Insufficient number of predictions, point discarded')
            #All points within the IOU region are discarded
            else:
                print('nr poitns',len(other_pred))
                other_pred.reverse()
                for pop_idx in other_pred:
                    class_points.pop(pop_idx)
                    pred_vector.pop(pop_idx)
                #The operation is repeated to discard overlapping frames
                #other_pred = []
                #for idx, point in enumerate(class_points):
                #    for old_point in older_pred:
                #        if ( win_size[idx]*.5 + win_size[old_point_idx]*.5  - abs(point[0] - old_point[0]) ) * ( win_size[idx]*.5 + win_size[old_point_idx]*.5 - abs(point[1] -old_point[1]) ) > (IOU * ( (win_size[idx]*.5 + win_size[old_point_idx]*.5)**2)) and \
                #            ( win_size[idx]*.5 + win_size[old_point_idx]*.5 - abs(point[0] - old_point[0]) )>0 and ( win_size[idx]*.5 + win_size[old_point_idx]*.5 - abs(point[1] -old_point[1]) ) > 0:
                #            avg_point[0], avg_point[1] = avg_point[0] + point[0] * pred_vector[idx] , avg_point[1] + point[1] * pred_vector[idx]
                #            avg_size += win_size[idx] * pred_vector[idx]
                #            other_pred.append(idx)
                #            older_pred.append(point)
                #            avg_ct += pred_vector[idx]
                #            break
                #Average size and coordinate are calculated through weighed average division
                avg_point = np.round( avg_point / avg_ct ).astype(np.int)
                avg_size = np.round( avg_size / avg_ct ).astype(np.int)
                print('avg_size: ' + str(avg_size)), print('avg_point: ' + str(avg_point))
                #All points within the IOU region are discarded
                #other_pred.reverse()
                #for pop_idx in other_pred:
                #    class_points.pop(pop_idx)
                #    pred_vector.pop(pop_idx)
                #The data of the central coordinate of each BB and its width and height are attached
                if bounding_box_center.shape[0] == 0:
                    bounding_box_center = np.reshape( np.array( [avg_point[0],avg_point[1]] ), (1, 2) ).astype(int)
                    bounding_box_wh = np.reshape( np.array( [avg_size, avg_size] ), (1,2) ).astype(int)
                    bounding_box_score = np.reshape( np.array( [max_bb_pred] ), (1, 1) )
                else:
                    bounding_box_center = np.concatenate ( (bounding_box_center,  np.reshape( np.array( [avg_point[0], avg_point[1]] ), (1, 2) ) ), axis = 0 ).astype(int)
                    bounding_box_wh = np.concatenate ( ( bounding_box_wh, np.reshape( np.array( [avg_size, avg_size] ), (1,2) ) ), axis = 0).astype(int)
                    bounding_box_score = np.concatenate ( ( bounding_box_score, np.reshape( np.array( [max_bb_pred] ), (1,1) ) ), axis = 0)
    return bounding_box_center, bounding_box_wh, bounding_box_score

#Averaging function with 'n' neighbors for smoothing predictions
def pred_lowpass(pred_mat, neighbours = 1):
    #For each prediction value, a smoothed vector is created
    if len(pred_mat.shape) == 2:
        pred_vector_lp = np.zeros(pred_mat.shape)
        pred_vector_lp = 0.5*pred_mat[:,:]
        for pred_idx in range(pred_mat.shape[0]):
            if neighbours > 0:
                for n in range(1, int(neighbours/2+1)): 
                    pred_vector_lp[pred_idx,:] += (0.5/neighbours) * pred_mat[np.max([0, pred_idx-n]),:] + (0.5/neighbours) * pred_mat[np.min([pred_mat.shape[0]-1, pred_idx+n]),:]
            else:
                pred_vector_lp[pred_idx,:] += pred_vector_lp[pred_idx,:]  
        return pred_vector_lp
    else:
        pred_vector_lp = np.zeros(pred_mat.shape)
        pred_vector_lp = 0.5*pred_mat[:,:,:]
        for pred_r in range(pred_mat.shape[0]):
            for pred_c in range(pred_mat.shape[1]):
                if neighbours > 0:
                    for n in range(1, int(neighbours+1)): 
                        pred_vector_lp[pred_r, pred_c,:] += (.0625/neighbours) * pred_mat[np.max([0, pred_r-n]), np.max([0, pred_c-n]),:] + (.0625/neighbours) * pred_mat[np.max([0, pred_r-n]), pred_c,:]\
                            + (.0625/neighbours) * pred_mat[np.max([0, pred_r-n]), np.min([pred_c+n, pred_mat.shape[1]-1]),:] + (.0625/neighbours) * pred_mat[pred_r, np.max([0, pred_c]),:]\
                            + (.0625/neighbours) * pred_mat[pred_r, np.min([pred_c+n, pred_mat.shape[1]-1]),:]\
                                + (.0625/neighbours) * pred_mat[np.min([pred_r, pred_mat.shape[0]-1]), np.max([0, pred_c-n]),:]\
                                    + (.0625/neighbours) * pred_mat[np.min([pred_r, pred_mat.shape[0]-1]), pred_c,:]\
                                        + (.0625/neighbours) * pred_mat[np.min([pred_r, pred_mat.shape[0]-1]), np.min([pred_c, pred_mat.shape[1]-1]),:]
                else:
                    pred_vector_lp[pred_r, pred_c,:] += pred_vector_lp[pred_r, pred_c,:]
        return pred_vector_lp
    
#Function that segments regions over an image using a SVM model
def SVM_region_seg(img_rgb, model, class_dict, selection_dict, feats_param_dict, multi_res_win_size = (1440, 1440), multi_res_name = 'LV', method = 'box_region',\
    overlap_factor_heatmap = .75, overlap_factor_multires = 0.5, IOU_multires = .25, IOU_hm = .75, pred_batch = 32, r_neighbours = 0, region_wl_thresh = 0.5,\
        heatmap_name = 'trebol', region_hm_thresh = .5):
    #The list of classes and the size of the image that will enter the network are loaded
    class_list = class_dict['class_n_list']
    class_name_list = class_dict['class_name_list']
    new_win_shape = selection_dict['img_shape']
    img_gray = cv2.equalizeHist( cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) )
    #If segmentation is done by bounding boxes
    if method == 'box_region':
        #If only one multiresolution window size is delivered
        frame_size_array = []
        if type(multi_res_win_size) == tuple:
            #The frame array for multiresolution prediction is created
            frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
            slide_window_creator(img_gray, win_size = multi_res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_multires, data_type = 'CNN')
            frame_array_in = np.float32(frame_array_in)/255
            #Features are computed for the SVM classifier
            for frame_count, frame in enumerate(frame_array_in):
                lbp_frame_size, lbp_type, lbp_points, lbp_r = feats_param_dict['lbp_frame_size'], feats_param_dict['lbp_type'], feats_param_dict['lbp_points'], feats_param_dict['lbp_r']
                pixels_x, pixels_y, block_num_x, block_num_y, orientations_n = feats_param_dict['hog_pixels_x'], feats_param_dict['hog_pixels_y'], feats_param_dict['block_num_x'], feats_param_dict['block_num_y'], feats_param_dict['orientations_n']
                #Calculation of HOG and LBP features
                lbp_total = np.asarray( lbp_feats_calc(frame, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
                hog_total = np.asarray( hog_feats_calc(frame, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
                #If it is the first image processed, feature and label vectors are created. If not, they are concatenated
                if frame_count == 0: X = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                else: X = np.concatenate( ( X,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
            X_clean_indices = selection_dict['X_train_clean_indices']
            X = X[:, X_clean_indices]
            class_pred = to_categorical(model.predict(X), num_classes = len(class_name_list))
            class_mat = np.reshape(class_pred, (frames_r, frames_c, len(class_list)))
            class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
            class_pred_lp_flat = np.reshape(class_pred_lp, (class_pred.shape[0], class_pred.shape[1]))
            class_pred_lp = class_pred_lp_flat
            for _ in range(class_pred.shape[0]) : frame_size_array.append(multi_res_win_size[0])
        #If a list of window sizes is entered
        elif type(multi_res_win_size) == list:
            #The frame array for multiresolution prediction is created
            new_frame_coordinates_in  = np.zeros(0)
            frame_size_array = []
            for res_win_size in multi_res_win_size:
                frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
                slide_window_creator(img_gray, win_size = res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_multires, data_type = 'CNN')
                frame_array_in = np.float32(frame_array_in)/255
                #Features are computed for the SVM classifier
                for frame_count, frame in enumerate(frame_array_in):
                    lbp_frame_size, lbp_type, lbp_points, lbp_r = feats_param_dict['lbp_frame_size'], feats_param_dict['lbp_type'], feats_param_dict['lbp_points'], feats_param_dict['lbp_r']
                    pixels_x, pixels_y, block_num_x, block_num_y, orientations_n = feats_param_dict['hog_pixels_x'], feats_param_dict['hog_pixels_y'], feats_param_dict['block_num_x'], feats_param_dict['block_num_y'], feats_param_dict['orientations_n']
                    #Calculation of HOG and LBP features
                    lbp_total = np.asarray( lbp_feats_calc(frame, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
                    hog_total = np.asarray( hog_feats_calc(frame, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
                    #If it is the first image processed, feature and label vectors are created. If not, they are concatenated
                    if frame_count == 0: X = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                    else: X = np.concatenate( ( X,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
                X_clean_indices = selection_dict['X_train_clean_indices']
                X = X[:, X_clean_indices]
                class_pred = to_categorical(model.predict(X), num_classes = len(class_name_list))
                class_mat = np.reshape(class_pred, (frames_r, frames_c, len(class_list)))
                class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
                class_pred_lp_flat = np.reshape(class_pred_lp, (class_pred.shape[0], class_pred.shape[1]))
                class_pred_lp = class_pred_lp_flat
                #If the process had already been started, the data are added to the vector of predictions and coordinates. If not, they are created from scratch
                if new_frame_coordinates_in.shape[0] == 0:
                    new_frame_coordinates_in = frame_coordinates_in
                    new_class_pred = class_pred_lp
                else:
                    new_frame_coordinates_in = np.concatenate( ( new_frame_coordinates_in, frame_coordinates_in ) )
                    new_class_pred = np.concatenate( ( new_class_pred, class_pred_lp ) )
                for _ in range(class_pred.shape[0]) : frame_size_array.append(res_win_size[0])
            #The value of the prediction vector and coordinates is updated for later on
            class_pred_lp = new_class_pred
            frame_coordinates_in = new_frame_coordinates_in       
        #If an invalid format is entered for window sizes
        else: raise ValueError('The only formats admitted for multi-resolution windows are tuple and list')
        #For each class a bounding box + NMS detection is performed
        region_coordinates = []
        bb_wh_list = []
        region_pred = [ [] for _ in range(len(class_list)) ]
        region_coordinates = [ [] for _ in range(len(class_list)) ]
        region_pred = [ [] for _ in range(len(class_list)) ]
        bb_wh_list =  [ [] for _ in range(len(class_list)) ]
        class_n_multires = class_dict['class_name_list'].index(multi_res_name)
        multires_coord = np.where(class_pred_lp[:,class_n_multires]> region_wl_thresh )[0]
        multires_points = np.zeros(0)
        for coord_idx in multires_coord:
            if multires_points.shape[0] == 0: multires_points = np.reshape( np.array([frame_coordinates_in[coord_idx,0]*.5+frame_coordinates_in[coord_idx,1]*.5 ,\
                frame_coordinates_in[coord_idx,2]*.5+frame_coordinates_in[coord_idx,3]*.5 ]), (1, 2))
            else: multires_points = np.concatenate( (multires_points,  np.reshape( np.array([frame_coordinates_in[coord_idx,0]*.5+frame_coordinates_in[coord_idx,1]*.5 ,\
                frame_coordinates_in[coord_idx,2]*.5+frame_coordinates_in[coord_idx,3]*.5 ]), (1, 2))), axis = 0 )
        for class_idx in range(class_pred_lp.shape[1]):
            #The bounding boxes are calculated and added to the index corresponding to the predicted class
            if class_idx == class_dict['class_name_list'].index(multi_res_name): 
                pred_vector_coordinates = class_pred_lp[:,class_idx]>=region_wl_thresh
                pred_vector = class_pred_lp[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                    frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector, np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_multires, method = 'average_box')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = class_pred_lp[pred_vector_coordinates, class_idx]
            elif class_idx == class_dict['class_name_list'].index(heatmap_name):
                pred_vector_coordinates = class_pred_lp[:,class_idx]>=region_hm_thresh
                pred_vector = class_pred_lp[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector,np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_hm, method = 'absolute_max')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = class_pred_lp[pred_vector_coordinates, class_idx]
        return region_coordinates, bb_wh_list, region_pred

#Function that segments regions over an image using a CSN model
def CSN_region_seg(img_rgb, model, class_dict, semantic_net = [], feat_mean = [], method = 'sld_win_CSN', win_size = (256, 256), min_frames_region = 32, overlap_factor = 0.5, thresh = 'mean_half', \
    multi_res_win_size = (1480, 1480), multi_res_name = 'wild lettuce', IOU_multires = .25, IOU_hm = .75, pred_batch = 32, r_neighbours = 0 , region_wl_thresh = 0.5, heatmap_name = 'trebol', region_hm_thresh = .5):
    
    
    #Si se elige la predicción con distancia a las características promedio
    if method == 'sld_win_feat':
        
        #Se cargan las clases y la lista de "n" asociada a ellas
        class_list = class_dict['class_n_list']
        class_name_list = class_dict['class_name_list']
        new_win_shape = model.layers[0].input_shape[1:-1]
        
        #Se carga el número de clase que corresponde a la clase que se revisará por multi-resolución
        class_n_multires = class_name_list.index(multi_res_name)
        
        #Se crea el arreglo de cuadros para la predicción multiresolución y se predicen los valores posibles
        frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array, frame_array_tot = \
        slide_window_creator(img_rgb, win_size = multi_res_win_size, new_win_shape = new_win_shape, overlap_factor = .85, data_type = 'CSN')
        '''
        #Se extraen las características para predecir la clase a través de los cuadros
        _, class_pred , feat_pred = CSN_centroid_pred(frame_array_in, frame_array_tot, model, X_train, Y_train, class_dict, win_size = win_size, thresh = thresh)
        argmin_indexes = np.where(np.argmin(feat_pred, axis = 1)==2)[0]
        
        
        #TODO: desarrollar esto para generar regiones multiescala y no mostrar solo imagenes
        for index in argmin_indexes:
            cv2.imshow( 'Objeto clase: ' + multi_res_name,  frame_array_tot[index, :, :, :] )
            cv2.waitKey(0)
        '''
        
        #Se crea el array de la imagen con la forma especificada
        sld_win_array, frame_coordinates_in, img_rgb_new, frames, frame_array, frame_array_tot = slide_window_creator(img_rgb, win_size = win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor, data_type = 'CSN')
        empty_img = np.zeros(img_rgb_new.shape, dtype = np.uint8)
        
        #Se extraen las características para predecir la clase a través de los cuadros
        #TODO: error acá por la eliminación de X_train y Y_train, revisar como se hace en box_seg
        _, class_pred , feat_pred = CSN_centroid_pred(sld_win_array, frame_array_tot, model, X_train, Y_train, class_dict, win_size = win_size, thresh = thresh)
        #Se predice el valor de cada cuadro y se rellena con la predicción hecha el cuadro correspondiente
        class_list = class_dict['class_n_list']
        class_img = np.zeros( (img_rgb_new.shape[0], img_rgb_new.shape[1], len(class_list)), dtype = np.float32 )
        class_list = class_dict['class_n_list']
        class_name_list = class_dict['class_name_list']
        
        #Para cada predicción se suma el valor a la "imagen" de clases
        for pred_index in range(feat_pred.shape[0]):
            
            frame_coordinate = frame_array[pred_index]
            pred = feat_pred[pred_index,:]
            #pred = np.concatenate ( (feat_pred[pred_index, 0 : class_n_multires ], feat_pred[pred_index, class_n_multires : -1] ), axis = 0)
            
            class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] =\
                class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] + pred
            #for class_n in class_list: class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], class_n ] += pred[class_n]
        
        #TODO: actualizar esto, no está mal
        heat_map = np.ones( ( class_img.shape[0], class_img.shape[1], len(class_list) ), dtype = np.uint8) * 255
    
        for class_n in range(len(class_list)): heat_map[:,:,class_n] = heat_map[:,:,class_n] - class_img[:,:,class_n] / np.max(class_img[:,:,class_n]) * 255
        
        #Se crea una máscara que corresponde a la distancia mínima que se encuentra en cada pixel y se multiplica por la imagen con tamaño ajustado
        class_mask = ( np.argmin(class_img, axis = 2).astype(np.uint8)) 
        class_coord_list = []
        
        #Se crea un diccionario para cada clase conteniendo las coordenadas que se asignaron a cada etiqueta
        for class_n in class_list:
            #Se crea el arreglo de coordenadas para cada clase, se adjunta a una lista para guardarlo en un diccionario con las etiquetas
            class_n_coord = np.concatenate( ( np.reshape(np.where(class_mask == class_n) [0],(  np.where(class_mask == class_n) [0].shape[0], 1 ) ) , \
                np.reshape(np.where(class_mask == class_n) [1],( np.where(class_mask == class_n) [1].shape[0], 1 ) ) ), axis = 0 )
            class_coord_list.append(class_n_coord)
            
        #Se crean imágenes que muestran solamente las clases identificadas
        class_mask_new = np.zeros(class_mask.shape, dtype = np.uint8)
        seg_img = np.zeros(img_rgb_new.shape, dtype = np.uint8)
        '''
        for class_n in class_list:
            
            class_mask_new[class_mask == int(class_n)] = 1
            class_mask_new[class_mask != int(class_n)] = 0
            
            seg_img[:,:,0:3] = np.multiply( img_rgb_new[:,:,0:3], np.concatenate( ( np.reshape( class_mask_new, (class_mask_new.shape[0], class_mask_new.shape[1], 1)),\
                np.reshape( class_mask_new, (class_mask_new.shape[0], class_mask_new.shape[1], 1)), np.reshape( class_mask_new, (class_mask_new.shape[0], class_mask_new.shape[1], 1))), axis = 2 ) )
            #seg_img[:,:,1] = np.multiply(img_rgb_new[:,:,1], class_mask_new)
            #seg_img[:,:,2] = np.multiply(img_rgb_new[:,:,2], class_mask_new)
            cv2.imshow('Imagen segmentada para la clase: ' + str(class_name_list[class_n]), cv2.resize( seg_img, (960,960)))
            cv2.waitKey(0)
        '''
        class_coord_dict = {'class_coord_list': class_coord_list, 'class_name_list': class_name_list, 'class_n_list': class_list, 'weed_class_list': class_dict['weed_class_list'], 'heat_map': heat_map}
        cv2.destroyAllWindows()
        return class_mask, class_coord_dict, img_rgb_new
    
    #Si se elige el reconocimiento de clases dentro de bounding boxes
    elif method == 'box_region':
        
        tic = time.time()
        #Del modelo ingresado se extraen todas las capas menos el cálculo de distancia (si no se ingresó la forma "semántica")
        if not semantic_net:
            semantic_net = Sequential()
            #print(semantic_net.summary())
            for layer in model.layers[0:-1]: semantic_net.add(layer)
            #Se extrae el modelo sin la capa de salida
            sem_model = Sequential()
            feat_model = Sequential()
            for _, layer in enumerate(model.layers):
                if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                    for in_layer in layer.layers:
                        if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                            sem_model.add(in_layer)
                            feat_model.add(in_layer)
                        elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten) or isinstance(in_layer, GlobalAveragePooling2D): feat_model.add(in_layer)
                elif isinstance(layer, GlobalAveragePooling2D) or isinstance(layer, Dense):
                    feat_model.add(layer)
        else: feat_model = semantic_net
        toc =  time.time() - tic
        new_win_shape = feat_model.layers[0].input_shape[1:-1]
        class_list = class_dict['class_n_list']  
        class_n_multires = class_dict['class_name_list'].index(multi_res_name)
        #Si se entrega solamente un tamaño de ventana multiresolución
        frame_size_array = []
        if type(multi_res_win_size) == tuple:
            #Se crea el arreglo de cuadros para la predicción multiresolución y se predicen los valores posibles
            frame_array_in, frame_coordinates_in, _, [frames_r, frames_c], frame_array = \
            slide_window_creator(img_rgb, win_size = multi_res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor, data_type = 'CNN')
            #class_pred = model.predict(frame_array_in, batch_size = pred_batch)
            semantic_X = np.repeat(feat_model.predict(frame_array_in, batch_size = pred_batch)[:,np.newaxis,:], len(class_list), axis = 1)
            d_mat =  np.sqrt( np.sum(np.square(semantic_X-feat_mean), axis = 2) )
            CSN_pred = 1-d_mat
            class_pred = np.argmax(CSN_pred, axis = 1)
            class_n_multires = class_dict['class_name_list'].index(multi_res_name)
            #TODO: prueba de matriz de predicciones
            class_mat = np.reshape(CSN_pred, (frames_r, frames_c, len(class_list)))
            class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
            class_pred_lp_flat = np.reshape(class_pred_lp, (CSN_pred.shape[0], CSN_pred.shape[1]))
            class_pred_lp = class_pred_lp_flat
            #Se lleva a cabo un pasabajos sobre las predicciones
            #class_pred_lp = pred_lowpass(class_pred, neighbours = 2)
            for _ in range(class_pred.shape[0]) : frame_size_array.append(multi_res_win_size[0])
        #Si se entrega una lista de tamaños de ventana
        elif type(multi_res_win_size) == list:
            #Se crea el arreglo de cuadros para cada ventana multiresolución y se predicen los valores posibles
            new_frame_coordinates_in  = np.zeros(0)
            frame_size_array = []
            new_frame_array_in = np.zeros(0)
            for win_size_idx, res_win_size in enumerate(multi_res_win_size):
                frame_array_in, frame_coordinates_in, _, [frames_r, frames_c], frame_array = \
                slide_window_creator(img_rgb, win_size = res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor , data_type = 'CNN')
                
                #Se la cantidad de vecinos para el pasabajos es 0, no es necesario hacer nada más que concatenar los datos y predecir más tarde
                if r_neighbours == 0:
                    if new_frame_array_in.shape[0] == 0:
                        new_frame_array_in = frame_array_in
                        new_frame_coordinates_in = frame_coordinates_in
                    else:
                        new_frame_array_in = np.concatenate( (new_frame_array_in, frame_array_in) )
                        new_frame_coordinates_in = np.concatenate( ( new_frame_coordinates_in, frame_coordinates_in ) )
                        
                    #Si es la última iteración se calculan los valores de predicción para todo el vector
                    if multi_res_win_size.index(res_win_size) == len(multi_res_win_size)-1:
                        for _ in range(frame_array_in.shape[0]) : frame_size_array.append(res_win_size[0])
                        frame_array_in = new_frame_array_in 
                        frame_coordinates_in = new_frame_coordinates_in
                        semantic_X = np.repeat(feat_model.predict(frame_array_in, batch_size = pred_batch)[:,np.newaxis,:], len(class_list), axis = 1)
                        print(semantic_X.shape)
                        d_mat =  np.sqrt( np.sum(np.square(semantic_X-feat_mean), axis = 2) )
                        new_CSN_pred = 1-d_mat
                        new_class_pred = np.copy(new_CSN_pred)
                
                    else:
                        for _ in range(frame_array_in.shape[0]) : frame_size_array.append(res_win_size[0])
                    
                #Si se elige un vecino o más es necesario realizar la operación de pasabajos para cada cuadro
                else:
                #class_pred = model.predict(frame_array_in, batch_size = pred_batch)
                    semantic_X = np.repeat(feat_model.predict(frame_array_in, batch_size = pred_batch)[:,np.newaxis,:], len(class_list), axis = 1)
                    d_mat =  np.sqrt( np.sum(np.square(semantic_X-feat_mean), axis = 2) )
                    CSN_pred = 1-d_mat
                    class_pred = np.argmax(CSN_pred, axis = 1)
                    #TODO: prueba de matriz de predicciones
                    class_mat = np.reshape(CSN_pred, (frames_r, frames_c, len(class_list)))
                    class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
                    class_pred_lp_flat = np.reshape(class_pred_lp, (CSN_pred.shape[0], CSN_pred.shape[1]))
                    class_pred_lp = class_pred_lp_flat
                    #Si ya se había empezado el proceso se agregan los datos al vector de predicciones y coordenadas. Si no, se crean desde cero
                    if new_frame_coordinates_in.shape[0] == 0:
                        new_frame_coordinates_in = frame_coordinates_in
                        new_class_pred = class_pred_lp
                        new_CSN_pred = CSN_pred
                        new_frame_array_in = frame_array_in
                    else:
                        new_frame_coordinates_in = np.concatenate( ( new_frame_coordinates_in, frame_coordinates_in ) )
                        new_class_pred = np.concatenate( ( new_class_pred, class_pred_lp ) )
                        new_CSN_pred = np.concatenate( (new_CSN_pred, CSN_pred))
                        new_frame_array_in = np.concatenate( (new_frame_array_in, frame_array_in) )
                    for _ in range(class_pred.shape[0]) : frame_size_array.append(res_win_size[0])
            
            #Se actualiza el valor del vector de predicción y de coordenadas para más adelante
            class_pred_lp = new_class_pred
            frame_coordinates_in = new_frame_coordinates_in
            CSN_pred = new_CSN_pred
            frame_array_in = new_frame_array_in 
        #Si se entrega un formato no válido
        else: raise ValueError('The only formats admitted for multi-resolution windows are tuple and list')
        
        #Para cada clase se realiza una segmentación por bounding box + NMS
        region_coordinates = []
        bb_wh_list = []
        region_pred = [ [] for _ in range(len(class_list)) ]
        
        region_coordinates = [ [] for _ in range(len(class_list)) ]
        region_pred = [ [] for _ in range(len(class_list)) ]
        bb_wh_list =  [ [] for _ in range(len(class_list)) ]
        
        multires_coord = np.where(CSN_pred[:,class_n_multires]> region_wl_thresh )[0]
        multires_vector = CSN_pred[multires_coord, class_n_multires]
        multires_points = np.zeros(0)
        '''
        print('borrar las siguientes lineas')
        img_for_paper = np.copy(img_rgb)
        color_array = np.uint8(256*np.random.random((multires_coord.shape[0],3)))
        '''
        for frame_num_idx, coord_idx in enumerate(multires_coord):
            if multires_points.shape[0] == 0: multires_points = np.reshape( np.array([frame_coordinates_in[coord_idx,0]*.5+frame_coordinates_in[coord_idx,1]*.5 ,\
                frame_coordinates_in[coord_idx,2]*.5+frame_coordinates_in[coord_idx,3]*.5 ]), (1, 2))
            else: multires_points = np.concatenate( (multires_points,  np.reshape( np.array([frame_coordinates_in[coord_idx,0]*.5+frame_coordinates_in[coord_idx,1]*.5 ,\
                frame_coordinates_in[coord_idx,2]*.5+frame_coordinates_in[coord_idx,3]*.5 ]), (1, 2))), axis = 0 )
        '''
            print('borrar las siguientes lineas')
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,0]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,3], 0] = color_array[frame_num_idx,0]
            img_for_paper[frame_coordinates_in[coord_idx,1]:frame_coordinates_in[coord_idx,1]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,3], 0] = color_array[frame_num_idx,0]
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,1]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,2]+30, 0] = color_array[frame_num_idx,0]
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,1], \
                frame_coordinates_in[coord_idx,3]:frame_coordinates_in[coord_idx,3]+30, 0] = color_array[frame_num_idx,0]
            
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,0]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,3], 1] = 0
            img_for_paper[frame_coordinates_in[coord_idx,1]:frame_coordinates_in[coord_idx,1]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,3], 1] = 0
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,1]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,2]+30, 1] = 0
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,1], \
                frame_coordinates_in[coord_idx,3]:frame_coordinates_in[coord_idx,3]+30, 1] = 0
            
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,0]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,3], 2] = color_array[frame_num_idx,2]
            img_for_paper[frame_coordinates_in[coord_idx,1]:frame_coordinates_in[coord_idx,1]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,3], 2] = color_array[frame_num_idx,2]
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,1]+30, \
                frame_coordinates_in[coord_idx,2]:frame_coordinates_in[coord_idx,2]+30, 2] = color_array[frame_num_idx,2]
            img_for_paper[frame_coordinates_in[coord_idx,0]:frame_coordinates_in[coord_idx,1], \
                frame_coordinates_in[coord_idx,3]:frame_coordinates_in[coord_idx,3]+30, 2] = color_array[frame_num_idx,2]
            
            cv2.imshow('img_for_paper', cv2.resize(img_for_paper, None, fx = 0.1, fy = 0.1))
            cv2.waitKey(0)
        cv2.imwrite(os.getcwd() + '/multires_img/neuva/' + str(coord_idx) + '.jpg', img_for_paper)
        print(os.getcwd() + '/multires_img/neuva/' + str(coord_idx) + '.jpg')
        print("#############################")
        '''
            
        #bb_center, bb_wh,bb_score  =  NMS_bb(multires_points, multires_vector, frame_size_array, IOU = IOU_multires, method = 'average_box')
        
        #for bb_index in range(bb_center.shape[0]):
        #    img_rgb_new[int(bb_center[bb_index,0]-bb_wh[bb_index,0]*.5):int(bb_center[bb_index,0]+bb_wh[bb_index,0]*.5),\
        #        int(bb_center[bb_index,1]-bb_wh[bb_index,1]*.5):int(bb_center[bb_index,1]+bb_wh[bb_index,1]*.5), 1] = int( 255*(1-bb_score[bb_index]) )
            
        for class_idx in range(CSN_pred.shape[1]):
            
            #Se calculan los bounding box y se agregan al índice correspondiente a la clase que corresponden.
            if class_idx == class_dict['class_name_list'].index(multi_res_name):
                
                pred_vector_coordinates = CSN_pred[:,class_idx]>=region_wl_thresh#np.where(CSN_pred[:,class_idx]>.5)[0]#np.where( ( (class_pred == class_idx) * ( CSN_pred[:, class_idx] > 0.5) ) == 1 )[0] 
                pred_vector = CSN_pred[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                    frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector, np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_multires, method = 'average_box')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = CSN_pred[pred_vector_coordinates, class_idx]
            elif class_idx == class_dict['class_name_list'].index(heatmap_name):
                pred_vector_coordinates = CSN_pred[:,class_idx]>=region_hm_thresh#np.where(CSN_pred[:,class_idx]>.5)[0]#np.where( ( (class_pred == class_idx) * ( CSN_pred[:, class_idx] > 0.5) ) == 1 )[0] 
                pred_vector = CSN_pred[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector,np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_hm, method = 'absolute_max')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = CSN_pred[pred_vector_coordinates, class_idx]
            #TODO: probablemente basta solo con borrar esto
            '''    
            else:
                pred_vector_coordinates = CSN_pred[:,class_idx]>.5#np.where(CSN_pred[:,class_idx]>.5)[0]#np.where( ( (class_pred == class_idx) * ( CSN_pred[:, class_idx] > 0.5) ) == 1 )[0] 
                pred_vector = CSN_pred[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector,np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_hm, method = 'absolute_max')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = CSN_pred[pred_vector_coordinates, class_idx]
            '''
        return region_coordinates, bb_wh_list, region_pred
    
#Función para clasificar objetos dentro de una imagen con redes neuronales convolucionales no siamesas
def CNN_region_seg(img_rgb, model, class_dict, win_size = (256, 256),overlap_factor_heatmap = .75, overlap_factor_multires = 0.5,\
    multi_res_win_size = (1440, 1440), multi_res_name = 'wild lettuce', method = 'box_region', IOU_multires = .25, IOU_hm = .75, pred_batch = 32, r_neighbours = 0,\
        region_wl_thresh = 0.5, heatmap_name = 'trebol', region_hm_thresh = .5):
    
    #Se cargan el listado de clases y el tamaño de imagen que entrará a la red
    class_list = class_dict['class_n_list']
    class_name_list = class_dict['class_name_list']
    new_win_shape = model.layers[0].input_shape[1:-1]
    
    #Si se decide segmentar por bounding boxes
    if method == 'box_region':
        #Si se entrega solamente un tamaño de ventana multiresolución
        frame_size_array = []
        if type(multi_res_win_size) == tuple:
            #Se crea el arreglo de cuadros para la predicción multiresolución y se predicen los valores posibles
            frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
            slide_window_creator(img_rgb, win_size = multi_res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_multires, data_type = 'CNN')
            class_pred = model.predict(frame_array_in, batch_size = pred_batch)
                        
            class_mat = np.reshape(class_pred, (frames_r, frames_c, len(class_list)))
            class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
            class_pred_lp_flat = np.reshape(class_pred_lp, (class_pred.shape[0], class_pred.shape[1]))
            class_pred_lp = class_pred_lp_flat
            #Se lleva a cabo un pasabajos sobre las predicciones
            #class_pred_lp = pred_lowpass(class_pred, neighbours = 2)
            for _ in range(class_pred.shape[0]) : frame_size_array.append(multi_res_win_size[0])
        #Si se entrega una lista de tamaños de ventana
        elif type(multi_res_win_size) == list:
            #Se crea el arreglo de cuadros para cada ventana multiresolución y se predicen los valores posibles
            new_frame_coordinates_in  = np.zeros(0)
            frame_size_array = []
            for res_win_size in multi_res_win_size:
                frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
                slide_window_creator(img_rgb, win_size = res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_multires, data_type = 'CNN')
                class_pred = model.predict(frame_array_in, batch_size = pred_batch)
                class_mat = np.reshape(class_pred, (frames_r, frames_c, len(class_list)))
                class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
                class_pred_lp_flat = np.reshape(class_pred_lp, (class_pred.shape[0], class_pred.shape[1]))
                class_pred_lp = class_pred_lp_flat
                #Si ya se había empezado el proceso se agregan los datos al vector de predicciones y coordenadas. Si no, se crean desde cero
                if new_frame_coordinates_in.shape[0] == 0:
                    new_frame_coordinates_in = frame_coordinates_in
                    new_class_pred = class_pred_lp
                else:
                    new_frame_coordinates_in = np.concatenate( ( new_frame_coordinates_in, frame_coordinates_in ) )
                    new_class_pred = np.concatenate( ( new_class_pred, class_pred_lp ) )
                for _ in range(class_pred.shape[0]) : frame_size_array.append(res_win_size[0])
            #Se actualiza el valor del vector de predicción y de coordenadas para más adelante
            class_pred_lp = new_class_pred
            frame_coordinates_in = new_frame_coordinates_in
                 
        #Si se entrega un formato no válido
        else: raise ValueError('The only formats admitted for multi-resolution windows are tuple and list')
        
        #Para cada clase se realiza una segmentación por bounding box + NMS
        region_coordinates = []
        bb_wh_list = []
        region_pred = [ [] for _ in range(len(class_list)) ]
        
        region_coordinates = [ [] for _ in range(len(class_list)) ]
        region_pred = [ [] for _ in range(len(class_list)) ]
        bb_wh_list =  [ [] for _ in range(len(class_list)) ]
        class_n_multires = class_dict['class_name_list'].index(multi_res_name)
        
        multires_coord = np.where(class_pred_lp[:,class_n_multires]> region_wl_thresh )[0]
        multires_vector = class_pred_lp[multires_coord, class_n_multires]
        multires_points = np.zeros(0)
        for coord_idx in multires_coord:
            if multires_points.shape[0] == 0: multires_points = np.reshape( np.array([frame_coordinates_in[coord_idx,0]*.5+frame_coordinates_in[coord_idx,1]*.5 ,\
                frame_coordinates_in[coord_idx,2]*.5+frame_coordinates_in[coord_idx,3]*.5 ]), (1, 2))
            else: multires_points = np.concatenate( (multires_points,  np.reshape( np.array([frame_coordinates_in[coord_idx,0]*.5+frame_coordinates_in[coord_idx,1]*.5 ,\
                frame_coordinates_in[coord_idx,2]*.5+frame_coordinates_in[coord_idx,3]*.5 ]), (1, 2))), axis = 0 )
        
            #cv2.imwrite(os.getcwd() + '/multires/' + str(coord_idx) + '.jpg', frame_array_in[coord_idx])
            
        #bb_center, bb_wh,bb_score  =  NMS_bb(multires_points, multires_vector, frame_size_array, IOU = IOU_multires, method = 'average_box')
        
        #for bb_index in range(bb_center.shape[0]):
        #    img_rgb_new[int(bb_center[bb_index,0]-bb_wh[bb_index,0]*.5):int(bb_center[bb_index,0]+bb_wh[bb_index,0]*.5),\
        #        int(bb_center[bb_index,1]-bb_wh[bb_index,1]*.5):int(bb_center[bb_index,1]+bb_wh[bb_index,1]*.5), 1] = int( 255*(1-bb_score[bb_index]) )
            
        for class_idx in range(class_pred_lp.shape[1]):
            
            #Se calculan los bounding box y se agregan al índice correspondiente a la clase que corresponden.
            if class_idx == class_dict['class_name_list'].index(multi_res_name):
                
                pred_vector_coordinates = class_pred_lp[:,class_idx]>=region_wl_thresh#np.where(CSN_pred[:,class_idx]>.5)[0]#np.where( ( (class_pred == class_idx) * ( CSN_pred[:, class_idx] > 0.5) ) == 1 )[0] 
                pred_vector = class_pred_lp[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                    frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector, np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_multires, method = 'average_box')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = class_pred_lp[pred_vector_coordinates, class_idx]
            elif class_idx == class_dict['class_name_list'].index(heatmap_name):
                pred_vector_coordinates = class_pred_lp[:,class_idx]>=region_hm_thresh#np.where(CSN_pred[:,class_idx]>.5)[0]#np.where( ( (class_pred == class_idx) * ( CSN_pred[:, class_idx] > 0.5) ) == 1 )[0] 
                pred_vector = class_pred_lp[pred_vector_coordinates ,class_idx]
                class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
                frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
                
                bounding_box_center, bounding_box_wh, bb_score = NMS_bb(class_points, pred_vector,np.array(frame_size_array)[pred_vector_coordinates], IOU = IOU_hm, method = 'absolute_max')
                region_coordinates[class_idx] = bounding_box_center
                bb_wh_list[class_idx]= bounding_box_wh
                region_pred[class_idx] = class_pred_lp[pred_vector_coordinates, class_idx]
           
        return region_coordinates, bb_wh_list, region_pred

        #Se retornan las coordenadas calculadas para regiones
        return region_coordinates, bb_wh_list, region_pred
    
    #Si se elige primero usar una ventana para encontrar wild lettuce
    elif method == 'multires':
                
        #Se crea el array de la imagen con la forma especificada para las clases que se elijan segmentar con sliding window
        frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
        slide_window_creator(img_rgb, win_size = win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_heatmap, data_type = 'CNN')
        
        #Se predice el valor de cada cuadro y se rellena con la predicción hecha el cuadro correspondiente
        class_pred = model.predict(frame_array_in, batch_size = pred_batch)
        class_img = np.zeros( (img_rgb_new.shape[0], img_rgb_new.shape[1], len(class_list) ), dtype = np.float32 )
        
        #Para cada predicción se suma el valor a la "imagen" de clases
        for pred_index in range(class_pred.shape[0]):

            frame_coordinate = frame_coordinates_in[pred_index]
            pred = class_pred[pred_index,:]
            #time.sleep(0.1)
            class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] =\
                class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] + pred
            
            #for class_n in class_list: class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], class_n ] += pred[class_n]
        
        #TODO: Se debe normalizar en las esquinas, ver relación matemática con overlap_factor
        #Se crea un mapa de calor con las predicciones de la red
        heat_map = np.zeros( ( class_img.shape[0], class_img.shape[1], len(class_list)), dtype = np.uint8)
        
        for class_n in range(len(class_list)): heat_map[:,:,class_n] = class_img[:,:,class_n] / np.max(class_img[:,:,class_n]) * 255
        
        ##MULTIRES##
        #Se crea el array de la imagen con la forma especificada para la clase multires
        frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
        slide_window_creator(img_rgb, win_size = multi_res_win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_multires, data_type = 'CNN')
        
        #Se predice el valor de cada cuadro multires
        class_pred = model.predict(frame_array_in, batch_size = pred_batch)
        class_points = np.zeros((0))
        pred_vector = np.zeros((0))
        
        #Se encuentran las coordenadas donde la clase correspondiente predomina y se encuentran las bounding boxes por medio de NMS y eliminación dentro de un borde
        multi_res_class = class_name_list.index(multi_res_name)
        pred_vector_coordinates = np.where( class_pred[:, multi_res_class] >= 0.5 )[0]
        pred_vector = class_pred[pred_vector_coordinates,multi_res_class]
        class_points = np.transpose(np.reshape(np.array([[frame_coordinates_in[pred_vector_coordinates,1]*0.5 + frame_coordinates_in[pred_vector_coordinates,0]*0.5,\
            frame_coordinates_in[pred_vector_coordinates,3]*0.5 + frame_coordinates_in[pred_vector_coordinates,2]*0.5]]).astype(int), (2, pred_vector.shape[0])))
        IOU = 0.05
        if pred_vector > 0:
            bounding_box_center, bounding_box_wh = NMS_bb(class_points, pred_vector, multi_res_win_size[0], IOU = IOU)
            bb_index_list = []
            
            #Se eliminan los puntos encontrados fuera del límite imagen-IOU externa
            for bb_index in range(bounding_box_center.shape[0]):
                if (bounding_box_center[bb_index, 0] <= img_rgb.shape[0] +  multi_res_win_size[0] * multi_res_win_size[0]/8) \
                    and (bounding_box_center[bb_index, 0] <= img_rgb.shape[1] +  multi_res_win_size[1] * multi_res_win_size[1]/8 ): 
                    bb_index_list.append(bb_index)
            bounding_box_center = bounding_box_center[bb_index_list]
        
        else: 
            bounding_box_center = np.array([[0,0]])
            bounding_box_wh = np.array([[0,0]])
        
        class_list = class_dict['class_n_list']
        class_name_list = class_dict['class_name_list']
        new_win_shape = model.layers[0].input_shape[1:-1]
        
        '''
        cv2.imshow('Mapa de calor pasto', cv2.resize( heat_map[:,:,0] , None, fx = 0.25, fy = 0.25) )
        cv2.imshow('Mapa de calor trebol', cv2.resize( heat_map[:,:,1] , None, fx = 0.25, fy = 0.25) )
        cv2.imshow('Mapa de calor wild lettuce', cv2.resize( heat_map[:,:,2] , None, fx = 0.25, fy = 0.25) )
        cv2.waitKey(0)
        '''
        #Se crea una máscara que corresponde a la clase máxima que se encuentra en cada pixel y se multiplica por la imagen con tamaño ajustado
        class_mask = ( np.argmax(class_img, axis = 2).astype(np.uint8)) 
        class_coord_list = []
        
        #TODO: En el futuro pensar en cambiar esto por creación de cuadros restringiendo las coordenadas multires
        #Se corrigen el mapa de calor y de clase utilizando los valores obtenidos por la etapa multiresolución
        for bb_idx in range(bounding_box_center.shape[0]):
            heat_map[bounding_box_center[bb_idx,0] - int(bounding_box_wh[bb_idx,0]/2):bounding_box_center[bb_idx,0] + int(bounding_box_wh[bb_idx,0]/2), \
                bounding_box_center[bb_idx,1] - int(bounding_box_wh[bb_idx,1]/2):bounding_box_center[bb_idx,1] + int(bounding_box_wh[bb_idx,1]/2),:] = 0
            
            heat_map[bounding_box_center[bb_idx,0] - int(bounding_box_wh[bb_idx,0]/2):bounding_box_center[bb_idx,0] + int(bounding_box_wh[bb_idx,0]/2), \
                bounding_box_center[bb_idx,1] - int(bounding_box_wh[bb_idx,1]/2):bounding_box_center[bb_idx,1] + int(bounding_box_wh[bb_idx,1]/2),class_name_list.index(multi_res_name)] = 255
            
            class_mask[bounding_box_center[bb_idx,0] - int(bounding_box_wh[bb_idx,0]/2):bounding_box_center[bb_idx,0] + int(bounding_box_wh[bb_idx,0]/2), \
                bounding_box_center[bb_idx,1] - int(bounding_box_wh[bb_idx,1]/2):bounding_box_center[bb_idx,1] + int(bounding_box_wh[bb_idx,1]/2)] = class_name_list.index(multi_res_name)
    
        #Se crea un diccionario para cada clase conteniendo las coordenadas que se asignaron a cada etiqueta
        '''
        for class_n in class_list:
            #Se crea el arreglo de coordenadas para cada clase, se adjunta a una lista para guardarlo en un diccionario con las etiquetas
            class_n_coord = np.concatenate( ( np.reshape(np.where(class_mask == class_n) [0],(  np.where(class_mask == class_n) [0].shape[0], 1 ) ) , \
                np.reshape(np.where(class_mask == class_n) [1],( np.where(class_mask == class_n) [1].shape[0], 1 ) ) ), axis = 1 )
            class_coord_list.append(class_n_coord)
            print('DBS')
            print(class_n_coord.shape)
        '''
        #clustering = DBSCAN(eps = win_size[0], min_samples = 4).fit(class_n_coord[0:1000, :])
        #print(clustering.labels_)
        
        #Se crean imágenes que muestran solamente las clases identificadas
        class_mask_new = np.zeros(class_mask.shape, dtype = np.uint8)
        seg_img = np.zeros(img_rgb_new.shape, dtype = np.uint8)
    
        '''
        for class_n in class_list:
            
            class_mask_new[class_mask == int(class_n)] = 1
            class_mask_new[class_mask != int(class_n)] = 0
            
            seg_img[:,:,0:3] = np.multiply( img_rgb_new[:,:,0:3], np.concatenate( ( np.reshape( class_mask_new, (class_mask_new.shape[0], class_mask_new.shape[1], 1)),\
                np.reshape( class_mask_new, (class_mask_new.shape[0], class_mask_new.shape[1], 1)), np.reshape( class_mask_new, (class_mask_new.shape[0], class_mask_new.shape[1], 1))), axis = 2 ) )
            #seg_img[:,:,1] = np.multiply(img_rgb_new[:,:,1], class_mask_new)
            #seg_img[:,:,2] = np.multiply(img_rgb_new[:,:,2], class_mask_new)
            win_show = 'Imagen segmentada para la clase: ' + str(class_name_list[class_n])
            cv2.namedWindow(win_show, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(win_show, 0, 0)
            cv2.imshow(win_show, cv2.resize( seg_img, (960,960)))
            cv2.waitKey(0)
        '''
        class_coord_dict = {'class_coord_list': class_coord_list, 'class_name_list': class_name_list, 'class_n_list': class_list, 'weed_class_list': class_dict['weed_class_list'], 'heat_map': heat_map}
        cv2.destroyAllWindows()
        frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array = \
        slide_window_creator(img_rgb, win_size = win_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_heatmap, data_type = 'CNN')
        return class_mask, class_coord_dict, img_rgb_new

#Función que ejecuta el pipeline principal del algoritmo de detección de maleza
def pipeline(img, first_model, second_model ,class_dict_reg, class_dict_sld, feat_model_first = [], feat_model_second = [], frame_size = (256, 256), feat_mean_first = [], feat_mean_second = [],\
    overlap_factor_first = 0.5, overlap_factor_second = 0.75, multi_res_win_size = (1280, 1280), multi_res_name = 'wild lettuce', IOU_multires = .25, IOU_hm = .75, heat_map_class = 'trebol',\
        heat_map_display = True, bg_class = 'pasto', class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'], savedir = '', overwrite = False, pred_batch = [32, 32], r_neighbours = 0,\
            seg_dict = {}, imsave = True, zero_pad = False,hm_thresh = .5 , region_wl_thresh = 0.5, region_hm_thresh = 0.5, selection_dict_reg = '', feats_param_dict_reg = '',\
                selection_dict_sld = '', feats_param_dict_sld = ''):
    
    #Se empieza a contar el tiempo que toma el algoritmo
    total_tic = time.time()
    #Se verifica que la imagen no sea un NONE y tenga el tamaño apropiado
    if type(img) == None: raise ValueError('Ingrese una imagen válida')
    else:
        if len(img.shape) != 3: raise ValueError('Ingrese una imagen válida')
    
    #Se lee el número de clases, el nuevo tamaño de entrada para las imágenes y se ejecutan las predicciones en base al modelo entregado
    class_list = class_dict_reg['class_n_list']
    class_name_list = class_dict_reg['class_name_list']
    
    #Se procede a realizar la segmentación pedida    
    region_tic = time.time()
    if method == 'region_sld': 
        if model_type[0] == 'CNN':
            region_coordinates, bb_wh_list, region_pred= \
            CNN_region_seg(img, first_model, class_dict_reg, win_size = frame_size, overlap_factor_multires = overlap_factor_first,\
                multi_res_win_size = multi_res_win_size, multi_res_name = multi_res_name, method = 'box_region', IOU_multires = IOU_multires, IOU_hm = IOU_hm, pred_batch = pred_batch[0], \
                    r_neighbours = r_neighbours, region_wl_thresh = region_wl_thresh, heatmap_name = heat_map_class, region_hm_thresh = region_hm_thresh )
        elif model_type[0] == 'CSN':
            region_coordinates, bb_wh_list, region_pred= \
                CSN_region_seg(img, first_model, class_dict_reg, semantic_net = feat_model_first, feat_mean =feat_mean_first, method = 'box_region', win_size = frame_size, min_frames_region = 32, overlap_factor = overlap_factor_first,\
                    thresh = 'mean_half',multi_res_win_size = multi_res_win_size, multi_res_name = multi_res_name, IOU_multires = IOU_multires, IOU_hm = IOU_hm, pred_batch = pred_batch[0],\
                        r_neighbours = r_neighbours, region_wl_thresh = region_wl_thresh, heatmap_name = heat_map_class, region_hm_thresh = region_hm_thresh )
        elif model_type[0] == 'SVM':
            region_coordinates, bb_wh_list, region_pred= \
                SVM_region_seg(img, first_model, class_dict_reg, selection_dict_reg, feats_param_dict_reg, multi_res_win_size = multi_res_win_size, multi_res_name = multi_res_name, method = 'box_region',\
                    overlap_factor_multires = overlap_factor_first, IOU_multires = IOU_multires, IOU_hm = IOU_hm, pred_batch = pred_batch[0], r_neighbours = r_neighbours, region_wl_thresh = region_wl_thresh,\
                        heatmap_name = heat_map_class, region_hm_thresh = region_hm_thresh)
        #elif model_type[0] == 'SVM':
            
        ##PARTE 1: Multi-resolución
        #TODO: la ventana se debe definir de otra forma
        #if type(multi_res_win_size) == list: multi_res_win_size = (1800, 1800)
        #Se analizan los cuadros que fueron identificados bajo el nombre "multi_res_name" con el segundo modelo para verificar que hayan sido identificados correctamente
        
        new_win_shape = second_model.layers[0].input_shape[1:-1] if model_type[1] != 'SVM' else selection_dict_sld['img_shape']
        frame_multires_mat = np.zeros( (region_coordinates[class_name_list.index(multi_res_name)].shape[0], new_win_shape[0], new_win_shape[1], 3) , dtype= np.uint8)
        multires_coord = region_coordinates[class_name_list.index(multi_res_name)]
        multires_pred = region_pred[class_name_list.index(multi_res_name)]
        multires_idx = []
        multires_img = np.zeros(img.shape, dtype=np.uint16)
        multires_predimg = np.zeros( ( img.shape[0], img.shape[1] ) )
        wh_multires_list = bb_wh_list[class_name_list.index(multi_res_name)] 
        FP_multires = []
        #Se reconstruye la imagen con las coordenadas predichas como LV
        for coord_idx in range(multires_coord.shape[0]):
            [multi_r, multi_c], multi_wh = multires_coord[coord_idx, :], wh_multires_list[coord_idx, :]
            #Si se eligió rellenar con ceros
            if zero_pad:
                frame_multires = img[ np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]),0]) \
                : multires_coord[coord_idx,0],\
                np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]), 0]) \
                    : multires_coord[coord_idx,1] ,:]
            #Si no
            else:
                frame_multires = img[ np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]/2),0]) \
                    : np.min([multires_coord[coord_idx,0] + int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]/2),img.shape[0]]),\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]/2),0]) \
                    : np.min([multires_coord[coord_idx,1] + int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]/2),img.shape[1]]) ,:]

            '''
            ###IMAGEN MULTIRESOLUCION##########################################################################################
            if zero_pad:
                multires_img[np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]),0]) \
                    : multires_coord[coord_idx,0],\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]), 0]) \
                        : multires_coord[coord_idx,1] ,:] = np.uint16 ( img[ np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]),0]) \
                    : multires_coord[coord_idx,0],\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]), 0]) \
                        : multires_coord[coord_idx,1] ,:] * multires_pred[coord_idx]  )
                multires_predimg[np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]),0]) \
                    : multires_coord[coord_idx,0],\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]), 0]) \
                        : multires_coord[coord_idx,1] ] +=  multires_pred[coord_idx]  
            else:
                multires_img[np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]/2),0]) \
                    : np.min([multires_coord[coord_idx,0] + int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]/2),img.shape[0]]),\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]/2),0]) \
                    : np.min([multires_coord[coord_idx,1] + int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]/2),img.shape[1]]) ,:]\
                        = np.uint16 ( img[ np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]/2),0]) \
                    : np.min([multires_coord[coord_idx,0] + int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]/2),img.shape[0]]),\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]/2), 0]) \
                        : np.min([multires_coord[coord_idx,1] + int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]/2),img.shape[1]]) ,:] * multires_pred[coord_idx]  )
                multires_predimg[np.max([multires_coord[coord_idx,0] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 0]),0]) \
                    : multires_coord[coord_idx,0],\
                    np.max([multires_coord[coord_idx,1] - int(bb_wh_list[class_name_list.index(multi_res_name)][coord_idx, 1]), 0]) \
                        : multires_coord[coord_idx,1] ] +=  multires_pred[coord_idx]  
            '''
            #######################################################################################################################
            #Si el cuadro encontrado es de tamaño menor a un cuarto del cuadro se descarta. En caso contrario, se padea con los 0's necesarios y se agrega a la matriz de cuadros multiresolución
            #Se calcula la clase predominante dentro de cada cuadro de LV para evitar malas detecciones
            
            
            multi_sum = 1#np.sum( ( new_pred[:,  class_name_list.index(multi_res_name) ] > wl_thresh ) ) > np.sum( ( new_pred[:,  class_name_list.index(heat_map_class) ] > hm_thresh ) ) \
                #and np.sum( ( new_pred[:,  class_name_list.index(multi_res_name) ] > wl_thresh ) ) > np.sum( ( new_pred[:,  class_name_list.index(bg_class) ] > 1-wl_thresh ) )
            if (frame_multires.shape[0] > wh_multires_list[coord_idx,0] *.25 or frame_multires.shape[1] > wh_multires_list[coord_idx,1] *.25) and multi_sum:
                multires_idx.append(coord_idx)
                frame_multires_mat[coord_idx, :,:,:] = cv2.resize(frame_multires, (new_win_shape[0], new_win_shape[1]) )
            else: FP_multires.append([ multi_r , multi_c ])

        #multires_img = ( multires_img * ( 255 /( np.max(multires_img ) + 1  ) ) ).astype(np.uint8)
        #multires_predimg = multires_predimg / ( np.max(multires_predimg ) + 1 )
        #multires_predimg[multires_predimg<.5] = 0
        multires_coord = multires_coord[multires_idx]
        wh_multires_list = wh_multires_list[multires_idx]

        region_tictoc = time.time()-region_tic
        ##PARTE 2: Heat-map
        hm_tic = time.time()
        #Se realiza el método de heat map para las imágenes que correspondan a las coordenadas de la clase heat map
        heatmap_coord = region_coordinates[class_name_list.index(heat_map_class)]
        heatmap_pred = region_pred[class_name_list.index(heat_map_class)]
        heatmap_idx = []
        heatmap_img = np.zeros(img.shape, dtype=np.uint8)
        wh_hm_list = bb_wh_list[class_name_list.index(heat_map_class)]
        #Se eliminan los puntos que crearían cuadros dentro del IOU regiones multi-resolución 
        #ones_multires_img = np.zeros(img.shape, dtype=np.uint8)
        #ones_multires_img[multires_img>0] = 1
        [minhm_r, minhm_c, maxhm_r, maxhm_c] = [img.shape[0], img.shape[1], 0, 0]
        
        #PARA MOSTRAR LAS REGIONES HEAT MAP
        heatmap_bool = True
        for coord_idx in range(heatmap_coord.shape[0]):
            #Se eliminan las regiones del mapa de calor que queden dentro del IOU de las regiones multi-resolución
            for multires_idx in range(multires_coord.shape[0]):
                if ( np.max([0, wh_multires_list[multires_idx, 0] - abs( multires_coord[multires_idx,0] - heatmap_coord[coord_idx,0] ) ]) \
                    * np.max([0, wh_multires_list[multires_idx, 1] - abs( multires_coord[multires_idx,1] - heatmap_coord[coord_idx,1] ) ]) ) < 0.25*wh_multires_list[multires_idx, 0]*wh_multires_list[multires_idx,1]:
                    heatmap_bool = True
                else:
                    heatmap_bool = False
                    break
            
            #Si existen regiones fuera de dicho IOU, se procede a rellenar la imagen de mapa de calor
            if heatmap_bool:
                #Si la imagen se rellenó con ceros
                if zero_pad:
                    frame_heatmap = img[ heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]) \
                        : heatmap_coord[coord_idx,0] ,\
                        heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]) \
                            : heatmap_coord[coord_idx,1] ,:]
                    
                    heatmap_img[heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]) \
                    : heatmap_coord[coord_idx,0] ,\
                    heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]) \
                        : heatmap_coord[coord_idx,1] ,:] = img[ heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]) \
                    : heatmap_coord[coord_idx,0],\
                    heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]) \
                        : heatmap_coord[coord_idx,1] ,:]
                #Si no
                else:
                    frame_heatmap = img[ np.max( [ heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]/2), 0]) \
                        : np.min ([ heatmap_coord[coord_idx,0] + int(wh_hm_list[coord_idx, 0]/2), img.shape[0]]),\
                        np.max([heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]/2), 0]) \
                            : np.min([heatmap_coord[coord_idx,1] + int(wh_hm_list[coord_idx, 1]/2), img.shape[1]]) ,:]
                    
                    heatmap_img[np.max( [ heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]/2), 0]) \
                        : np.min ([ heatmap_coord[coord_idx,0] + int(wh_hm_list[coord_idx, 0]/2), img.shape[0]]),\
                        np.max([heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]/2), 0]) \
                            : np.min([heatmap_coord[coord_idx,1] + int(wh_hm_list[coord_idx, 1]/2), img.shape[1]]) ,:] =\
                    img[ np.max( [ heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]/2), 0]) \
                        : np.min ([ heatmap_coord[coord_idx,0] + int(wh_hm_list[coord_idx, 0]/2), img.shape[0]]),\
                        np.max([heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]/2), 0]) \
                            : np.min([heatmap_coord[coord_idx,1] + int(wh_hm_list[coord_idx, 1]/2), img.shape[1]])  ,:]
                #######################################################################################################################
                #Si el cuadro encontrado es de tamaño menor a la mitad del cuadro se descarta. En caso contrario, se padea con los 0's necesarios y se agrega a la matriz de cuadros heatmap
                if frame_heatmap.shape[0] > wh_hm_list[coord_idx, 0] *.01 or frame_heatmap.shape[1] > wh_hm_list[coord_idx, 1] *.01:
                    minhm_r = np.min( [ minhm_r, np.max( [ heatmap_coord[coord_idx,0] - int(wh_hm_list[coord_idx, 0]/2), 0])])
                    minhm_c = np.min( [ minhm_c, np.max( [ heatmap_coord[coord_idx,1] - int(wh_hm_list[coord_idx, 1]/2), 0])])
                    maxhm_r = np.max( [ maxhm_r, np.min( [ heatmap_coord[coord_idx,0] + int(wh_hm_list[coord_idx, 0]/2), img.shape[0]])])
                    maxhm_c = np.max( [ maxhm_c, np.min( [ heatmap_coord[coord_idx,1] + int(wh_hm_list[coord_idx, 1]/2), img.shape[1]])])
                    heatmap_idx.append(coord_idx)
                    
        full_heat_map = np.zeros((img.shape[0], img.shape[1],1))
        heat_map_coord = heatmap_coord[heatmap_idx]
        bounding_box_wh = bb_wh_list[class_name_list.index(heat_map_class)]
    
        class_name_list_sld = class_dict_sld['class_name_list']
                
        #Para la imagen de regiones de trébol se calcula el heat map 
        #nonzero_hm = np.where(heatmap_img!=0)
        #[minhm_r, minhm_c, maxhm_r, maxhm_c] = [np.min(nonzero_hm[0]), np.min(nonzero_hm[1]), np.max(nonzero_hm[0]), np.max(nonzero_hm[1])]\
        #    if nonzero_hm[0].shape[0] != 0 and nonzero_hm[1].shape[0] != 0 else [0, 0, 0, 0]
        frames_r = np.floor( (maxhm_r-minhm_r-frame_size[0]) / ( frame_size[0] * (1-overlap_factor_second) ) + 1  ).astype(int)
        frames_c = np.floor( (maxhm_c-minhm_c-frame_size[1]) / ( frame_size[1] * (1-overlap_factor_second) ) + 1  ).astype(int)
        remainder_r = int(frame_size[0] - (maxhm_r - minhm_r - frames_r*frame_size[0]*(1-overlap_factor_second)))
        remainder_c = int(frame_size[1] - (maxhm_c - minhm_c - frames_c*frame_size[1]*(1-overlap_factor_second)))
        [pad_minhm_r , pad_maxhm_r, pad_minhm_c, pad_maxhm_c] = int(np.ceil(minhm_r-remainder_r/2)), int(np.ceil(maxhm_r+remainder_r/2)), int(np.ceil(minhm_c-remainder_c/2)), int(np.ceil(maxhm_c+remainder_c/2))
        img_rgb = heatmap_img[minhm_r : maxhm_r, minhm_c : maxhm_c,:]
        
        #Si no se encuentra ningún cuadro compatible con heatmap se salta esa etapa
        if img_rgb.shape[0] != 0 and img_rgb.shape[1] != 0:
            #Se crea el array de la imagen con la forma especificada para las clases que se elijan segmentar con sliding windows
            frame_array_in, frame_coordinates_in, _, [frames_r, frames_c], frame_array = \
                slide_window_creator(img_rgb, win_size = frame_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_second, data_type = 'CNN_sld')
            #Se predice el valor de cada cuadro y se rellena con la predicción hecha el cuadro correspondiente
            #Si es con CNN, se extraen simplemente las predicciones
            time_pred = time.time()
            if model_type[1] == 'CNN': class_pred = second_model.predict(frame_array_in, batch_size = pred_batch[1])#second_model.predict(frame_array_in, batch_size = pred_batch[1])
            elif model_type[1] == 'CSN':
                #Del modelo ingresado se extraen todas las capas menos el cálculo de distancia (si no se ingresó la forma "semántica")
                if not feat_model_second:
                    semantic_net = Sequential()
                    #print(semantic_net.summary())
                    for layer in second_model.layers[0:-1]: semantic_net.add(layer)
                    #Se extrae el modelo sin la capa de salida
                    sem_model = Sequential()
                    feat_model = Sequential()
                    for idx, layer in enumerate(second_model.layers):
                        if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                            for in_layer in layer.layers:
                                if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                                    sem_model.add(in_layer)
                                    feat_model.add(in_layer)
                                elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model.add(in_layer)
                else: feat_model = feat_model_second
                
                new_win_shape = sem_model.layers[0].input_shape[1:-1]
                pred_feats = feat_model.predict(frame_array_in, batch_size = pred_batch[1])
                semantic_X = np.repeat(pred_feats[:,np.newaxis,:], len(class_list), axis = 1)
                d_mat =  np.sqrt( np.sum(np.square(semantic_X-feat_mean_second), axis = 2) )
                class_pred = 1-d_mat
            else:
                img_gray = cv2.equalizeHist( cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) )
                frame_array_in, frame_coordinates_in, _, [frames_r, frames_c], frame_array = \
                slide_window_creator(img_gray, win_size = frame_size, new_win_shape = new_win_shape, overlap_factor = overlap_factor_second, data_type = 'CNN_sld')
                #Se calculan las características para el clasificador SVM
                for frame_count, frame in enumerate(frame_array_in):
                    lbp_frame_size, lbp_type, lbp_points, lbp_r = feats_param_dict_sld['lbp_frame_size'], feats_param_dict_sld['lbp_type'], feats_param_dict_sld['lbp_points'], feats_param_dict_sld['lbp_r']
                    pixels_x, pixels_y, block_num_x, block_num_y, orientations_n = feats_param_dict_sld['hog_pixels_x'], feats_param_dict_sld['hog_pixels_y'], feats_param_dict_sld['block_num_x'], feats_param_dict_sld['block_num_y'], feats_param_dict_sld['orientations_n']
                    #Cálculo de características HOG y LBP
                    lbp_total = np.asarray( lbp_feats_calc(frame, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
                    hog_total = np.asarray( hog_feats_calc(frame, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
                    #Si es la primera imagen procesada, se crean los vectores de características y de etiquetas
                    if frame_count == 0: X = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                    else: X = np.concatenate( ( X,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
                X_clean_indices = selection_dict_sld['X_train_clean_indices']
                X = X[:, X_clean_indices]
                class_pred = to_categorical(second_model.predict(X), num_classes = len(class_name_list))
                class_mat = np.reshape(class_pred, (frames_r, frames_c, len(class_list)))
                class_pred_lp = pred_lowpass(class_mat, neighbours = r_neighbours)
                class_pred_lp_flat = np.reshape(class_pred_lp, (class_pred.shape[0], class_pred.shape[1]))
                class_pred_lp = class_pred_lp_flat
                
            class_img = np.zeros( (img_rgb.shape[0]+remainder_r, img_rgb.shape[1]+remainder_c, 1 ), dtype = np.float64 )
            print('time_pred: ' + str(time.time()-time_pred))
            #Para cada predicción se suma el valor a la "imagen" de clases
            #TODO: la normalización seguramente se puede hacer más eficiente
            #TODO: revisar el factor de división, ojalá evitar lo hardcodeado que está
            #TODO: intentar cythonizar
            class_pred = class_pred/4 if overlap_factor_second <=.5 else class_pred/16
            tic_sum = time.time()
            for pred_index in range(class_pred.shape[0]):
                #TODO: hay frame_coordinates iguales a 0, revisar la función de cuadros
                frame_coordinate = frame_coordinates_in[pred_index]
                pred = class_pred[pred_index,class_name_list_sld.index(heat_map_class)]
                #print(minhm_r), print(minhm_c), print(maxhm_r),print(maxhm_c)
                #input(str(frame_coordinate))
                #class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] =\
                #    class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] + pred
                #frame_class_img = class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] + pred
                #if np.max(frame_class_img) > 1: frame_class_img = frame_class_img / np.max(frame_class_img)
                #class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] += pred
                #Se generan las sumas para la división pedida
                class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] += pred 
                #if frame_coordinate[0] + frame_coordinate[2] == (minhm_r + minhm_c): class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] += pred
                #else: class_img[ frame_coordinate[0]:frame_coordinate[1], frame_coordinate[2]:frame_coordinate[3], : ] += pred/16
            tictoc_sum_1 = time.time()-tic_sum
            print('time_sum_1: ' + str(tictoc_sum_1))
            #Se eliminan las zonas donde la imagen original es 0, además de la corrección del factor divisor en las esquinas
            #class_img[(img_rgb[:,:,0] == 0) & (img_rgb[:,:,1] == 0) & (img_rgb[:,:,2] == 0)] = 0
            if overlap_factor_second<=.5:
                class_img[0 : int( frame_size[0]*(1-overlap_factor_second) ), 0: int( frame_size[1]*(1-overlap_factor_second) )] *=4
                class_img[0 : int( frame_size[0]*(1-overlap_factor_second) ), class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *= 4
                class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1 , 0 : int( frame_size[1]*(1-overlap_factor_second) )] *=4
                class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1 , class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *=4
                
                class_img[int( frame_size[0]*(1-overlap_factor_second) ) : class_img.shape[0] -int( frame_size[0]*(1-overlap_factor_second) ), 0: int( frame_size[1]*(1-overlap_factor_second) )] *=2
                class_img[int( frame_size[0]*(1-overlap_factor_second) ) : class_img.shape[0] -int( frame_size[0]*(1-overlap_factor_second) ), class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *=2
                class_img[ 0 : int( frame_size[0]*(1-overlap_factor_second) ), int( frame_size[1]*(1-overlap_factor_second) ) : class_img.shape[1] -int( frame_size[1]*(1-overlap_factor_second) )] *=2
                class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1, int( frame_size[1]*(1-overlap_factor_second) ) : class_img.shape[1] -int( frame_size[1]*(1-overlap_factor_second) ) ] *=2
                
            else:
                class_img[0 : int( frame_size[0]*(1-overlap_factor_second) ), 0: int( frame_size[1]*(1-overlap_factor_second) )] *= 16
                class_img[0 : int( frame_size[0]*(1-overlap_factor_second) ), class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *= 16
                class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1 , 0 : int( frame_size[1]*(1-overlap_factor_second) )] *= 16
                class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1 , class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *= 16
                
                class_img[int( frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    0:int( frame_size[1]*(1-overlap_factor_second) )] *= 8
                class_img[0:int( frame_size[0]*(1-overlap_factor_second) ),\
                    int( frame_size[1]*(1-overlap_factor_second) ):int(2* frame_size[1]*(1-overlap_factor_second) )] *= 8
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    0:int( frame_size[1]*(1-overlap_factor_second) )] *= 8
                class_img[class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ):-1,\
                    int( frame_size[1]*(1-overlap_factor_second) ):int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 8    
                
                class_img[int( frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) ) : -1] *= 8
                class_img[0:int( frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int(2* frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int(frame_size[1]*(1-overlap_factor_second) )] *= 8
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) ):-1] *= 8
                class_img[class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ):-1,\
                    class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) )] *= 8
                
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    0:int( frame_size[1]*(1-overlap_factor_second) )] *= 16/3
                class_img[0:int( frame_size[0]*(1-overlap_factor_second) ),\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int(3* frame_size[1]*(1-overlap_factor_second) )] *= 16/3
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    0:int( frame_size[1]*(1-overlap_factor_second) )] *= 16/3
                class_img[class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ):-1,\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int( 3*frame_size[1]*(1-overlap_factor_second) )] *= 16/3
                
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) ) : -1] *= 16/3
                class_img[0:int( frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int(3* frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int(2*frame_size[1]*(1-overlap_factor_second) )] *= 16/3
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) ):-1] *= 16/3
                class_img[class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ):-1,\
                    class_img.shape[1]-int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 16/3
                
                
                class_img[int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    0:int( frame_size[1]*(1-overlap_factor_second) )] *= 4
                class_img[int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) ):-1] *= 4
                class_img[0:int( frame_size[0]*(1-overlap_factor_second) ),\
                    int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 3*frame_size[0]*(1-overlap_factor_second) )] *= 4
                class_img[class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ):-1,\
                    int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 3*frame_size[0]*(1-overlap_factor_second) )] *= 4
                
                class_img[int( frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    int( frame_size[1]*(1-overlap_factor_second) ):int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 4
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    int( frame_size[1]*(1-overlap_factor_second) ):int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 4
                class_img[int( frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) )] *= 4
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) )] *= 4
                
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    int( frame_size[1]*(1-overlap_factor_second) ):int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                class_img[int(frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int( 3*frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    int( frame_size[1]*(1-overlap_factor_second) ):int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int( 3*frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                class_img[int(frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 8/3
                
                class_img[int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    int( frame_size[1]*(1-overlap_factor_second) ):int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 2
                class_img[int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( frame_size[1]*(1-overlap_factor_second) )] *= 2
                class_img[int( frame_size[0]*(1-overlap_factor_second) ):int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 3*frame_size[0]*(1-overlap_factor_second) )] *= 2
                class_img[class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( frame_size[0]*(1-overlap_factor_second) ),\
                    int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 3*frame_size[0]*(1-overlap_factor_second) )] *= 2
                
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int( 3*frame_size[1]*(1-overlap_factor_second) )] *= 16/9
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int( 3*frame_size[1]*(1-overlap_factor_second) )] *= 16/9
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 16/9
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 16/9
                
                class_img[int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 2*frame_size[1]*(1-overlap_factor_second) ):int( 3*frame_size[1]*(1-overlap_factor_second) )] *= 1.25
                class_img[int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    class_img.shape[1]-int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 2*frame_size[1]*(1-overlap_factor_second) )] *= 1.25
                class_img[int( 2*frame_size[0]*(1-overlap_factor_second) ):int( 3*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 3*frame_size[0]*(1-overlap_factor_second) )] *= 1.25
                class_img[class_img.shape[0]-int( 3*frame_size[0]*(1-overlap_factor_second) ):class_img.shape[0]-int( 2*frame_size[0]*(1-overlap_factor_second) ),\
                    int( 3*frame_size[1]*(1-overlap_factor_second) ):class_img.shape[1]-int( 3*frame_size[0]*(1-overlap_factor_second) )] *= 1.25
                
                #class_img[0 : int( frame_size[0]*(1-overlap_factor_second) ), class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *= 16
                #class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1 , 0 : int( frame_size[1]*(1-overlap_factor_second) )] *= 16
                #class_img[class_img.shape[0] - int( frame_size[0]*(1-overlap_factor_second) ) : -1 , class_img.shape[1] - int( frame_size[1]*(1-overlap_factor_second) ) : -1 ] *= 16
            fill_r, fill_c = int(remainder_r%2 > 0), int(remainder_c%2 > 0)
            class_img = class_img[int(remainder_r/2):-int(remainder_r/2)-fill_r, int(remainder_c/2):-int(remainder_c/2)-fill_c]
            tictoc_sum = time.time()-tic_sum
            if np.max(class_img) >1 : raise ValueError('aaaaa')
            else: print(np.max(class_img))
            print('time_sum: ' + str(tictoc_sum))
            #cv2.imshow(str(hm_thresh) + '+' + str(np.max(class_img)), cv2.resize(class_img, None, fx = .15, fy = .15)), cv2.waitKey(0)
                
            #class_img[ (np.sum(img_rgb, axis = 2) == 0) ] = 0
            class_img[class_img<hm_thresh] = 0
            
            #Se crea un mapa de calor con las predicciones de la red y se suma al mapa global en las coordenadas en las que se construye cada predicción
            full_heat_map[minhm_r : maxhm_r, minhm_c : maxhm_c,:] = class_img[ : , :, :]
                        
        #Se crea un mapa con las predicciones de multi_res y de heat_map, incluyendo las zonas donde no se predijo ningún objeto como clase fondo
        seg_img = np.copy(img)
        
        ##HEAT MAP
        #Por fines demostrativos se resaltan solo las zonas dónde la predicción es mayor 0.5 y se escalan los resultados entre 0.5 y 1
        if imsave == True:
            heat_map_3d = np.repeat(full_heat_map[:, :, :]/(np.max(full_heat_map)+np.spacing(1)), 3, axis = 2)
            seg_mask = seg_img[full_heat_map[:,:,0] >= hm_thresh,1]*.5 - np.multiply( heat_map_3d , img)[full_heat_map[:,:,0] >= hm_thresh,1]*.5
            seg_mask[seg_mask<0] = 0
            seg_img[full_heat_map[:,:,0] >= hm_thresh, 1] = seg_mask.astype(np.uint8)
            hm_img = np.copy(seg_img)
            
        calc_img = np.copy(full_heat_map)
        #calc_img[full_heat_map[:,:,0] <= .5] = 0
        
        '''
        #Si no se encontraron regiones heat_map se rellena con ceros
        else:
            seg_img = np.copy(img)
            seg_mask = np.zeros(seg_img.shape)
            calc_img = np.copy(seg_mask)
        '''  
        hm_tictoc = time.time()-hm_tic
        ##MULTI-RES
        img_tic = time.time()
        #Para todas las coordenadas predichas como clase multi-resolución se dibuja la bounding box correspondiente y se eliminan las falsas detecciones
        bb_wh = bb_wh_list[class_name_list.index(multi_res_name)]
        bb_t = 20
        corrected_multires, FP_multires = [], []
        for multires_idx in range(multires_coord.shape[0]):
            [multi_r, multi_c], multi_wh = multires_coord[multires_idx, :], wh_multires_list[multires_idx, :]
            frame_multires = img[np.max([0, multi_r - int(multi_wh[0]/2)]):np.min([multi_r + int(multi_wh[0]/2), img.shape[0]]), \
                np.max([0, multi_c - int(multi_wh[1]/2)]):np.min([multi_c + int(multi_wh[1]/2), img.shape[1]]),:]
            #Se calcula la clase predominante dentro de cada cuadro de LV para evitar malas detecciones
            multi_sum = 1 #np.sum( ( new_pred[:,  class_name_list.index(multi_res_name) ] > wl_thresh ) ) > np.sum( ( new_pred[:,  class_name_list.index(heat_map_class) ] > hm_thresh ) ) \
                #and np.sum( ( new_pred[:,  class_name_list.index(multi_res_name) ] > wl_thresh ) ) > np.sum( ( new_pred[:,  class_name_list.index(bg_class) ] > 1-wl_thresh ) )

            if imsave and multi_sum:
                seg_img[np.max([0,  (multi_r-int(bb_wh[multires_idx,0]/2))-bb_t ] ) : np.max([0, (multi_r-int(bb_wh[multires_idx,0]/2))+bb_t ] ),\
                    np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2)) ] ): np.min( [multi_c+int(bb_wh[multires_idx,1]/2), seg_img.shape[1] ]),0:2] = 0
                seg_img[np.max([0,  (multi_r-int(bb_wh[multires_idx,0]/2))-bb_t ] ) : np.max([0, (multi_r-int(bb_wh[multires_idx,0]/2))+bb_t ] ),\
                    np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2)) ] ): np.min( [multi_c+int(bb_wh[multires_idx,1]/2), seg_img.shape[1] ]),2] = 255
                seg_img[ np.max([0,  multi_r+int(bb_wh[multires_idx,0]/2)-bb_t] ) : np.min([multi_r+int(bb_wh[multires_idx,1]/2)+bb_t, seg_img.shape[0]]),\
                    np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2)) ]) : np.min([multi_c + int(bb_wh[multires_idx,1]/2), seg_img.shape[1]]),0:2] = 0
                seg_img[ np.max([0,  multi_r+int(bb_wh[multires_idx,0]/2)-bb_t] ) : np.min([multi_r+int(bb_wh[multires_idx,1]/2)+bb_t, seg_img.shape[0]]),\
                    np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2)) ]) : np.min([multi_c + int(bb_wh[multires_idx,1]/2), seg_img.shape[1]]),2] = 255
                
                seg_img[np.max([0, (multi_r-int(bb_wh[multires_idx,0]/2))]) : np.min([(multi_r+int(bb_wh[multires_idx,0]/2)),seg_img.shape[0]]),\
                    np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2))-bb_t]): np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2))+bb_t]),0:2] = 0
                seg_img[np.max([0, (multi_r-int(bb_wh[multires_idx,0]/2))]) : np.min([(multi_r)+int(bb_wh[multires_idx,0]/2),seg_img.shape[0]]),\
                    np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2))-bb_t]): np.max([0, (multi_c-int(bb_wh[multires_idx,1]/2))+bb_t]),2] = 255
                seg_img[ np.max([0, (multi_r-int(bb_wh[multires_idx,0]/2))]): np.min([ (multi_r) + int(bb_wh[multires_idx,0]/2),\
                    seg_img.shape[0]]), (multi_c)+int(bb_wh[multires_idx,1]/2)-bb_t:(multi_c)+int(bb_wh[multires_idx,1]/2)+bb_t,0:2] = 0
                seg_img[(multi_r-int(bb_wh[multires_idx,0]/2)):(multi_r)+int(bb_wh[multires_idx,0]/2), (multi_c)+int(bb_wh[multires_idx,1]/2)-bb_t:(multi_c)+int(bb_wh[multires_idx,1]/2)+bb_t,2] = 255
            
            
            if multi_sum: corrected_multires.append([ multi_r -int(bb_wh[multires_idx,0]/2), multi_c - int(bb_wh[multires_idx,1]/2) ]) if zero_pad else corrected_multires.append([ multi_r , multi_c  ])
            else: FP_multires.append([ multi_r , multi_c ])
        corrected_multires = np.array([coord for coord in corrected_multires if coord not in FP_multires])
        img_tictoc = time.time()-img_tic
        #Si se ingresó un directorio para guardar la información obtenida
        if savedir:
            #Termina el tiempo de algoritmo
            total_tictoc = time.time()-total_tic
            
            #Se crean los directorios para guardar las imágenes y estadísticas
            os.makedirs(savedir, exist_ok=True)
            os.makedirs(savedir + '/orig/', exist_ok=True)
            os.makedirs(savedir + '/seg/', exist_ok=True)
            os.makedirs(savedir + '/heat_map/', exist_ok=True)
            
            #Se encuentra el número de imagen que se creará
            img_num =  len([s for s in os.listdir(savedir + '/orig/') if s.startswith('img')]) if not overwrite else 0
            img_num_seg = len([s for s in os.listdir(savedir + '/seg/') if s.startswith('seg_img')]) if not overwrite else 0
            img_num_hm_frame = len([s for s in os.listdir(savedir + '/heat_map/') if s.startswith('heatmap_frames')]) if not overwrite else 0
            img_num_hm = len([s for s in os.listdir(savedir + '/heat_map/') if s.startswith('heatmap_dens')]) if not overwrite else 0
            
            #Se guardan las imágenes correspondientes
            if imsave:
                cv2.imwrite(savedir + '/orig' + '/img_' + str(img_num + 1) + '.jpg', img)
                cv2.imwrite(savedir + '/seg' + '/seg_img_' + str(img_num_seg + 1) + '.jpg', seg_img)
                cv2.imwrite(savedir + '/heat_map' + '/heatmap_frames_'+ str(img_num_hm_frame + 1) + '.jpg', heatmap_img)
                cv2.imwrite(savedir + '/heat_map' + '/heatmap_dens_'+ str(img_num_hm + 1) + '.jpg', hm_img)
            
            #Se guarda un txt con las estadísticas extraídas, además de apendizar los datos de la imagen al diccionario de detección
            estimated_density = np.sum(calc_img) / ( seg_img.shape[0] * seg_img.shape[1] )
            print('La densidad de maleza tipo trébol encontrada en la imagen es: ' + str(estimated_density))
            #Si el diccionario de segmentación está vacío se rellenan los datos desde cero, si no, se rellena con el dato siguiente
            param_dict = { 'overlaps' : [overlap_factor_first, overlap_factor_second], 'IOUs' : [IOU_multires, IOU_hm], 'pred_batch' : pred_batch, 'neighbours' : r_neighbours,\
                'classes': [heat_map_class, multi_res_name, bg_class], 'region_size' : multi_res_win_size, 'frame_size' : frame_size }
            if not seg_dict: seg_dict = { 'img_size': [img.shape], 'multires_coords': [corrected_multires], 'multires_wh' : [wh_multires_list],'hm': [full_heat_map[:,:,0]], 'bin_hm' : [(full_heat_map[:,:,0] > hm_thresh).astype(np.uint8)], 'est_dens' : [estimated_density],\
                'param_dict' : [param_dict], 'compute_time' : [total_tictoc] }
            else:
                seg_dict['img_size'].append(img.shape)
                seg_dict['multires_coords'].append(corrected_multires)
                seg_dict['multires_wh'].append(wh_multires_list)
                seg_dict['hm'].append(full_heat_map[:,:,0])
                seg_dict['bin_hm'].append((full_heat_map[:,:,0] > hm_thresh).astype(np.uint8))
                seg_dict['est_dens'].append(estimated_density)
                seg_dict['param_dict'].append(param_dict)
                seg_dict['compute_time'].append(total_tictoc)
            
            txt_name = savedir + '/stats_log.txt'
            if multires_coord.shape[0] > 0:
                multires_string = '- Coordenadas multiresolución: '
                for multires_coord_idx in range(corrected_multires.shape[0]): multires_string = multires_string + str(corrected_multires[multires_coord_idx]) + '\r\n' + ' ' * 31 * (corrected_multires.shape[0]>1)
            else: multires_string = ''
            
            #Si ya existe el archivo simplemente se apendizan los datos
            if os.path.isfile(txt_name):
                txt_file = open(txt_name,'a')
                txt_2_save = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Imagen '+ str(img_num + 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '- Tamaño de la imagen: ' + str(img.shape) + '\r\n' +\
                    '- Tamaño de los cuadros para región: ' + str(multi_res_win_size) + '\r\n' +\
                    '- Overlap de cuadros para región: ' + str(overlap_factor_first) + '\r\n' +\
                    '- Cantidad de regiones multiresolución encontradas: ' + str(multires_coord.shape[0]) + '\r\n' +\
                    multires_string + \
                    '- Batch de predicción para regiones: ' + str(pred_batch[0]) + '\r\n'+\
                    '- Tiempo de cómputo para regiones: ' + str(region_tictoc) + ' [s]\r\n'+\
                    '- Tamaño de los cuadros para estimación de densidad: '+ str(frame_size) + '\r\n' +\
                    '- Overlap de cuadros para estimación de densidad: ' + str(overlap_factor_second) + '\r\n' +\
                    '- Densidad estimada de maleza tipo trébol: ' + str(estimated_density) + '\r\n' +\
                    '- Batch de predicción para estimación de densidad: ' + str(pred_batch[1]) + '\r\n'+\
                    '- Tiempo de cómputo para estimación de densidad: ' + str(hm_tictoc) + ' [s]\r\n'+\
                    '- Tiempo de cómputo total: ' + str(total_tictoc) + ' [s]\r\n'+\
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
            #Si no, se escribe con encabezado
            else:
                txt_file = open(txt_name,'w+')
                txt_2_save = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Características de detección por imagen~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Imagen '+ str(img_num + 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '- Tamaño de la imagen: ' + str(img.shape) + '\r\n' +\
                    '- Tamaño de los cuadros para región: ' + str(multi_res_win_size) + '\r\n' +\
                    '- Overlap de cuadros para región: ' + str(overlap_factor_first) + '\r\n' +\
                    '- Cantidad de regiones multiresolución encontradas: ' + str(multires_coord.shape[0]) + '\r\n' +\
                    multires_string + \
                    '- Batch de predicción para regiones: ' + str(pred_batch[0]) + '\r\n'+\
                    '- Tiempo de cómputo para regiones: ' + str(region_tictoc) + ' [s]\r\n'+\
                    '- Tamaño de los cuadros para estimación de densidad: '+ str(frame_size) + '\r\n' +\
                    '- Overlap de cuadros para estimación de densidad: ' + str(overlap_factor_second) + '\r\n' +\
                    '- Densidad estimada de maleza tipo trébol: ' + str(estimated_density) + '\r\n' +\
                    '- Batch de predicción para estimación de densidad: ' + str(pred_batch[1]) + '\r\n'+\
                    '- Tiempo de cómputo para estimación de densidad: ' + str(hm_tictoc) + ' [s]\r\n'+\
                    '- Tiempo de cómputo: ' + str(total_tictoc) + ' [s]\r\n'+\
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
            
            txt_file.write(txt_2_save)
            txt_file.close()
            #Se retornan las coordenadas detectadas como wild lettuce, el heat_map y su estimación de densidad correspondiente y la imagen segmentada completa
            
            return multires_coord, seg_img, (np.sum(calc_img) / ( seg_img.shape[0] * seg_img.shape[1] ) ), seg_img, seg_dict
        
        #cv2.imshow('Imagen semantica binaria', cv2.resize( seg_img, None, fx = .25, fy = .25) )
        #cv2.imshow('Imagen semantica', cv2.resize( seg_img, None, fx = .25, fy = .25) )           
        
        #cv2.destroyAllWindows()
        #cv2.waitKey(0)

        #Se retornan las coordenadas detectadas como wild lettuce, el heat_map y su estimación de densidad correspondiente y la imagen segmentada completa
        return multires_coord, seg_img, (np.sum(calc_img) / ( seg_img.shape[0] * seg_img.shape[1] ) ), seg_img
        
    #Si se elige el método de sliding windows con multi-resolución
    elif method == 'multires_sld':
        class_mask, class_coord_dict, new_img = \
            CNN_region_seg(img, second_model, class_dict, win_size = frame_size,overlap_factor_heatmap = overlap_factor_second, overlap_factor_multires = overlap_factor_first,\
                multi_res_win_size = multi_res_win_size, multi_res_name = multi_res_name, method = 'multires')
    
        '''
        if model_type == 'CNN': class_mask, class_coord_dict, new_img = \
            CNN_region_seg(img, first_model, class_dict, win_size = frame_size, overlap_factor = overlap_factor, multi_res_win_size = multi_res_win_size, multi_res_name = multi_res_name )
        else: class_mask, class_coord_dict, new_img = CSN_region_seg(img, first_model, X_train, Y_train, class_dict, method = 'sld_win_feat', win_size = frame_size, \
            min_frames_region = 32, overlap_factor = overlap_factor, thresh = 'mean_half', multi_res_win_size = multi_res_win_size, multi_res_name = multi_res_name )
        '''
        #Se despliegan los resultados pedidos
        if heat_map_display: 
            heat_map = class_coord_dict['heat_map']
            class_name_list = class_coord_dict['class_name_list']
            
            win_show = 'Imagen original'
            cv2.namedWindow(win_show, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(win_show, 0, 0)
            cv2.imshow(win_show, cv2.resize( img, (960,960)))
            
            for idx in range(heat_map.shape[-1]):
                win_show = 'Mapa de calor para la clase: ' + str(class_name_list[idx])
                new_class_mask = np.zeros( ( class_mask.shape[0] , class_mask.shape[1]), dtype = np.uint8 )
                new_class_mask[ class_mask == idx ] = 1
                map_x_img = np.zeros( new_img.shape, dtype = np.uint8 )
                map_x_img[:,:,0] = np.multiply ( ( np.multiply( new_img[:,:,0].astype(np.float32), heat_map[:,:,idx].astype(np.float32) ) / 255 ).astype(np.uint8), new_class_mask)
                map_x_img[:,:,1] = np.multiply ( ( np.multiply( new_img[:,:,1].astype(np.float32), heat_map[:,:,idx].astype(np.float32) ) / 255 ).astype(np.uint8), new_class_mask)
                map_x_img[:,:,2] = np.multiply ( ( np.multiply( new_img[:,:,2].astype(np.float32), heat_map[:,:,idx].astype(np.float32) ) / 255 ).astype(np.uint8), new_class_mask)
                cv2.namedWindow(win_show, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(win_show, 0, 0)
                cv2.imshow(win_show, cv2.resize( map_x_img, (960,960)))
                print(idx)
                
        if class_mask_display:  
            win_show = 'Mapa de clases'
            cv2.namedWindow(win_show, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(win_show, 0, 0)
            cv2.imshow(win_show, cv2.resize( 84 * (class_mask+1), (960,960)))
            
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
#Función que ejecuta pipelines sucesivas sobre una carpeta de imágenes especificada
def folder_pipeline(folder_dir,first_model, second_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = [], feat_mean_second = [], overlap_factor_first = 0.5,\
    overlap_factor_second = 0.75, multi_res_win_size = (1280, 1280), multi_res_name = 'wild lettuce', IOU_multires = .25, IOU_hm = .75, heat_map_class = 'trebol', heat_map_display = True, bg_class = 'pasto',\
        class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'], savedir = '', overwrite = False, pred_batch = [32, 32], r_neighbours = 0 , imsave = True, feat_model_only = True, \
            hm_thresh = 0.5, region_wl_thresh = 0.5, region_hm_thresh = 0.5, selection_dict_sld = '', feats_param_dict_sld = '',  selection_dict_reg = '', feats_param_dict_reg = ''):
            
    #Se lee la carpeta de origen de los datos y se adjuntan las imágenes que contengan un formato permitido
    new_name_list = [folder_dir + '/' + s for s in os.listdir(folder_dir) if ( s.endswith('jpg') or s.endswith('png') or s.endswith('jpeg') )]
    seg_dict = {}
    #Se cuenta el tiempo de algoritmo para todas las imágenes
    tic = time.time()
    #Se leen las imágenes contenidas en la carpeta y se borra la carpeta si se eligió sobreescribir
    new_savedir = savedir + '/' + model_type[0] +  '-' + model_type[1]
    #Si el primer modelo es CSN se extrae sin capa de salida
    if model_type[0] == 'CSN' and feat_model_only:
        semantic_net = Sequential()
        #print(semantic_net.summary())
        for layer in first_model.layers[0:-1]: semantic_net.add(layer)
        #Se extrae el modelo sin la capa de salida
        sem_model = Sequential()
        feat_model_first = Sequential()
        for idx, layer in enumerate(first_model.layers):
            if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                for in_layer in layer.layers:
                    if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                        sem_model.add(in_layer)
                        feat_model_first.add(in_layer)
                    elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model_first.add(in_layer)
            elif isinstance(layer,Dense) or isinstance(layer, GlobalAveragePooling2D):
                feat_model_first.add(layer)
    else: feat_model_first = []
    #Si el segundo modelo también es CSN se extrae sin capa de salida
    if model_type[1] == 'CSN' and feat_model_only:
        semantic_net = Sequential()
        #print(semantic_net.summary())
        for layer in second_model.layers[0:-1]: semantic_net.add(layer)
        #Se extrae el modelo sin la capa de salida
        sem_model = Sequential()
        feat_model_second = Sequential()
        for idx, layer in enumerate(first_model.layers):
            if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                for in_layer in layer.layers:
                    if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                        sem_model.add(in_layer)
                        feat_model_second.add(in_layer)
                    elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model_second.add(in_layer)
    else: feat_model_second = []
    
    if overwrite and os.path.isdir(new_savedir): rmtree(new_savedir)
    for img_name in new_name_list:
        
        img = cv2.imread(img_name)
        img_mean_size = np.min([img.shape[0], img.shape[1]])
        #img_mean_size = int(img.shape[0]/2 + img.shape[1]/2)
        #img_mean_size = int(img.shape[0]/2 + img.shape[1]/2)/2 + np.min([img.shape[0], img.shape[1]])/2
        print(img_mean_size)
        multi_res_win_size_new = [(int(img_mean_size*win_size), int(img_mean_size*win_size)) for win_size in multi_res_win_size]
        print(multi_res_win_size_new)
        #img = cv2.cvtColor(np.array(Image.open(img_name)), cv2.COLOR_RGB2BGR)
        multires_coord, final_hm_img, estimated_density, seg_img, seg_dict =\
            pipeline(img, first_model, second_model ,class_dict_reg, class_dict_sld, feat_model_first = feat_model_first, frame_size = frame_size, feat_mean_first = feat_mean_first, feat_mean_second = feat_mean_second,\
            overlap_factor_first = overlap_factor_first, overlap_factor_second = overlap_factor_second, multi_res_win_size = multi_res_win_size_new, multi_res_name = multi_res_name,\
                IOU_multires = IOU_multires, IOU_hm = IOU_hm, heat_map_class = heat_map_class, heat_map_display = heat_map_display, bg_class = bg_class, class_mask_display = class_mask_display, method = method,\
                    model_type = model_type, savedir = new_savedir, overwrite = False, pred_batch = pred_batch, r_neighbours = r_neighbours, seg_dict = seg_dict, imsave = imsave, hm_thresh = hm_thresh,\
                        region_wl_thresh = region_wl_thresh, region_hm_thresh = region_hm_thresh, selection_dict_sld = selection_dict_sld, feats_param_dict_sld = feats_param_dict_sld,\
                            selection_dict_reg = selection_dict_reg, feats_param_dict_reg = feats_param_dict_reg)
            
    #Se calcula el tiempo promedio del algoritmo por imagen
    tictoc = time.time()-tic
    mean_time = tictoc / len(new_name_list)
    
    if savedir:
        txt_name = savedir + '/stats_log.txt'
        txt_file = open(txt_name,'a')
        txt_file.write('- Tiempo promedio por imagen: ' + str(mean_time) + '[s]\r\n')
        txt_file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n')
        txt_file.close()
        
        #Se guarda el diccionario de las estadísticas calculadas
        seg_dict_name = new_savedir + '/seg_dict.pickle'
        with open(seg_dict_name, 'wb') as handle: pickle_dump(seg_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
    
    return mean_time
    
def video_pipeline(video_dir, first_model, second_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = [], feat_mean_second = [], overlap_factor_first = 0.5,\
    overlap_factor_second = 0.75, multi_res_win_size = (1280, 1280), multi_res_name = 'wild lettuce', IOU_multires = .25, IOU_hm = .75, heat_map_class = 'trebol', heat_map_display = True, bg_class = 'pasto',\
        class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'], savedir = '', overwrite = False, pred_batch = [32, 32], r_neighbours = 0 , imsave = True, feat_model_only = True, \
            fps = 30, w = 1920, h = 1080, hm_thresh = .5, region_wl_thresh = 0.5,  region_hm_thresh = 0.5):
    
    #Se lee la carpeta de origen de los datos y se adjuntan los videos que contengan un formato permitido
    video_name_list = [video_dir + '/' + s for s in os.listdir(video_dir) if ( s.endswith('.mpeg') or s.endswith('.mp4') or s.endswith('.avi') )]
    seg_dict = {}
    #Se cuenta el tiempo de algoritmo para todos las videos
    tic = time.time()
    #Se leen las imágenes contenidas en la carpeta y se borra la carpeta si se eligió sobreescribir
    new_savedir = savedir + '/' + model_type[0] +  '-' + model_type[1]
    #Si el primer modelo es CSN se extrae sin capa de salida
    if model_type[0] == 'CSN' and feat_model_only:
        semantic_net = Sequential()
        #print(semantic_net.summary())
        for layer in first_model.layers[0:-1]: semantic_net.add(layer)
        #Se extrae el modelo sin la capa de salida
        sem_model = Sequential()
        feat_model_first = Sequential()
        for idx, layer in enumerate(first_model.layers):
            if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                for in_layer in layer.layers:
                    if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                        sem_model.add(in_layer)
                        feat_model_first.add(in_layer)
                    elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model_first.add(in_layer)
    else: feat_model_first = []
    #Si el segundo modelo también es CSN se extrae sin capa de salida
    if model_type[1] == 'CSN' and feat_model_only:
        semantic_net = Sequential()
        #print(semantic_net.summary())
        for layer in second_model.layers[0:-1]: semantic_net.add(layer)
        #Se extrae el modelo sin la capa de salida
        sem_model = Sequential()
        feat_model_second = Sequential()
        for idx, layer in enumerate(first_model.layers):
            if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                for in_layer in layer.layers:
                    if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                        sem_model.add(in_layer)
                        feat_model_second.add(in_layer)
                    elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model_second.add(in_layer)
    else: feat_model_second = []
    
    if overwrite and os.path.isdir(new_savedir): rmtree(new_savedir), os.makedirs(new_savedir, exist_ok= True)
    
    #Se lee cada video y se construyen los cuadros a partir de él
    for video_idx, video_name in enumerate(video_name_list):
        
        capture = cv2.VideoCapture(video_name)
        #Se instancia el video de salida con su codificador
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        new_savedir = new_savedir + '/' + os.listdir(video_dir)[video_idx]
        out = cv2.VideoWriter(new_savedir + '/output.mp4', fourcc, fps, (w,  h))
        os.makedirs(new_savedir, exist_ok= True)
        os.makedirs(new_savedir + '/frames', exist_ok = True )
        success,img = capture.read()
        count = 0
        while success and count < 750:
            img_mean_size = np.min([img.shape[0], img.shape[1]])
            #img_mean_size = int(img.shape[0]/2 + img.shape[1]/2)
            #img_mean_size = int(img.shape[0]/2 + img.shape[1]/2)/2 + np.min([img.shape[0], img.shape[1]])/2
            print(img_mean_size)
            multi_res_win_size_new = [(int(img_mean_size*win_size), int(img_mean_size*win_size)) for win_size in multi_res_win_size]
            print(multi_res_win_size_new)
            #img = cv2.cvtColor(np.array(Image.open(img_name)), cv2.COLOR_RGB2BGR)
            multires_coord, final_hm_img, estimated_density, seg_img, seg_dict =\
                pipeline(img, first_model, second_model ,class_dict_reg, class_dict_sld, feat_model_first = feat_model_first, frame_size = frame_size, feat_mean_first = feat_mean_first, feat_mean_second = feat_mean_second,\
                overlap_factor_first = overlap_factor_first, overlap_factor_second = overlap_factor_second, multi_res_win_size = multi_res_win_size_new, multi_res_name = multi_res_name,\
                    IOU_multires = IOU_multires, IOU_hm = IOU_hm, heat_map_class = heat_map_class, heat_map_display = heat_map_display, bg_class = bg_class, class_mask_display = class_mask_display, method = method,\
                        model_type = model_type, savedir = new_savedir, overwrite = False, pred_batch = pred_batch, r_neighbours = r_neighbours, seg_dict = seg_dict, imsave = imsave, hm_thresh = hm_thresh, region_wl_thresh = region_wl_thresh,  region_hm_thresh = region_hm_thresh)
            #Se guarda cada imagen perteneciente al video
            out.write(seg_img)
            #cv2.imwrite(new_savedir + '/frames' + '/' + str(count) + '.jpg', img)     
            success,img = capture.read()
            #print('Read a new frame: ', success)
            count += 1
            
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        #Se calcula el tiempo promedio del algoritmo por imagen
        tictoc = time.time()-tic
        mean_time = tictoc / count
        #Se comprime el video generado
        import subprocess
        in_name = new_savedir + '/output.mp4'
        out_name = new_savedir + '/output_compressed.mp4'
        subprocess.run('ffmpeg -y -i ' + in_name + ' -b 20000k ' + out_name)
        if savedir:
            txt_name = savedir + '/stats_log.txt'
            txt_file = open(txt_name,'a')
            txt_file.write('- Tiempo promedio por imagen: ' + str(mean_time) + '[s]\r\n')
            txt_file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n')
            txt_file.close()
            
            #Se guarda el diccionario de las estadísticas calculadas
            seg_dict_name = new_savedir + '/seg_dict.pickle'
            with open(seg_dict_name, 'wb') as handle: pickle_dump(seg_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
        