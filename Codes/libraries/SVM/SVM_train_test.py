#LIBRERÍA PARA EL PROCESAMIENTO DE DATOS PARA USO EN REDES NEURONALES
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Importe de librerías externas
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import cv2
import os, sys
from pickle import load as pickle_load, dump as pickle_dump, HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL
from numpy.random import seed
from shutil import rmtree, copyfile
from distutils.dir_util import copy_tree
from itertools import permutations
from skimage.feature import hog, local_binary_pattern as lbp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import time
import multiprocessing as mp

##PYBALU##
from pybalu import feature_selection as fs
from pybalu import feature_transformation as ft

##
##Importe de librerías propias
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from SVM.feature_functions import feature_extraction, multiclass_SVM_set_creator, multiclassparser, lbp_feats_calc, hog_feats_calc, multiclass_preprocessing
from data_handling.img_processing import random_augment

#Función que entrena un clasificador SVM
def SVM_train(X_train, Y_train, clf_folder = [], kernel = 'rbf', degree = 0, gamma = 0.01, Cost = 2, clean_feats = True, sfs_train_num = 25, pca_train_num = 10):
        
    if clean_feats:
        #Se limpian los datos con el método de PyBalu
        X_train_clean_indices = fs.clean(X_train)
        X_train_clean = X_train[:, X_train_clean_indices]
        X_train = X_train_clean
        #print('Data cleaning completo')
        
    else: X_train_clean_indices = [i for i in range(X_train.shape[1])]
    
    #Para obviar SFS igualarlo a 0
    if sfs_train_num != 0:
        
        sfs_n = np.min([sfs_train_num, X_train.shape[1]])
        #Se realiza un SFS con el método de PyBalu
        #print('Comienza el proceso de SFS')
        X_sfs_indices = fs.sfs(X_train, Y_train, n_features = sfs_n, method = 'fisher',show=False)
        X_train_sfs = X_train_clean[:, X_sfs_indices]
        X_train = X_train_sfs
        #print('SFS completo')
        
    else: X_sfs_indices = [i for i in range(X_train.shape[1])]
    
    #Para obviar PCA igualarlo a 0
    pca_n = np.min([pca_train_num, X_train.shape[1]])
    if pca_train_num != 0:
        
        #Se realiza una transformación PCA con el método de PyBalu
        #print('Comienza el proceso de PCA')
        X_train_pca, _lambda, A, mx, _ = ft.pca(X_train, n_components = int(pca_n), energy=0)
        X_train = X_train_pca[:, 0 : pca_n]
        #print('SFS completo')
        
    else: 
        
        A = np.zeros((0))
        mx = 0
        
    #Se crea y entrena el clasificador con los parámetros especificados
    #print('Comienza el entrenamiento de la SVM')
    svm_clf = SVC(C = Cost, kernel = kernel, gamma = gamma, degree = degree, max_iter = 5000, decision_function_shape='ovo')
    svm_clf.fit(X_train, Y_train)
    sfs_dict = {'X_train_clean_indices': X_train_clean_indices, 'X_sfs_indices': X_sfs_indices,  'pca_dict': {'A': A, 'mx': mx, 'pca_n': pca_n}}
    #print('Entrenamiento de SVM completo!')
    
    #Se guarda el diccionario de mejores configuraciones
    os.makedirs(clf_folder, exist_ok=True)
    with open(clf_folder + '/best_clf.pickle', 'wb') as handle: pickle_dump(svm_clf, handle, protocol=pickle_HIGHEST_PROTOCOL)
    
    #Se guarda el diccionario de mejores configuraciones
    with open(clf_folder + '/best_pca.pickle', 'wb') as handle: pickle_dump(sfs_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
    
    return svm_clf, sfs_dict

#Función para evaluar el desempeño de un clasificador SVM sobre un conjunto de prueba utilizando técnicas de selección de características
def SVM_test(X_test, Y_test, clf, sfs_dict):
    
    #Se cargan los parámetros de selección de características
    X_clean_indices = sfs_dict['X_train_clean_indices']
    X_sfs_indices = sfs_dict['X_sfs_indices']
    X_test_clean = X_test[:, X_clean_indices]
    X_test_sfs = X_test_clean[:, X_sfs_indices]
    
    #Se transforma X_test con las especificaciones de PCA obtenidas en el entrenamiento
    pca_dict = sfs_dict['pca_dict']
    mx = pca_dict['mx']
    A = pca_dict['A']
    pca_n = pca_dict['pca_n']
    mx_mat = np.matlib.repmat(mx, len(Y_test), 1)
    
    if pca_n != 0:
        X_test_pca = np.matmul( (X_test_sfs - mx_mat) , A )
        X_test_pca = X_test_pca[:, 0 : pca_n ]
    else: X_test_pca = X_test_sfs
    
    #Se predice sobre las características de entrada y se calcula el error obtenido con el clasificador correspondiente
    Y_pred = clf.predict( X_test_pca )
    accuracy = len( [y_correct for i, y_correct in enumerate(Y_pred) if y_correct == Y_test[i]] ) / Y_test.shape[0] * 100
    
    return accuracy

#Función para iterar sobre características y configuraciones de SVM para encontrar un óptimo sobre un conjunto de datos
def SVM_iterator(source_list, feat_destiny, clf_destiny, lbp_r_range = [1, 1 , 1], lbp_points_range = [8, 8, 1], lbp_type = 'nri_uniform', lbp_frame_size_range = [16, 64, 8],\
    pixels_x_range = [8, 16, 4], pixels_y_range = [8, 16, 4], block_num_x_range = [1, 3, 1], block_num_y_range = [1, 3, 1], orientations_n_range = [3, 8, 1], img_shape = [256, 256], \
        train_test_rate = 0.75, test_val_rate = 0.5,\
         kernel_list = ['linear', 'poly', 'rbf'], degree_range = [2, 4, 1], gamma_range = [0.01, 10, 0.01],  clean_feats = True, sfs_train_num_range = [10, 25, 5], pca_train_num_range = [5, 15, 5]):
    
    #Se generan los vectores para la iteración sobre la extracción de características
    lbp_r_vec = np.linspace(lbp_r_range[0], lbp_r_range[1], (lbp_r_range[1] - lbp_r_range[0]) / lbp_r_range[2] + 1).astype(int)
    lbp_points_vec = np.linspace(lbp_points_range[0], lbp_points_range[1], (lbp_points_range[1] - lbp_points_range[0]) / lbp_points_range[2] + 1).astype(int)
    lbp_frame_size_vec = np.linspace(lbp_frame_size_range[0], lbp_frame_size_range[1], (lbp_frame_size_range[1] - lbp_frame_size_range[0]) / lbp_frame_size_range[2] + 1).astype(int)
    pixels_x_vec = np.linspace(pixels_x_range[0], pixels_x_range[1], (pixels_x_range[1] - pixels_x_range[0]) / pixels_x_range[2] + 1).astype(int)
    pixels_y_vec = pixels_x_vec
    block_num_x_vec = np.linspace(block_num_x_range[0], block_num_x_range[1], (block_num_x_range[1] - block_num_x_range[0]) / block_num_x_range[2] + 1).astype(int)
    block_num_y_vec = np.linspace(block_num_y_range[0], block_num_y_range[1], (block_num_y_range[1] - block_num_y_range[0]) / block_num_y_range[2] + 1).astype(int)
    orientations_n_vec = np.linspace(orientations_n_range[0], orientations_n_range[1], (orientations_n_range[1] - orientations_n_range[0]) / orientations_n_range[2] + 1).astype(int)
    gamma_vec = np.linspace(gamma_range[0], gamma_range[1], (gamma_range[1] - gamma_range[0]) / gamma_range[2] + 1)
    if sfs_train_num_range[2] != 0: sfs_vec = np.linspace(sfs_train_num_range[0], sfs_train_num_range[1], (sfs_train_num_range[1] - sfs_train_num_range[0]) / sfs_train_num_range[2] + 1).astype(int)
    else: sfs_vec = np.zeros(0)

    best_val_acc = 0
    #Se itera sobre todos los parámetros de características y del kernel
    for lbp_r in lbp_r_vec:
        for lbp_points in lbp_points_vec:
            for lbp_frame_size in lbp_frame_size_vec:
                for pixels_x in pixels_x_vec:
                    for pixels_y in pixels_y_vec:
                        for block_num_x in block_num_x_vec:
                            for block_num_y in block_num_y_vec:
                                for orientations_n in orientations_n_vec:
                                    
                                    X, Y, class_dict = feature_extraction(source_list, feat_destiny, lbp_r = lbp_r, lbp_points = lbp_points, lbp_type = lbp_type, lbp_frame_size = [lbp_frame_size, lbp_frame_size],\
                                    pixels_x = pixels_x, pixels_y = pixels_y, block_num_x = block_num_x, block_num_y = block_num_y, orientations_n = orientations_n, img_shape = img_shape)
                                    X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = \
                                        multiclass_SVM_set_creator(X, Y, class_dict, feat_destiny, train_test_rate = train_test_rate, test_val_rate = test_val_rate)
                                    for kernel in kernel_list:
                                        
                                        #Si el kernel del clasificador es polinomial se crea un vector de grados, si no, no
                                        if kernel == 'poly': degree_vec = np.linspace(degree_range[0], degree_range[1], (degree_range[1] - degree_range[0]) / degree_range[2] + 1).astype(int)
                                        else: degree_vec = np.array([1])
                                        print(kernel)
                                        for degree in degree_vec:
                                            for gamma in gamma_vec:
                                                if sfs_vec.shape[0] != 0:
                                                    for sfs_n in sfs_vec:
                                                        print('sfs_n:' + str(sfs_n))
                                                        #Se evita que pca tenga más dimensiones que sfs
                                                        pca_vec = np.linspace(np.min ( [pca_train_num_range[0], sfs_n ] ),\
                                                            np.min( [pca_train_num_range[1], sfs_n ] ), \
                                                                ( np.min( [pca_train_num_range[1], sfs_n] ) - np.min([pca_train_num_range[0], sfs_n] ))\
                                                                    / pca_train_num_range[2] + 1).astype(int)
                                                        for pca_n in pca_vec:
                                                            print(pca_n)
                                                            tic = time.time()
                                                            #Se entrena el clasificador con los parámetros escogidos sobre los conjuntos de características
                                                            svm_clf, sfs_dict = SVM_train(X_train, Y_train, clf_folder = clf_destiny,\
                                                                kernel = kernel, degree = degree, gamma = gamma, Cost = 2, clean_feats = clean_feats, sfs_train_num = sfs_n, pca_train_num = pca_n)
                                                            toc = time.time()
                                                            
                                                            print('tictoc:' + str(toc-tic))
                                                            #Si el desempeño sobre el conjunto de validación es mejor al histórico, se actualiza el mejor clasificador
                                                            val_acc = SVM_test(X_val, Y_val, svm_clf, sfs_dict)
                                                            
                                                            if val_acc > best_val_acc:
                                                                best_lbp_r = lbp_r
                                                                best_lbp_points = lbp_points
                                                                best_lbp_type = lbp_type
                                                                best_lbp_frame_size = lbp_frame_size
                                                                best_pixels_x = pixels_x
                                                                best_pixels_y = pixels_y
                                                                best_block_num_x = block_num_x
                                                                best_block_num_y = block_num_y
                                                                best_orientations_n = orientations_n
                                                                
                                                                print('La mejor accuracy encontrada para el clasificador SVM es: ' + str(np.round(val_acc, 2)) + '%')
                                                                best_val_acc = val_acc
                                                                best_svm_clf = svm_clf
                                                                best_sfs_dict = sfs_dict
                                                                best_feat_dict = {'best_lbp_r' : lbp_r, 'best_lbp_points' : lbp_points, 'best_lbp_frame_size': lbp_frame_size,\
                                                                    'best_pixels_x':pixels_x, 'best_pixels_y':pixels_y, 'best_block_num_x':block_num_x, 'best_block_num_y':block_num_y,\
                                                                        'best_orientations_n':orientations_n}
                                                                best_dict = {'best_svm_clf': best_svm_clf, 'best_sfs_config': best_sfs_dict, 'best_feat_params': best_feat_dict}
                                                                
                                                                #Se guarda la mejor configuración
                                                                with open(clf_destiny + '/best_dic.pickle', 'wb') as handle: pickle_dump(best_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)

                                                else:
                                                    tic = time.time()
                                                    #Se entrena el clasificador con los parámetros escogidos sobre los conjuntos de características
                                                    svm_clf, sfs_dict = SVM_train(X_train, Y_train, clf_folder = clf_destiny,\
                                                        kernel = kernel, degree = degree, gamma = gamma, Cost = 2, clean_feats = clean_feats, sfs_train_num = 0, pca_train_num = 0)
                                                    toc = time.time()
                                                    
                                                    print('tictoc:' + str(toc-tic))
                                                    #Si el desempeño sobre el conjunto de validación es mejor al histórico, se actualiza el mejor clasificador
                                                    val_acc = SVM_test(X_val, Y_val, svm_clf, sfs_dict)
                                                    
                                                    if val_acc > best_val_acc:
                                                        best_lbp_r = lbp_r
                                                        best_lbp_points = lbp_points
                                                        best_lbp_type = lbp_type
                                                        best_lbp_frame_size = lbp_frame_size
                                                        best_pixels_x = pixels_x
                                                        best_pixels_y = pixels_y
                                                        best_block_num_x = block_num_x
                                                        best_block_num_y = block_num_y
                                                        best_orientations_n = orientations_n
                                                        
                                                        print('La mejor accuracy sobre el conjunto de validación encontrada para el clasificador SVM es: ' + str(np.round(val_acc, 2)) + '%')
                                                        print(svm_clf.predict(X_val))
                                                        best_val_acc = val_acc
                                                        best_svm_clf = svm_clf
                                                        best_sfs_dict = sfs_dict
                                                        best_feat_dict = {'best_lbp_r' : lbp_r, 'best_lbp_points' : lbp_points, 'best_lbp_frame_size': lbp_frame_size,\
                                                            'best_pixels_x':pixels_x, 'best_pixels_y':pixels_y, 'best_block_num_x':block_num_x, 'best_block_num_y':block_num_y,\
                                                                'best_orientations_n':orientations_n}
                                                        best_dict = {'best_svm_clf': best_svm_clf, 'best_sfs_config': best_sfs_dict, 'best_feat_params': best_feat_dict}
                                                        
                                                        #Se guarda la mejor configuración
                                                        with open(clf_destiny + '/best_dic.pickle', 'wb') as handle: pickle_dump(best_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
                                                        val_accuracy_string = 'La mejor exactitud encontrada para el conjunto de validación es: ' + str(np.round(val_acc, 2)) + '%'
                                                        val_txt_file = open(clf_destiny + '/val_acc.txt','w+')
                                                        val_txt_file.write(val_accuracy_string)
                                                        val_txt_file.close()
    #Se calcula y guarda el valor de exactitud para la mejor configuración encontrada
    X, Y, class_dict = feature_extraction(source_list, feat_destiny, lbp_r = best_lbp_r, lbp_points = best_lbp_points, lbp_type = best_lbp_type, lbp_frame_size = [best_lbp_frame_size, best_lbp_frame_size],\
                                    pixels_x = best_pixels_x, pixels_y = best_pixels_y, block_num_x = best_block_num_x, block_num_y = best_block_num_y, orientations_n = best_orientations_n, img_shape = img_shape)
    _, _, X_test, _, _, Y_test, class_dict = \
                                        multiclass_SVM_set_creator(X, Y, class_dict, feat_destiny, train_test_rate = train_test_rate, test_val_rate = test_val_rate)
    test_acc = SVM_test(X_test, Y_test, best_svm_clf, best_sfs_dict)
    test_accuracy_string = 'La mejor exactitud encontrada para el conjunto de prueba es: ' + str(np.round(test_acc, 2)) + '%'
    test_txt_file = open(clf_destiny + '/test_acc.txt','w+')
    test_txt_file.write(test_accuracy_string)
    test_txt_file.close()
    
#Función para realizar un entrenamiento con validación cruzada a partir de la técnica de nfolds
def SVM_crossval_train_iterator( origin_folder, img_destiny, clf_folder, train_val_rate = 0.8, test_rate = .1, nfolds = 5, lbp_r= 1, lbp_points= 8, lbp_type = 'nri_uniform', lbp_frame_size = [64, 64],\
    pixels_x = 32, pixels_y= 32, block_num_x = 2, block_num_y = 1, orientations_n = 3, img_shape = [256, 256], kernel = 'rbf', degree = 3, gamma = 'scale',  clean_feats = True,\
        sfs_train_num = 0, pca_train_num = 0, data_aug = True, keep_orig = False):
    
    
    params_dict = {'lbp_r': lbp_r, 'lbp_points': lbp_points, 'lbp_type': lbp_type, 'lbp_frame_size': lbp_frame_size, 'hog_pixels_x': pixels_x,\
        'hog_pixels_y': pixels_y, 'block_num_x': block_num_x, 'block_num_y': block_num_y, 'orientations_n':orientations_n}
    #Se lee el directorio y se crean los nfolds estratificados para mantener la proporción de cada clases en las carpetas
    sources_list, class_name_list = multiclassparser(origin_folder)
    X, Y, class_dict = multiclass_preprocessing(sources_list, class_name_list, image_size = img_shape)
    X_test, Y_test = np.zeros(X.shape), np.zeros(Y.shape)
    #Se extraen los datos correspondientes a testeo para que resulten completamente ortogonales al conjunto de entrenamiento/validación (si se eligió un valor mayor a 0 y menor a .3)
    if test_rate > 0 and test_rate <.3:
        os.makedirs(img_destiny, exist_ok= True)
        #Para cada clase se extrae al azar el conjunto de testeo
        X_test, Y_test = np.zeros(0), np.zeros(0)
        X_new, Y_new = np.zeros(0), np.zeros(0)
        np.random.seed(5)
        class_n_img_names = []
        for class_n in class_dict['class_n_list']:
            #Se extraen los nombres de archivo válidos para la carpeta correspondiente a la clase que se está estratitificando
            class_n_img_names  = class_n_img_names + [img_name for img_name in os.listdir(sources_list[class_n]) if img_name.endswith('.jpg') or img_name.endswith('.jpeg')]
            
            #Se eligen los datos de testeo al azar y se extraen dichos datos del conjunto que servirá para entrenamiento y validación
            Y_n_indices = np.where(Y == class_n)[0]
            Y_n_test_indices = np.random.choice( Y_n_indices, size = int( np.round( test_rate*Y_n_indices.shape[0] ) ), replace= False )
            Y_n_train_indices = [Y_n_index  for Y_n_index in Y_n_indices if Y_n_index not in Y_n_test_indices]
            X_n_test, Y_n_test = X[ Y_n_test_indices, : ], Y[ Y_n_test_indices, : ]
            X_n_train, Y_n_train = X[ Y_n_train_indices, : ], Y[ Y_n_train_indices, : ]
            
            #Si no se han creado los conjuntos todavía, se crea la matriz de testeo y la nueva matriz de entrenamiento
            if X_test.shape[0] == 0:
                X_test = X_n_test
                Y_test = Y_n_test
                X_new = X_n_train
                Y_new = Y_n_train
            else:
                X_test = np.concatenate ( ( X_test, X_n_test ), axis = 0 )
                Y_test = np.concatenate ( ( Y_test, Y_n_test ), axis = 0 )
                X_new = np.concatenate ( ( X_new, X_n_train ), axis = 0 )
                Y_new = np.concatenate ( ( Y_new, Y_n_train ), axis = 0 )
            
            #Para efectos de repetibilidad y chequeo de datos se guardan las imágenes para cada conjunto
            class_n_destiny = img_destiny + '/' + class_dict['class_name_list'] [class_n]
            os.makedirs(class_n_destiny + '/train', exist_ok= True)
            os.makedirs(class_n_destiny + '/test', exist_ok= True)
            for n in Y_n_test_indices:
                test_img_name = class_n_img_names[n]
                cv2.imwrite(class_n_destiny + '/test/' + test_img_name, X[n,:,:,:])     
            for n in Y_n_train_indices:
                train_img_name = class_n_img_names[n]
                cv2.imwrite(class_n_destiny + '/train/' + train_img_name, X[n,:,:,:])  
        #Se actualizan los valores de matrices para entrenamiento            
        X, Y = X_new, Y_new
    
    #Se estratifican los datos dada la cantidad de nfolds ingresada
    skf = StratifiedKFold(n_splits=nfolds)
    best_val_acc = 0
    nfold_valacc_list = []
    nfold_testacc_list = []
    nfold_valmulticlass_list = []
    nfold_testmulticlass_list = []
    nfold_Cval_list = []
    nfold_Ctest_list = []
    fold_idx = 0
    
    for train_index, val_index in skf.split(X, Y):
        fold_idx += 1
        #Se separan en las posiciones para cada nfold
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        
        #Si se eligió aumentar
        if data_aug:
            #Si se mantiene la imagen que se aumentó es necesario repetir el valor de Y_train
            new_X_train =  random_augment(X_train, img_destiny + '/train_aug_' + str(fold_idx), augment_list = ['flip', 'rot', 'bright', 'pad', 'color', 'zoom', 'noise' ],\
                bright_change = 20, color_variation = 5, keep_orig = keep_orig, overwrite = True,\
                double_aug = 'rand', color_rand = False, pad_rand = True, zoom_factor = 0.75, noise_var = 30, original_folder_name = '', input_array = True)
            if keep_orig:
                new_Y_train = np.zeros( (new_X_train.shape[0], 1 ) )
                for y_index in range(new_Y_train.shape[0]) : new_Y_train[y_index] = Y_train[ int(y_index/2) ].astype(int)
            else: new_Y_train = np.copy(Y_train)
            #Se actualizan los valores de X_train y Y_train
            X_train, Y_train = new_X_train, new_Y_train
        
        #X_test, Y_test = np.zeros(X_val.shape), np.zeros(Y_val.shape)
        #Se crean los conjuntos de características para SVM de entrenamiento
        for img_ct, img in enumerate(X_train):
            #Transformación a escala de grises y cálculo de características HOG y LBP
            img_gray_eq = cv2.equalizeHist( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ) / 255
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp_total = np.asarray( lbp_feats_calc(img_gray_eq, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
            hog_total = np.asarray( hog_feats_calc(img_gray_eq, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
            #Si es la primera imagen procesada, se crean los vectores de características y de etiquetas
            if img_ct == 0: 
                new_X_train = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                new_Y_train = np.array([Y_train[img_ct]])
            else:
                new_X_train = np.concatenate( ( new_X_train,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
                new_Y_train = np.concatenate( (new_Y_train,  np.array([Y_train[img_ct]]) ), axis = 0 )    
        #Se crean los conjuntos de características para SVM de validación
        for img_ct, img in enumerate(X_val):
            #Transformación a escala de grises y cálculo de características HOG y LBP
            img_gray_eq = cv2.equalizeHist( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ) / 255
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp_total = np.asarray( lbp_feats_calc(img_gray_eq, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
            hog_total = np.asarray( hog_feats_calc(img_gray_eq, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
            #Si es la primera imagen procesada, se crean los vectores de características y de etiquetas
            if img_ct == 0: 
                new_X_val = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                new_Y_val = np.array([Y_val[img_ct]])
            else:
                new_X_val = np.concatenate( ( new_X_val,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
                new_Y_val = np.concatenate( (new_Y_val,  np.array([Y_val[img_ct]]) ), axis = 0 )    
        #Se crean los conjuntos de características para SVM de prueba
        for img_ct, img in enumerate(X_test):
            #Transformación a escala de grises y cálculo de características HOG y LBP
            img_gray_eq = cv2.equalizeHist( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ) / 255
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp_total = np.asarray( lbp_feats_calc(img_gray_eq, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
            hog_total = np.asarray( hog_feats_calc(img_gray_eq, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
            #Si es la primera imagen procesada, se crean los vectores de características y de etiquetas
            if img_ct == 0: 
                new_X_test = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                new_Y_test = np.array([Y_test[img_ct]])
            else:
                new_X_test = np.concatenate( ( new_X_test,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
                new_Y_test = np.concatenate( (new_Y_test,  np.array([Y_test[img_ct]]) ), axis = 0 )    
        #Se entrena el modelo SVM con el conjunto correspondiente
        new_Y_train, new_Y_val, new_Y_test = np.ravel(new_Y_train), np.ravel(new_Y_val), np.ravel(new_Y_test)
        best_val_fold = 0
        svm_clf, sfs_dict = SVM_train(new_X_train, new_Y_train, clf_folder = clf_folder + '/' + str(fold_idx), kernel = kernel, degree = degree, gamma = gamma, Cost = 2,\
            clean_feats = clean_feats, sfs_train_num = sfs_train_num, pca_train_num = pca_train_num)
        #Si el desempeño sobre el conjunto de validación es mejor al histórico, se actualiza el mejor clasificador
        val_acc = SVM_test(new_X_val, new_Y_val, svm_clf, sfs_dict)
        train_acc = SVM_test(new_X_train, new_Y_train, svm_clf, sfs_dict)
        
        if val_acc > best_val_fold:
            print(val_acc)
            X_val_clean_indices = sfs_dict['X_train_clean_indices'] 
            new_X_val, new_X_test = new_X_val[:, X_val_clean_indices], new_X_test[:, X_val_clean_indices]
            best_val_fold = val_acc
            
            multiclass_pred_val = np.uint8(svm_clf.predict(new_X_val))
            multiclass_pred_test = np.uint8(svm_clf.predict(new_X_test))
            test_multiclass_acc = accuracy_score(multiclass_pred_test, new_Y_test)
            val_multiclass_acc = accuracy_score(multiclass_pred_val, new_Y_val)
            
            #Cálculo de las matrices de confusión
            C_val = confusion_matrix(new_Y_val, multiclass_pred_val).astype(float) 
            C_test = confusion_matrix(new_Y_test, multiclass_pred_test).astype(float) 
            for class_n in range(C_val.shape[0]): C_val[class_n,:] = C_val[class_n, :] / np.sum(C_val[class_n], axis = 0)
            for class_n in range(C_test.shape[0]): C_test[class_n,:] = C_test[class_n, :] / np.sum(C_test[class_n], axis = 0)
        #Se agrega el valor de la mejor accuracy de fold a la lista, además de calcular y agregar las métricas para cada clase
        nfold_valacc_list.append(best_val_fold) 
        nfold_valmulticlass_list.append(val_multiclass_acc)
        nfold_testacc_list.append(test_multiclass_acc) 
        nfold_testmulticlass_list.append(test_multiclass_acc)
        nfold_Cval_list.append(C_val)
        nfold_Ctest_list.append(C_test)
        
        #Si la accuracy de la fold correspondiente es la mejor, se reemplazan los valores
        if best_val_fold > best_val_acc: 
            best_fold_idx = fold_idx
            best_val_acc = best_val_fold
            best_val_multiclass = nfold_valmulticlass_list[best_fold_idx-1]
            best_test_acc = nfold_testacc_list[best_fold_idx-1]
            best_test_multiclass = nfold_testmulticlass_list[best_fold_idx-1]
        
    #Se calculan las métricas
    nfold_valacc_avg = np.mean(nfold_valacc_list)
    nfold_valacc_sd = np.std(nfold_valacc_list)
    nfold_testacc_avg = np.mean(nfold_testacc_list)
    nfold_testacc_sd = np.std(nfold_testacc_list)
    
    nfold_valmulticlass_avg = np.mean(nfold_valmulticlass_list)
    nfold_valmulticlass_sd = np.std(nfold_valmulticlass_list)
    nfold_testmulticlass_avg = np.mean(nfold_testmulticlass_list)
    nfold_testmulticlass_sd = np.std(nfold_testmulticlass_list)

    #Se guardan los datos de la iteración sobre las nfolds en un diccionario
    nfold_dict_name = clf_folder + '/nfold_data.pickle'
    nfold_dict = {'nfolds' : nfolds, 'train_val_rate' : train_val_rate, 'test_rate' : test_rate, 'data_aug' : data_aug, 'class_dict': class_dict,\
        'best_val_acc' : best_val_acc, 'best_val_multiclass': best_val_multiclass, 'best_test_acc' : best_test_acc, 'best_test_multiclass' : best_test_multiclass,\
            'best_fold' : best_fold_idx, 'val_acc_stats' : [nfold_valacc_avg, nfold_valacc_sd], 'test_acc_stats' : [nfold_testacc_avg, nfold_testacc_sd], \
            'val_multiclass_stats' : [nfold_valmulticlass_avg, nfold_valmulticlass_sd], 'test_multiclass_stats' : [nfold_testmulticlass_avg, nfold_testmulticlass_sd],\
                'val_acc_list': nfold_valacc_list, 'test_acc_list':nfold_testacc_list,  'val_multiclass_list': nfold_valmulticlass_list, 'test_multiclass_list':nfold_testmulticlass_list, \
                    'Conf_val' : nfold_Cval_list, 'Conf_test' : nfold_Ctest_list,  'img_shape' : img_shape, 'params_dict' : params_dict}
    with open(nfold_dict_name, 'wb') as handle: pickle_dump(nfold_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)  
    
    return nfold_dict  

#Function to load the SVM model in a delivered directory
def SVM_load_model(model_dir):
    if os.path.isfile(model_dir + '/best_clf.pickle'): 
        with open(model_dir + '/best_clf.pickle', 'rb') as handle: SVM_clf = pickle_load(handle) 
        with open(model_dir + '/best_pca.pickle', 'rb') as handle: selection_dict = pickle_load(handle) 
    else: 
        raise ValueError('There is no classifier in the input directory')
    return SVM_clf, selection_dict

#Función que extrae estadísticas desde una carpeta matriz donde se encuentran (primero sub-carpetas) los datos obtenidos a partir de la validación cruzada, además de retornar el mejor valor obtenido para
#validación y testeo
def SVM_crossval_stat_calc(origin_folder, stat_save_destiny):
    
    #Se encuentran todos los nombres coincidentes con el diccionario generado en validación cruzada
    dict_names = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(origin_folder)
    for f in files if f.endswith('nfold_data.pickle')]
    
    #Para cada diccionario se encuentran los datos
    best_avg_acc = 0
    for dict_name in dict_names:
        with open(dict_name, 'rb') as handle: nfold_data_dict = pickle_load(handle)
        
        #Se recupera el valor promedio de la iteración y se compara con el mejor acumulado
        nfold_valacc_avg = nfold_data_dict['val_multiclass_stats'][0]
        print('val_acc_avg'), print(nfold_valacc_avg), print('best_avg_acc'), print(best_avg_acc)
        best_test_acc = 0
        #print( 'val_stats: ' +  str( nfold_data_dict['val_multiclass_stats']) )
        #input('press enter')
        if nfold_valacc_avg > best_avg_acc:
            #Se reinician los valores para el análisis por nfold
            #best_test_acc, best_val_acc = 0, 0
            best_avg_acc = nfold_valacc_avg
            #Se encuentra el mejor valor de validación (y posteriormente de testeo) dentro de los valores actuales
            best_val_nfold,  best_val_nfold_idx = np.max(nfold_data_dict['val_multiclass_list']),np.argmax(nfold_data_dict['val_multiclass_list'])
            best_test_nfold = nfold_data_dict['test_multiclass_list'][best_val_nfold_idx]
            #Si el valor de testeo actual es mejor al histórico se guardan los datos de mejor testeo
            if best_test_nfold>best_test_acc:
                print('best_validation_infold',best_val_nfold)
                best_val_acc, best_test_acc = best_val_nfold, best_test_nfold
                best_fold_name = dict_name
                best_confusionmat_test = nfold_data_dict['Conf_test'][best_val_nfold_idx]
                best_confusionmat_val = nfold_data_dict['Conf_val'][best_val_nfold_idx]
                #Cargado del mejor modelo de red neuronal encontrado
                best_fold_name = best_fold_name.rsplit('\\', 1)[:-1] [0] + '/' + str(best_val_nfold_idx + 1)
                best_feats_param_dict = nfold_data_dict['params_dict']
                best_val_stats = nfold_data_dict['val_multiclass_stats']
                #Obtención y guardado de valores promedio de las matrices de confusión para validación además del mejor de validación y de testeo
                nfold_confusionmat_val = nfold_data_dict['Conf_val']
                confusionmat_val_avg, confusionmat_val_sd = np.mean( nfold_confusionmat_val, axis = 0 ), np.std(nfold_confusionmat_val, axis = 0)
                
                #conf_matrix('', '', '', '', feat_mean = [], class_names = ['LV', 'pasto', 'trebol'], destiny = stat_save_destiny + '/conf_mats/nfold', show_bool = False,\
                #    pred = False, C_in = [confusionmat_val_avg, confusionmat_val_sd], title = 'N-fold validation data')
                #conf_matrix('', '', '', '', feat_mean = [], class_names = ['LV', 'pasto', 'trebol'], destiny = stat_save_destiny + '/conf_mats/bestfoldval', show_bool = False,\
                #    pred = False, C_in = best_confusionmat_val, title = 'Best fold validation data')
                #conf_matrix('', '', '', '', feat_mean = [], class_names = ['LV', 'pasto', 'trebol'], destiny = stat_save_destiny + '/conf_mats/bestfoldtest', show_bool = False,\
                #    pred = False, C_in = best_confusionmat_test, title = 'Best fold test data')
                #copyfile(dict_name, stat_save_destiny + '/best_dict.pickle')
                #plt.close('all')
                #input('ala')
            print('best_test' + str(best_test_acc))
            print(best_fold_name)
    
    #Se copia la carpeta que contiene el mejor modelo entrenado en la carpeta de destino y se retorna el modelo completo
    copy_tree(best_fold_name, stat_save_destiny + '/best_model/')
    model, selection_dict= SVM_load_model(best_fold_name)
    selection_dict['img_shape'] = nfold_data_dict['img_shape']
    return model, selection_dict, best_feats_param_dict