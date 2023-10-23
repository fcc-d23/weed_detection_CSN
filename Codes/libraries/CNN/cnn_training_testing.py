#TRAINING AND TESTING LIBRARY FOR USE IN NEURAL NETWORKS
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Import of external libraries
from math import nan
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from pickle import load as pickle_load, dump as pickle_dump, HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL
from shutil import rmtree, copyfile
from itertools import permutations
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, auc
from distutils.dir_util import copy_tree
from sklearn.model_selection import StratifiedKFold

##KERAS##
from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras.layers import Flatten, Dense, Lambda


##TENSORFLOW##
from tensorflow.compat.v1 import initialize_all_variables, initialize_all_variables, set_random_seed
from tensorflow import initialize_all_variables, get_default_graph

##
##Import of own libraries
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from data_handling.img_processing import random_augment
from CNN.cnn_database import load_data, multiclassparser, multiclass_preprocessing, multiclass_set_creator, CSN_pair_creator
from CNN.cnn_configuration import tensor_init, defineCallBacks, fully_CNN_creator
from CNN.csn_functions import contrastive_loss, CSN_accuracy, CSN_dist2pred, contrastive_loss_fcn, contrastive_loss, CSN_accuracy, CSN_centroid_finder
from CNN.seg_functions import folder_pipeline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Implementation of own functions

#Function that returns the accuracy rate, TNR, TPR and F1 score of a model over a given set
def CNN_test_rates(model, X_test, Y_test, CNN_class_name = 'CNN'):
    #Accuracy values are initialized to 1 and subtracted for each incorrect value
    accuracy, FNR, FPR = 1, 0, 0
    #Discount rates for incorrect data are computed for all metrics
    acc_discount_rate = 1 / Y_test.shape[0]
    FPR_total = len ( np.where( Y_test == [1, 0]) [0] ) / 2
    FPR_discount_rate = 1 / FPR_total
    FNR_total = len ( np.where( Y_test == [0, 1]) [0] ) / 2
    FNR_discount_rate = 1 / FNR_total
    #Predictions are approximated by rounding
    if ( CNN_class_name == 'CNN' ) or ( CNN_class_name == 'TL' ):
        Y_pred = np.round( model.predict(X_test, batch_size=10) )
        #For all predictions, the following metrics are calculated
        for i in range(0, Y_pred.shape[0]):
            #For convenience, an auxiliary variable is used
            y_pred, y_test = Y_pred[i], Y_test[i]            
            #If the prediction is different from the labeled data, it is discounted to the current rate
            if np.array_equal( y_pred, y_test ) is False:
                accuracy -= acc_discount_rate
                #If the difference is on a label that should be 0, the FPR is increased. Otherwise, FNR is increased
                if np.array_equal ( y_test, np.array( [1, 0] ) ): FPR += FPR_discount_rate
                else: FNR += FNR_discount_rate
    else:
        #Since CSNs DO NOT CALCULATE 1's OR 0's, the distance is first calculated and then transformed by a threshold to a list of 1's and 0's
        Y_distance = model.predict( [ X_test[0], X_test[1] ], batch_size=10) 
        Y_pred = CSN_dist2pred(Y_distance, dist = 0.5)
        #For all predictions, the following metrics are calculated
        for i in range(0, Y_pred.shape[0]):
            #For convenience, an auxiliary variable is used
            y_pred, y_test = Y_pred[i], Y_test[i]
            #If the prediction is different from the labeled data, it is discounted at the current rate
            if np.array_equal( y_pred, y_test ) is False:     
                accuracy -= acc_discount_rate
                #If the difference is on a label that should be 0, the FPR is increased. Otherwise, FNR is increased
                if np.array_equal ( y_test, np.array( [0] ) ): FPR += FPR_discount_rate
                else: FNR += FNR_discount_rate    
    #F1 score is calculated
    TNR, TPR = 1 - FNR, 1 - FPR
    F1 = ( TPR ) / ( TPR + 0.5*( FNR + FPR ) )
    return accuracy, TNR, TPR, F1

#Function to train or test a specific model
def CNN_train_test(data_dir, session, model, CNN_train_params_obj, best_val_acc = 0, destiny = 'modelo', model_name = 'generico',\
     CNN_class_name = 'CNN', train_bool = True, TL_model_name = 'VGG', acc_vec = False, return_model = False, ):
    #GPU session startup
    init_op = initialize_all_variables()
    #Training options are extracted
    batch_size = CNN_train_params_obj.batch_size
    epochs = CNN_train_params_obj.epochs
    verbose = CNN_train_params_obj.verbose
    shuffle = CNN_train_params_obj.shuffle
    patience = CNN_train_params_obj.patience
    min_epochs = CNN_train_params_obj.min_epochs
    #Load data for training/testing
    data_read_list = []
    data_read_list = [data_dir + '/' + data_name for data_name in os.listdir(data_dir) if data_name.endswith('dic.pickle')]
    #Files and directories are created to store the trained models
    destiny_folder = data_dir + '/' + destiny + '/'
    os.makedirs(destiny_folder , exist_ok = True)
    subfold = destiny_folder + model_name + '/'
    os.makedirs(subfold , exist_ok = True)
    best_model = subfold + 'best_model.h5'
    best_iter_model = subfold + 'best_iter_model.h5' 
    best_model_weights = subfold + 'best_model_weights.h5'
    CSN_bool = ( CNN_class_name == 'CSN')
    min_callbacks = defineCallBacks(best_model, min_epochs, CSN_bool)
    callbacks = defineCallBacks(best_model, patience, CSN_bool)
    txt_name = subfold + 'train_log.txt'
    epoch_adj = min_epochs * int(epochs > min_epochs)
    #For each batch of data iterate
    for data_file in data_read_list:
        #If Transfer Learning is chosen, the calculated features are loaded
        if CNN_class_name == 'TL':    
            X_train, Y_train, X_val, Y_val, X_test, Y_test, classes, img_shape = load_data(data_read_list[0], train = train_bool, TL = True)
            #If the process is a training process, the following code is executed
            if train_bool == True:
                #The following lines solve stability issues with TF/KERAS
                session.run(init_op)
                with session.as_default():
                    with session.graph.as_default():
                        #The model is trained with the minimum number of epochs entered, if the total number entered is greater than the minimum, it is retrained
                        history = model.fit(X_train, Y_train,
                                batch_size      = batch_size,
                                epochs          = min_epochs,
                                verbose         = verbose,
                                validation_data = (X_val, Y_val),
                                shuffle         = shuffle,
                                callbacks       = min_callbacks)
                        if epochs > min_epochs:
                            history = model.fit(X_train, Y_train,
                                    batch_size      = batch_size,
                                    epochs          = epochs - min_epochs,
                                    verbose         = verbose,
                                    validation_data = (X_val, Y_val),
                                    shuffle         = shuffle,
                                    callbacks       = callbacks)
                        val_acc_vector = np.array(history.history['val_acc'])
                        train_acc_vector = np.array(history.history['acc'])
                        #The maximum is found over the truncated vector
                        if val_acc_vector[np.where(val_acc_vector<train_acc_vector)[0]].shape[0] != 0 :
                            val_acc_max = np.max(val_acc_vector[np.where(val_acc_vector<train_acc_vector)[0]])
                            val_acc = history.history['val_acc'][np.where(val_acc_vector == val_acc_max)[0][0]]
                            train_acc = history.history['acc'][np.where(val_acc_vector == val_acc_max)[0][0]]
                        else: val_acc, train_acc = 0, 0
                        #Since the model is no longer occupied, it is discarded to free up memory
                        model = []
                        #If the accuracy in the validation set is greater than the historical accuracy of each configuration, the model is saved as the best one
                        if val_acc > best_val_acc :
                            best_val_acc = val_acc
                            best_modelo = load_model(best_model)
                            #The best model found and the pre-trained model used are saved                     
                            best_modelo.save(best_iter_model)
                            epoch_size = len(history.history['acc'])
                            print('The best accuracy found for the current network is: ' + str(best_val_acc))
                            write_string = 'Size of input images: ' + str(img_shape) + '\r\n' + \
                                            'Training batch size: ' + str(batch_size) + '\r\n' + \
                                            'Epoch of maximum validation accuracy:  ' + str(np.where(val_acc_vector == val_acc_max)[0][0] + 1 + epoch_adj) + '\r\n'+\
                                            'Training epochs: ' + str(epoch_size+ epoch_adj) + '\r\n' + \
                                            'Accuracy over training set: ' + str(train_acc) + '\r\n' + \
                                            'Accuracy over testing set: ' + str(best_val_acc) + '\r\n' + \
                                            'Transfer-Learning model: ' + TL_model_name
                            txt_file = open(txt_name,'w+')
                            txt_file.write(write_string)
                            txt_file.close()
                        else:  os.remove(best_model)
                K.clear_session()
                if acc_vec == True and return_model : return best_val_acc, subfold, destiny_folder, history.history['acc'], history.history['val_acc'], model
                elif acc_vec == False and return_model: return best_val_acc, subfold, destiny_folder, model
                else: return best_val_acc, subfold, destiny_folder
            #If the process is a testing process, the following code is executed
            else:
                #Network prediction metrics are calculated on the test data
                accuracy, TNR, TPR, F1 = CNN_test_rates(model, X_test, Y_test, CNN_class_name = CNN_class_name)
                #A txt file is saved with the calculated performance, as well as a dictionary in pickle format
                acc_dict = {'accuracy': accuracy, 'TNR': TNR, 'TPR': TPR, 'F1': F1}
                test_pickle_name = destiny_folder + '/test.pickle'
                with open(test_pickle_name, 'wb') as handle: pickle_dump(acc_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
                test_txt_name = destiny_folder + '/test.txt'
                test_txt_file = open(test_txt_name, 'w+')
                test_string = 'The performance metrics on the test set from the folder\r\n' + data_dir + '\r\nare as follows:\r\n' + \
                    'Accuracy: ' + str(accuracy) + '\r\n' + \
                    'TNR: ' + str(TNR) + '\r\n' + \
                        'TPR: ' + str(TPR) + '\r\n' + \
                            'F1: ' + str(F1)
                test_txt_file.write(test_string)
                test_txt_file.close() 
                return accuracy, TNR, TPR, F1  
        #If a CNN is chosen, the data is loaded directly
        elif CNN_class_name == 'CNN': 
            X_train, Y_train, X_val, Y_val, X_test, Y_test, _, _, _ = load_data(data_file, train = train_bool)
            img_shape = (X_train.shape[1], X_train.shape[2])
            #If the process is a training process, the following code is executed
            if train_bool == True:
                #The following lines solve stability issues with TF/KERAS
                session.run(init_op)
                with session.as_default():
                    with session.graph.as_default():
                        #The model is trained with the minimum number of epochs entered, if the total number entered is greater than the minimum, it is retrained
                        history = model.fit(X_train, Y_train,
                                batch_size      = batch_size,
                                epochs          = min_epochs,
                                verbose         = verbose,
                                validation_data = (X_val, Y_val),
                                shuffle         = shuffle,
                                callbacks       = min_callbacks)
                        if epochs > min_epochs:
                            history = model.fit(X_train, Y_train,
                                    batch_size      = batch_size,
                                    epochs          = epochs - min_epochs,
                                    verbose         = verbose,
                                    validation_data = (X_val, Y_val),
                                    shuffle         = shuffle,
                                    callbacks       = callbacks)
                        val_acc_vector = np.array(history.history['val_acc'])
                        train_acc_vector = np.array(history.history['acc'])
                        #The maximum is found over the truncated vector
                        if val_acc_vector[np.where(val_acc_vector<train_acc_vector)[0]].shape[0] != 0 :
                            val_acc_max = np.max(val_acc_vector[np.where(val_acc_vector<train_acc_vector)[0]])
                            val_acc = history.history['val_acc'][np.where(val_acc_vector == val_acc_max)[0][0]]
                            train_acc = history.history['acc'][np.where(val_acc_vector == val_acc_max)[0][0]]
                        else: val_acc, train_acc = 0, 0                            
                        #If the accuracy in the validation set is greater than the historical accuracy of each configuration, the model is saved as the best one
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_modelo = load_model(best_model)
                            best_modelo.save(best_iter_model)
                            best_modelo.save_weights(best_model_weights)
                            epoch_size = len(history.history['acc'])
                                
                            print('The best accuracy found for the current network is: ' + str(best_val_acc))
                            write_string =  'Size of input images: ' + str(img_shape) + '\r\n' + \
                                            'Training batch size: ' + str(batch_size) + '\r\n' + \
                                            'Epoch of maximum validation accuracy:  ' + str(np.where(val_acc_vector == val_acc_max)[0][0] + 1 + epoch_adj ) + '\r\n'+\
                                            'Training epochs: ' + str(epoch_size+ epoch_adj) + '\r\n' + \
                                            'Accuracy over training set: ' + str(train_acc) + '\r\n' + \
                                            'Accuracy over testing set: ' + str(best_val_acc) + '\r\n'
                            txt_file = open(txt_name,'w+')
                            txt_file.write(write_string)
                            txt_file.close()
                        else:  os.remove(best_model)
                K.clear_session()
                if acc_vec == True and return_model : return best_val_acc, subfold, destiny_folder, history.history['acc'], history.history['val_acc'], model
                if acc_vec == True and return_model == False : return best_val_acc, subfold, destiny_folder, history.history['acc'], history.history['val_acc']
                else: return best_val_acc, subfold, destiny_folder  
            #If the process is a testing process, the following code is executed
            else:
                #Network prediction metrics are calculated on the test data
                accuracy, TNR, TPR, F1 = CNN_test_rates(model, X_test, Y_test, CNN_class_name = CNN_class_name)
                #A txt file is saved with the calculated performance, as well as a dictionary in pickle format
                acc_dict = {'accuracy': accuracy, 'TNR': TNR, 'TPR': TPR, 'F1': F1}
                test_pickle_name = destiny_folder + '/test.pickle'
                with open(test_pickle_name, 'wb') as handle: pickle_dump(acc_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
                test_txt_name = destiny_folder + '/test.txt'
                test_txt_file = open(test_txt_name, 'w+')
                test_string = 'The performance metrics on the test set from the folder\r\n' + data_dir + '\r\nare as follows:\r\n' + \
                    'Accuracy: ' + str(accuracy) + '\r\n' + \
                    'TNR: ' + str(TNR) + '\r\n' + \
                        'TPR: ' + str(TPR) + '\r\n' + \
                            'F1: ' + str(F1)
                test_txt_file.write(test_string)
                test_txt_file.close() 
                return accuracy, TNR, TPR, F1     
        #If a CSN is chosen, the paired data is loaded
        else:
            CSN_X_train, CSN_Y_train, CSN_X_val, CSN_Y_val, CSN_X_test, CSN_Y_test, classes,_,_ = load_data(data_file, train = train_bool)
            img_shape = (CSN_X_train.shape[2], CSN_X_train.shape[3])
            CSN_Y_train = np.argmax(CSN_Y_train, axis = -1)
            CSN_Y_val = np.argmax(CSN_Y_val, axis = -1)
            #If the process is a training process, the following code is executed
            if train_bool == True:
                #The following lines solve stability issues with TF/KERAS
                session.run(init_op)
                with session.as_default():
                    with session.graph.as_default():
                        #The model is trained with the minimum number of epochs entered, if the total number entered is greater than the minimum, it is retrained
                        session.run(init_op)
                        history = model.fit([CSN_X_train[0], CSN_X_train[1]], CSN_Y_train,
                                batch_size      = batch_size,
                                epochs          = min_epochs,
                                verbose         = verbose,
                                validation_data = ([CSN_X_val[0], CSN_X_val[1]], CSN_Y_val),
                                shuffle         = shuffle,
                                callbacks       = min_callbacks)
                        if epochs > min_epochs:
                            history = model.fit([CSN_X_train[0], CSN_X_train[1]], CSN_Y_train,
                                    batch_size      = batch_size,
                                    epochs          = epochs - min_epochs,
                                    verbose         = verbose,
                                    validation_data = ([CSN_X_val[0], CSN_X_val[1]], CSN_Y_val),
                                    shuffle         = shuffle,
                                    callbacks       = callbacks)
                        val_acc_vector = np.array(history.history['val_CSN_accuracy'])
                        train_acc_vector = np.array(history.history['CSN_accuracy'])
                        #The maximum is found over the truncated vector
                        if val_acc_vector[np.where(val_acc_vector<train_acc_vector)[0]].shape[0] != 0 :
                            val_acc_max = np.max(val_acc_vector[np.where(val_acc_vector<train_acc_vector)[0]])
                            val_acc = history.history['val_CSN_accuracy'][np.where(val_acc_vector == val_acc_max)[0][0]]
                            train_acc = history.history['CSN_accuracy'][np.where(val_acc_vector == val_acc_max)[0][0]]
                        else: val_acc, train_acc = 0, 0
                        #If the accuracy in the validation set is greater than the historical accuracy of each configuration, the model is saved as the best one
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_modelo = load_model(best_model, custom_objects={ 'contrastive_loss': contrastive_loss, 'CSN_accuracy' : CSN_accuracy })
                            best_modelo.save(best_iter_model)
                            best_modelo.save_weights(best_model_weights)
                            epoch_size = len(history.history['CSN_accuracy'])
                                
                            print('The best accuracy found for the current network is: ' + str(best_val_acc))
                            write_string =  'Size of input images: ' + str(img_shape) + '\r\n' + \
                                            'Training batch size: ' + str(batch_size) + '\r\n' + \
                                            'Epoch of maximum validation accuracy:  ' + str(np.where(val_acc_vector == val_acc_max)[0][0] + 1 + epoch_adj ) + '\r\n'+\
                                            'Training epochs: ' + str(epoch_size+ epoch_adj) + '\r\n' + \
                                            'Accuracy over training set: ' + str(train_acc) + '\r\n' + \
                                            'Accuracy over testing set: ' + str(best_val_acc) + '\r\n'
                            txt_file = open(txt_name,'w+')
                            txt_file.write(write_string)
                            txt_file.close()
                        else:  os.remove(best_model)
                K.clear_session()     
                if acc_vec == True and return_model: return best_val_acc, subfold, destiny_folder, history.history['CSN_accuracy'], history.history['val_CSN_accuracy'], model
                elif acc_vec == True and return_model == False: return best_val_acc, subfold, destiny_folder, history.history['CSN_accuracy'], history.history['val_CSN_accuracy']
                else: return best_val_acc, subfold, destiny_folder  
            
            #If the process is a testing process, the following code is executed
            else:
                #Network prediction metrics are calculated on the test data
                with open(subfold + '/centroid_dic.pickle', 'rb') as handle: centroid_dict = pickle_load(handle)
                accuracy, TNR, TPR, F1 = CNN_test_rates(model, CSN_X_test, CSN_Y_test, CNN_class_name = 'CSN')
                #A txt file is saved with the calculated performance, as well as a dictionary in pickle format
                acc_dict = {'accuracy': accuracy, 'TNR': TNR, 'TPR': TPR, 'F1': F1}
                test_pickle_name = destiny_folder + '/test.pickle'
                with open(test_pickle_name, 'wb') as handle: pickle_dump(acc_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
                test_txt_name = destiny_folder + '/test.txt'
                test_txt_file = open(test_txt_name, 'w+')
                test_string = 'The performance metrics on the test set from the folder\r\n' + data_dir + '\r\nare as follows:\r\n' + \
                    'Accuracy: ' + str(accuracy) + '\r\n' + \
                    'TNR: ' + str(TNR) + '\r\n' + \
                        'TPR: ' + str(TPR) + '\r\n' + \
                            'F1: ' + str(F1)
                test_txt_file.write(test_string)
                test_txt_file.close() 
                return accuracy, TNR, TPR, F1   

#Function to train various CNN and CSN models, setting a range for the number of layers (and corresponding neurons for each) of the classification stage
def CNN_train_iterator(CNN_params_obj, CNN_train_params_obj,  img_shape, data_dir, destiny = 'modelo', model_name = 'generico',\
    class_layers_range = 5, class_neuron_range = [5, 100], last_check = True,  ntimes = 10, tf_seed = 1, acc_vec = False, autoencoder = False, GAP = False, delete_data = False):
    #If classification layers have been added, they are deleted from the input
    CNN_params_obj.classlayer_shape_clear()
    CNN_class_name = CNN_params_obj.CNN_class
    #A dictionary is created so as not to start from 0 each iteration
    iters_dict = {'class_layers_n' : 0,'neuron_permut_check_list' : [], 'ovrl_best_val_acc' : 0, 'layern_best_val_acc' : 0,\
         'best_ovrl_sub_fold' : '', 'best_layern_sub_fold' : '' }
    dict_dir = data_dir  + '/' + destiny + '/iter.pickle'
    best_layern_sub_fold = ''
    #If you do not want to check the above parameters, the default setting is
    if last_check == False:
        class_layers_n_init = 1
        neuron_permut_check_list = []
        ovrl_best_val_acc = 0
        layern_best_val_acc = 0
        best_ovrl_sub_fold = ''
        best_layern_sub_fold =  ''   
    #If it is desired to return to the previous parameters, the saved dictionary is loaded
    else:
        #The dictionary is loaded
        if os.path.isfile(dict_dir):
            with open(dict_dir, 'rb') as handle: iters_dict = pickle_load(handle)
        #The individual dictionary data are loaded
        class_layers_n_init = iters_dict [ 'class_layers_n' ]
        neuron_permut_check_list = iters_dict [ 'neuron_permut_check_list' ]
        ovrl_best_val_acc = iters_dict  [ 'ovrl_best_val_acc' ]        
        layern_best_val_acc = iters_dict [ 'layern_best_val_acc' ]
        best_ovrl_sub_fold = iters_dict [ 'best_ovrl_sub_fold' ]
        best_layern_sub_fold =  iters_dict ['best_layern_sub_fold'] 
    #Iteration on possible configurations for the classification network
    for class_layers_n in range(class_layers_n_init+1, class_layers_range +1):
        #If the range contains the number of iterations per layer, the permutation limits are changed
        if len(class_neuron_range) == 3: neuron_permut_list =  list ( permutations ( list( np.linspace( class_neuron_range[0], class_neuron_range[1], class_neuron_range[2], dtype = np.int32 ) )\
            * class_layers_n , class_layers_n ) )
        #If not, all iterations are created
        else: neuron_permut_list = list ( permutations ( list( range ( class_neuron_range[0], class_neuron_range[1] +1 ) ) * class_layers_n , class_layers_n ) )
        neuron_permut_list = list( dict.fromkeys( neuron_permut_list ) )
        neuron_permut_list = [item for item in neuron_permut_list if item not in neuron_permut_check_list]
        #For each layer list created it is iterated
        for layer_list in neuron_permut_list:
            #If classification layers have been added, they are deleted from the input
            CNN_params_obj.classlayer_shape_clear()
            #For each item in each list, the layer corresponding to the classification stage is added
            for layer_n in layer_list: CNN_params_obj.classlayer_shape_adder(layer_n)             
            #Iterate 'n' times to compensate for variability due to random initialization of weights
            val_acc_iter = 0
            #An input seed of the algorithm is instantiated
            set_random_seed(tf_seed)
            for i in range(ntimes):
                session, TL_model, CNN_class = fully_CNN_creator(CNN_params_obj, img_shape, autoencoder = autoencoder, GAP = GAP)
                model_name = str(layer_list)
                #If transfer learning is chosen, the features are calculated beforehand
                if CNN_class_name == 'TL':
                    #The training stage information is saved if requested
                    if not acc_vec:
                        val_acc, sub_fold, full_destiny = CNN_train_test(data_dir, session, CNN_class, CNN_train_params_obj, best_val_acc = val_acc_iter, destiny = destiny,\
                            model_name = model_name, CNN_class_name = CNN_class_name, train_bool = True, acc_vec = False)
                    else:
                        val_acc, sub_fold, full_destiny, acc_vector, val_acc_vector = CNN_train_test(data_dir, session, CNN_class, CNN_train_params_obj, best_val_acc = val_acc_iter, destiny = destiny,\
                            model_name = model_name, CNN_class_name = CNN_class_name, train_bool = True, acc_vec = True)
                        vector_dict = {'acc_vector': acc_vector, 'val_acc_vector': val_acc_vector, 'val_acc': val_acc}
                        #The dictionary containing the vector of iterations is saved
                        os.makedirs(sub_fold + 'vector_folder/', exist_ok=True)
                        with open( sub_fold + 'vector_folder/' + str(i) + '_iter_vector.pickle', 'wb') as handle: pickle_dump(vector_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
                #Otherwise, the model is trained directly
                else:
                    #The training stage information is saved if requested
                    if not acc_vec:
                        val_acc, sub_fold, full_destiny = CNN_train_test(data_dir, session, CNN_class, CNN_train_params_obj, best_val_acc = val_acc_iter, destiny = destiny,\
                            model_name = model_name, CNN_class_name = CNN_class_name, train_bool = True, acc_vec = False)
                    else:
                        val_acc, sub_fold, full_destiny, acc_vector, val_acc_vector = CNN_train_test(data_dir, session, CNN_class, CNN_train_params_obj, best_val_acc = val_acc_iter, destiny = destiny,\
                            model_name = model_name, CNN_class_name = CNN_class_name, train_bool = True, acc_vec = True)
                        vector_dict = {'acc_vector': acc_vector, 'val_acc_vector': val_acc_vector, 'val_acc': val_acc}
                        #The dictionary containing the vector of iterations is saved
                        os.makedirs(sub_fold + 'vector_folder/', exist_ok=True)
                        with open( sub_fold + 'vector_folder/' + str(i) + '_iter_vector.pickle', 'wb') as handle: pickle_dump(vector_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
                if val_acc > val_acc_iter: val_acc_iter = val_acc
            #Training results are plotted for each layer prior to elimination in case the corresponding performance does not exceed the current maximum
            inverse_sub_fold = sub_fold[::-1] 
            plot_destiny = sub_fold[0:len(sub_fold)-inverse_sub_fold[1:-1].index('/')-1] + 'vector_plots/' + sub_fold[len(sub_fold)-inverse_sub_fold[1:-1].index('/')-1:-1] + sub_fold[-1] 
            train_plot(sub_fold + 'vector_folder', plot_destiny)
            if val_acc_iter > layern_best_val_acc:
                layern_best_val_acc, best_layern_sub_fold = val_acc_iter, str(layer_list) 
                #The best performance value for the layer is updated
                iters_dict.update({'layern_best_val_acc' : layern_best_val_acc})
                iters_dict.update({'best_layern_sub_fold' : best_layern_sub_fold})
            elif val_acc_iter < layern_best_val_acc and delete_data: rmtree(sub_fold)
            #It is also checked whether the maximum performance of the layer is the maximum overall performance
            if layern_best_val_acc > ovrl_best_val_acc:
                ovrl_best_val_acc = layern_best_val_acc
                best_ovrl_sub_fold = best_layern_sub_fold
                #Best overall performance value is updated
                iters_dict.update({'ovrl_best_val_acc' : ovrl_best_val_acc})  
                #Best overall performance folder is updated
                iters_dict.update({'best_ovrl_sub_fold' : best_ovrl_sub_fold})
            #The current configuration is added to check for repeats
            iters_dict['neuron_permut_check_list'].append(layer_list)
            #The dictionary is saved
            with open(dict_dir, 'wb') as handle: pickle_dump(iters_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
        #The list is updated to check neurons, mainly to free up RAM
        iters_dict.update({'neuron_permut_check_list' : []})
        #If the best accuracy of the previous layer range is better than the global maximum, the latter is updated
        if layern_best_val_acc > ovrl_best_val_acc:
            ovrl_best_val_acc = layern_best_val_acc
            best_ovrl_sub_fold = best_layern_sub_fold
            #Best overall performance value is updated
            iters_dict.update({'ovrl_best_val_acc' : ovrl_best_val_acc})
            #Best overall performance folder is updated
            iters_dict.update({'best_ovrl_sub_fold' : best_ovrl_sub_fold})
        #All folders that do not correspond to the best-performing model are deleted
        check_folder_list = [full_destiny  + s for s in os.listdir(full_destiny)]
        folder_list = [item for item in check_folder_list if os.path.isdir(item)]
        iters_dict.update({'class_layers_n' : class_layers_n+1})
        for folder in folder_list:
            if folder != full_destiny + best_ovrl_sub_fold and not folder.endswith('vector_plots') and delete_data: rmtree(folder)    
        #The dictionary is updated again
        with open(dict_dir, 'wb') as handle: pickle_dump(iters_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)  
        #This value must be reset to re-evaluate layer by layer
        layern_best_val_acc = 0 
        
#Function to perform cross-validation training from the nfolds technique
def crossval_train_iterator(CNN_params_obj, CNN_train_params_obj,  img_shape, origin_folder, img_destiny, data_dir, train_val_rate = 0.8, test_rate = .1, nfolds = 5, destiny = 'modelo', model_name = 'generico',\
    class_layers_range = 5, class_neuron_range = [5, 100], last_check = True,  ntimes = 10, tf_seed = 1, acc_vec = False, autoencoder = False, GAP = False, pair_all = True, data_aug = True, keep_orig = True): 
    CNN_class_name = CNN_params_obj.CNN_class
    #The directory is read and the stratified nfolds are created to maintain the proportion of each class in the folders
    sources_list, class_name_list = multiclassparser(origin_folder)
    X, Y, class_dict = multiclass_preprocessing(sources_list, class_name_list, image_size = img_shape)
    X_test, Y_test = np.zeros(X.shape), np.zeros(Y.shape)
    #The data corresponding to the test are extracted so that they are completely split from the training/validation set (if a value greater than 0 and less than .3 was chosen)
    if test_rate > 0 and test_rate <.3:
        os.makedirs(img_destiny, exist_ok= True)
        #For each class, the test set is randomly drawn
        X_test, Y_test = np.zeros(0), np.zeros(0)
        X_new, Y_new = np.zeros(0), np.zeros(0)
        np.random.seed(5)
        class_n_img_names = []
        for class_n in class_dict['class_n_list']:
            #Valid file names are extracted for the folder corresponding to the class being stratified
            class_n_img_names  = class_n_img_names + [img_name for img_name in os.listdir(sources_list[class_n]) if img_name.endswith('.jpg') or img_name.endswith('.jpeg')]
            #Test data are chosen at random and extracted from the new set to be used for training and validation
            Y_n_indices = np.where(Y == class_n)[0]
            Y_n_test_indices = np.random.choice( Y_n_indices, size = int( np.round( test_rate*Y_n_indices.shape[0] ) ), replace= False )
            Y_n_train_indices = [Y_n_index  for Y_n_index in Y_n_indices if Y_n_index not in Y_n_test_indices]
            X_n_test, Y_n_test = X[ Y_n_test_indices, : ], Y[ Y_n_test_indices, : ]
            X_n_train, Y_n_train = X[ Y_n_train_indices, : ], Y[ Y_n_train_indices, : ]
            #If the sets have not been created yet, the test matrix and the new training matrix are created
            if X_test.shape[0] == 0: X_test, X_new, Y_test, Y_new = X_n_test, X_n_train, Y_n_test, Y_n_train
            else:
                X_test, X_new = np.concatenate ( ( X_test, X_n_test ), axis = 0 ), np.concatenate ( ( X_new, X_n_train ), axis = 0 )
                Y_test, Y_new = np.concatenate ( ( Y_test, Y_n_test ), axis = 0 ), np.concatenate ( ( Y_new, Y_n_train ), axis = 0 )
            #For repeatability and data checking purposes, images are saved for each set
            class_n_destiny = img_destiny + '/' + class_dict['class_name_list'] [class_n]
            os.makedirs(class_n_destiny + '/train', exist_ok= True)
            os.makedirs(class_n_destiny + '/test', exist_ok= True)
            for n in Y_n_test_indices:
                test_img_name = class_n_img_names[n]
                cv2.imwrite(class_n_destiny + '/test/' + test_img_name, X[n,:,:,:])     
            for n in Y_n_train_indices:
                train_img_name = class_n_img_names[n]
                cv2.imwrite(class_n_destiny + '/train/' + train_img_name, X[n,:,:,:])  
        #The values of training matrices are updated  
        X, Y = X_new, Y_new
    #The data are stratified given the number of nfolds entered
    skf = StratifiedKFold(n_splits=nfolds)
    best_val_acc, nfold_valacc_list, nfold_testacc_list, nfold_valmulticlass_list, nfold_testmulticlass_list, nfold_Cval_list, nfold_Ctest_list\
        = 0, [], [], [], [], [], []
    fold_idx = 0
    for train_index, val_index in skf.split(X, Y):
        fold_idx += 1
        #The training and validation sets are separated into the items for each nfold
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        #If augmentation was chosen
        if data_aug:
            #If the increased image is maintained, the Y_train value must be repeated
            new_X_train =  random_augment(X_train, img_destiny + '/train_aug_' + str(fold_idx), augment_list = ['flip', 'rot', 'bright', 'pad', 'color', 'zoom', 'noise' ],\
                bright_change = 20, color_variation = 5, keep_orig = keep_orig, overwrite = True,\
                double_aug = 'rand', color_rand = False, pad_rand = True, zoom_factor = 0.75, noise_var = 30, original_folder_name = '', input_array = True)
            if keep_orig:
                new_Y_train = np.zeros( (new_X_train.shape[0], 1 ) )
                for y_index in range(new_Y_train.shape[0]) : new_Y_train[y_index] = Y_train[ int(y_index/2) ].astype(int)
            else: new_Y_train = np.copy(Y_train)
            #X_train and Y_train values are updated
            X_train, Y_train = new_X_train, new_Y_train
        #If the model to be trained corresponds to a CSN, paired sets are created
        if CNN_class_name == 'CSN':
            new_X_train, new_Y_train, class_dict, CSN_class_dict = CSN_pair_creator(X_train, Y_train, class_dict, pair_all = pair_all)
            new_X_val, new_Y_val, class_dict, CSN_class_dict = CSN_pair_creator(X_val, Y_val, class_dict, pair_all = pair_all)
            new_X_test, new_Y_test, _, _ = CSN_pair_creator(X_test, Y_test, class_dict, pair_all = pair_all)
        else: new_X_train, new_Y_train,new_X_val, new_Y_val,new_X_test, new_Y_test, CSN_class_dict = X_train, Y_train, X_val, Y_val, X_test, Y_test, 0
        #The images are saved in the source folder with their corresponding names
        os.makedirs(data_dir, exist_ok = True)
        with open(data_dir + '/data_dic.pickle', 'wb') as handle: pickle_dump({'X_train' : new_X_train, 'X_val' : new_X_val, 'X_test': new_X_test,  'Y_train' : new_Y_train,\
        'Y_val' : new_Y_val, 'Y_test': new_Y_test, 'class_dict' : class_dict, 'CSN_class_dict' : CSN_class_dict}, handle, protocol=pickle_HIGHEST_PROTOCOL)
        ###TRAINING AND CROSS-VALIDATION
        #The seed to be used to achieve repeatability is loaded
        set_random_seed(tf_seed)
        #The training process is repeated 'n' times for each nfold
        best_val_fold = 0
        for train_n in range(ntimes):
            #The model is built and trained
            session, TL_model, CNN_class = fully_CNN_creator(CNN_params_obj, img_shape, autoencoder = autoencoder, GAP = GAP)
            val_acc, sub_folder, full_destiny, acc_vector, val_acc_vector, new_model = CNN_train_test(data_dir, session, CNN_class, CNN_train_params_obj, best_val_acc = best_val_fold, destiny = destiny,\
                            model_name = model_name + '/' + str(fold_idx), CNN_class_name = CNN_class_name, train_bool = True, acc_vec = True, return_model = True)
            vector_dict = {'acc_vector': acc_vector, 'val_acc_vector': val_acc_vector, 'val_acc': val_acc}
            #The dictionary containing the vector of iterations is saved
            os.makedirs(sub_folder + 'vector_folder/', exist_ok=True)
            with open( sub_folder + 'vector_folder/' + str(train_n) + '_iter_vector.pickle', 'wb') as handle: pickle_dump(vector_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)
            #If the validation value for the 'n' iteration is higher than the overall value for the corresponding nfold, 
            #the multiclass values are calculated and the best performance found is saved
            if val_acc > best_val_fold:
                best_val_fold = val_acc
                ovrl_best_folder = sub_folder
                best_model, session = best_model_finder( ovrl_best_folder, CNN_class_name = CNN_class_name)
                graph = get_default_graph()
                with graph.as_default():
                    #If the model to be trained corresponds to CSN
                    if CNN_class_name == 'CSN':
                        #The centroid and the feature prediction model are obtained
                        feat_mean, class_dict, semantic_net = CSN_centroid_finder(X_train, Y_train, best_model, class_dict)
                        #Then, the multiclass prediction is obtained by calculating the distance between the predicted features with respect to the training average
                        semantic_X_val = np.repeat(semantic_net.predict(X_val, batch_size = 32)[:,np.newaxis,:], len(class_dict['class_n_list']), axis = 1)
                        d_mat_val =  np.sqrt( np.sum(np.square(semantic_X_val-feat_mean), axis = 2) )
                        class_pred_val = 1-d_mat_val
                        class_pred_val[class_pred_val<0] = 0
                        multiclass_pred_val = np.argmax(class_pred_val, axis = 1)
                        #The previous process is now repeated for the test set
                        #The centroid and the feature prediction model are obtained
                        #The test accuracy is calculated with a fixed threshold so that a prediction 'd'<.5 implies equal pairs and 'd'>.5 different pairs
                        d_pred_test = best_model.predict( [new_X_test[0], new_X_test[1]], batch_size = 32 )
                        Y_pred_test = np.copy(d_pred_test)
                        Y_pred_test[d_pred_test>.5] = 0
                        Y_pred_test[d_pred_test<.5] = 1
                        #Then, the multiclass prediction is obtained by calculating the distance between the predicted features with respect to the training average
                        semantic_X_test = np.repeat(semantic_net.predict(X_test, batch_size = 32)[:,np.newaxis,:], len(class_dict['class_n_list']), axis = 1)
                        d_mat_test =  np.sqrt( np.sum(np.square(semantic_X_test-feat_mean), axis = 2) )
                        class_pred_test = 1-d_mat_test
                        class_pred_test[class_pred_test<0] = 0
                        multiclass_pred_test = np.argmax(class_pred_test, axis = 1)
                        #The global accuracy values are calculated on the test and multiclass on the same group
                        # if the trained model corresponds to a CNN they are equivalent) and validation
                        test_acc = accuracy_score(Y_pred_test, new_Y_test)
                        test_multiclass_acc = accuracy_score(multiclass_pred_test, Y_test)
                        val_multiclass_acc = accuracy_score(multiclass_pred_val, Y_val)
                    #If the model to be trained corresponds to CNN
                    else:
                        class_pred_val = best_model.predict(X_val)
                        multiclass_pred_val = np.argmax(class_pred_val, axis = 1)
                        class_pred_test = best_model.predict(X_test)
                        multiclass_pred_test = np.argmax(class_pred_test, axis = 1)
                        #The global accuracy values are calculated on the test and multiclass on the same group
                        # if the trained model corresponds to a CNN they are equivalent) and validation
                        test_acc = accuracy_score(multiclass_pred_test, Y_test)
                        test_multiclass_acc = test_acc
                        val_multiclass_acc = accuracy_score(multiclass_pred_val, Y_val)
                #Confusion matrices are calculated for the trained model given its predictions for the validation and test sets
                C_val = confusion_matrix(Y_val, multiclass_pred_val).astype(float) 
                C_test = confusion_matrix(Y_test, multiclass_pred_test).astype(float) 
                for class_n in range(C_val.shape[0]): C_val[class_n,:] = C_val[class_n, :] / np.sum(C_val[class_n], axis = 0)
                for class_n in range(C_test.shape[0]): C_test[class_n,:] = C_test[class_n, :] / np.sum(C_test[class_n], axis = 0)
        #The training curves are plotted for the respective fold
        inverse_sub_fold = sub_folder[::-1] 
        plot_destiny = sub_folder[0:len(sub_folder)-inverse_sub_fold[1:-1].index('/')-1] + 'vector_plots/' + sub_folder[len(sub_folder)-inverse_sub_fold[1:-1].index('/')-1:-1] + sub_folder[-1] 
        train_plot(sub_folder + 'vector_folder', plot_destiny)
        #The value of the best fold accuracy is added to the list, in addition to adding the metrics for each class
        nfold_valacc_list.append(best_val_fold) 
        nfold_valmulticlass_list.append(val_multiclass_acc)
        nfold_testacc_list.append(test_acc) 
        nfold_testmulticlass_list.append(test_multiclass_acc)
        nfold_Cval_list.append(C_val)
        nfold_Ctest_list.append(C_test)
        #If the accuracy of the respective fold exceeds the overall best, the old values are replaced
        if best_val_fold > best_val_acc: 
            best_fold_idx = fold_idx
            best_val_acc = best_val_fold
            best_val_multiclass = nfold_valmulticlass_list[best_fold_idx-1]
            best_test_acc = nfold_testacc_list[best_fold_idx-1]
            best_test_multiclass = nfold_testmulticlass_list[best_fold_idx-1]
    #The statistics of the model are calculated
    nfold_valacc_avg = np.mean(nfold_valacc_list)
    nfold_valacc_sd = np.std(nfold_valacc_list)
    nfold_testacc_avg = np.mean(nfold_testacc_list)
    nfold_testacc_sd = np.std(nfold_testacc_list)
    nfold_valmulticlass_avg = np.mean(nfold_valmulticlass_list)
    nfold_valmulticlass_sd = np.std(nfold_valmulticlass_list)
    nfold_testmulticlass_avg = np.mean(nfold_testmulticlass_list)
    nfold_testmulticlass_sd = np.std(nfold_testmulticlass_list)
    #The iteration data on the nfolds are stored in a dictionary
    nfold_dict_name = data_dir + '/' +  destiny + '/' + model_name + '/nfold_data.pickle'
    nfold_dict = {'nfolds' : nfolds, 'train_val_rate' : train_val_rate, 'test_rate' : test_rate, 'pair_all' : pair_all, 'data_aug' : data_aug, 'class_dict': class_dict,\
        'best_val_acc' : best_val_acc, 'best_val_multiclass': best_val_multiclass, 'best_test_acc' : best_test_acc, 'best_test_multiclass' : best_test_multiclass,\
            'best_fold' : best_fold_idx, 'val_acc_stats' : [nfold_valacc_avg, nfold_valacc_sd], 'test_acc_stats' : [nfold_testacc_avg, nfold_testacc_sd], \
            'val_multiclass_stats' : [nfold_valmulticlass_avg, nfold_valmulticlass_sd], 'test_multiclass_stats' : [nfold_testmulticlass_avg, nfold_testmulticlass_sd],\
                'val_acc_list': nfold_valacc_list, 'test_acc_list':nfold_testacc_list,  'val_multiclass_list': nfold_valmulticlass_list, 'test_multiclass_list':nfold_testmulticlass_list, \
                    'Conf_val' : nfold_Cval_list, 'Conf_test' : nfold_Ctest_list, 'CNN_params': CNN_params_obj, 'CNN_train_params' : CNN_train_params_obj, 'img_shape' : img_shape}
    with open(nfold_dict_name, 'wb') as handle: pickle_dump(nfold_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)  
    return nfold_dict

#Function that extracts statistics from a main folder where the data obtained from the cross validation is located,
#and returns the best value obtained for validation and testing
def crossval_stat_calc(origin_folder, stat_save_destiny, CNN_class_name = 'CNN', Y_val = np.array([]), nfolds = 5):
    #All names matching the dictionary format generated in cross validation are found and appended
    dict_names = [os.path.join(dirpath, f)
    for dirpath, _, files in os.walk(origin_folder)
    for f in files if f.endswith('nfold_data.pickle')]
    #For each dictionary, the data contained are extracted
    best_avg_acc = 0
    for dict_name in dict_names:
        with open(dict_name, 'rb') as handle: nfold_data_dict = pickle_load(handle)
        #The average value of the iteration is retrieved and compared with the best accumulated value
        nfold_valacc_avg = nfold_data_dict['val_multiclass_stats'][0]
        print('nfold_val_acc_avg'), print(nfold_valacc_avg), print('best_avg_acc'), print(best_avg_acc)
        best_test_acc = 0
        if nfold_valacc_avg > best_avg_acc:
            #The values for the analysis for each nfold are reset
            best_avg_acc = nfold_valacc_avg
            #The best validation (and subsequently testing) value for the current nfold is found
            best_val_nfold,  best_val_nfold_idx = np.max(nfold_data_dict['val_multiclass_list']),np.argmax(nfold_data_dict['val_multiclass_list'])
            best_test_nfold = nfold_data_dict['test_multiclass_list'][best_val_nfold_idx]
            print("BEST", nfold_data_dict['best_test_multiclass'])
            input("A")
            #If the performance on the current test set is better than the overall performance, the old values are replaced
            if best_test_nfold>best_test_acc:
                print('best_validation_infold',best_val_nfold)
                best_val_acc, best_test_acc = best_val_nfold, best_test_nfold
                best_fold_name = dict_name
                best_confusionmat_test = nfold_data_dict['Conf_test'][best_val_nfold_idx]
                best_confusionmat_val = nfold_data_dict['Conf_val'][best_val_nfold_idx]
                #The best Neural Network model found is loaded
                best_fold_name = best_fold_name.rsplit('\\', 1)[:-1] [0] + '/' + str(best_val_nfold_idx + 1)
                best_val_stats = nfold_data_dict['val_multiclass_stats']
                #The average values of the confusion matrices for validation are obtained and stored, in addition to the best validation and test values
                nfold_confusionmat_val = nfold_data_dict['Conf_val']
                confusionmat_val_avg, confusionmat_val_sd = np.mean( nfold_confusionmat_val, axis = 0 ), np.std(nfold_confusionmat_val, axis = 0)
                if Y_val.shape[0] == 0:
                    conf_matrix('', '', '', '', feat_mean = [], class_names = ['Lactuca virosa', 'Grass', 'Trifolium repens'], destiny = stat_save_destiny + '/conf_mats/nfold', show_bool = False,\
                        pred = False, C_in = [confusionmat_val_avg, confusionmat_val_sd], title = 'N-fold validation data')
                else:
                    conf_matrix('', '', '', Y_val, feat_mean = [], class_names = ['Lactuca virosa', 'Grass', 'Trifolium repens'], destiny = stat_save_destiny + '/conf_mats/nfold', show_bool = False,\
                        pred = False, C_in = [confusionmat_val_avg, confusionmat_val_sd], title = 'N-fold validation data', CI_calc = True, nfolds = nfolds)
                conf_matrix('', '', '', '', feat_mean = [], class_names = ['Lactuca virosa', 'Grass', 'Trifolium repens'], destiny = stat_save_destiny + '/conf_mats/bestfoldval', show_bool = False,\
                    pred = False, C_in = best_confusionmat_val, title = 'Best fold validation data')
                conf_matrix('', '', '', '', feat_mean = [], class_names = ['Lactuca virosa', 'Grass', 'Trifolium repens'], destiny = stat_save_destiny + '/conf_mats/bestfoldtest', show_bool = False,\
                    pred = False, C_in = best_confusionmat_test, title = 'Best fold test data')
                copyfile(dict_name, stat_save_destiny + '/best_dict.pickle')
                plt.close('all')
            print('best_test' + str(best_test_acc))
            print(best_fold_name)
    #The folder containing the best trained model is copied to the destination folder and then the complete model is returned
    copy_tree(best_fold_name, stat_save_destiny + '/best_model/')
    if CNN_class_name == 'CNN' or CNN_class_name == 'TL': model = load_model(best_fold_name + '/best_iter_model.h5')
    elif CNN_class_name == 'CSN':model = load_model(best_fold_name + '/best_iter_model.h5', custom_objects={ 'contrastive_loss': contrastive_loss , 'CSN_accuracy' : CSN_accuracy})
    return model
    
#Function to find and load the best model found on a specific training scheme
def best_model_finder(iter_dir, CNN_class_name = 'CNN', init = False):
    session = tensor_init() if init else []
    #The list of files and directories contained in the specified folder is created and the pickle dictionary containing the iteration data is loaded
    dict_list = [iter_dir + '/' + file_name for file_name in os.listdir(iter_dir) if file_name.endswith('iter.pickle')]
    if dict_list: 
        with open(dict_list[0], 'rb') as handle: iters_dict = pickle_load(handle)
        best_ovrl_sub_fold = iters_dict ['best_ovrl_sub_fold']
        model_name = iter_dir + '/' + best_ovrl_sub_fold + '/best_iter_model.h5'
    else:
        best_ovrl_sub_fold = iter_dir
        model_name = best_ovrl_sub_fold + '/best_iter_model.h5'
    #The address of the best model is obtained and the latter is loaded, and returned
    if CNN_class_name == 'CNN' or CNN_class_name == 'TL': model = load_model(model_name)
    elif CNN_class_name == 'CSN': model = load_model(model_name, custom_objects={ 'contrastive_loss': contrastive_loss , 'CSN_accuracy' : CSN_accuracy})
    return model, session

#Función para graficar los resultados de entrenamiento para los datos una arquitectura específica
def train_plot(train_vector_folder, plot_destiny):
    
    #Se revisa que exista al menos un archivo que contenga información de iteraciones en la carpeta proporcionada, en caso contrario se levanta un error
    vector_files = [train_vector_folder + '/' +  s for s in os.listdir(train_vector_folder) if s.endswith('_iter_vector.pickle')]
    num_vector_files = [None]*len(vector_files)
    
    #Se corrige el orden de los archivos por orden númerico
    for vector_name in vector_files:
        slash_index = len(vector_name)-vector_name[::-1].index('/')
        num_idx = int(vector_name[slash_index:slash_index+vector_name[slash_index:-1].index('_')])
        num_vector_files[num_idx] = vector_name
        
    vector_files = num_vector_files
    if not len(vector_files): raise ValueError('Ingrese una carpeta con archivos válidos')
    
    os.makedirs(plot_destiny, exist_ok=True)
    #Se leen todos los archivos y se encuentra la iteración de mayor rendimiento 
    best_val_acc_idx = 0
    best_val_acc = 0
    val_acc_vector_list = []
    train_acc_vector_list = []
    for file_idx, vector_file in enumerate(vector_files):
        with open(vector_file , 'rb') as handle: train_vector_dict = pickle_load(handle)
        val_acc_vector_list.append(train_vector_dict['val_acc_vector'])
        train_acc_vector_list.append(train_vector_dict['acc_vector'] )
        val_acc = train_vector_dict['val_acc']
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_idx = file_idx
    
    #Se dibujan los resultados en un gráfico semi-logarítmico comparando cada iteración con la mejor
    for vector_idx, val_acc_vector in enumerate(val_acc_vector_list):
        
        best_train_acc_vector = train_acc_vector_list[best_val_acc_idx]
        train_acc_vector = train_acc_vector_list[vector_idx]
        
        best_val_acc_vector = val_acc_vector_list[best_val_acc_idx]
        
        #Se crean y estilan los ejes a ocupar para el conjunto de entrenamiento
        train_accuracy_fig, train_accuracy_ax = plt.subplots()
        train_accuracy_fig.set_size_inches(16, 9) 
        train_accuracy_ax.set_title('Resultados de accuracy de entrenamiento')
        train_accuracy_ax.set_xlabel('Epochs de entrenamiento')
        train_accuracy_ax.set_ylabel('Valor sobre conjunto de entrenamiento')
        
        val_accuracy_fig, val_accuracy_ax = plt.subplots()
        val_accuracy_fig.set_size_inches(16, 9) 
        val_accuracy_ax.set_title('Resultados de accuracy de validación')
        val_accuracy_ax.set_xlabel('Epochs de validación')
        val_accuracy_ax.set_ylabel('Valor sobre conjunto de validación')
        
        trainval_accuracy_fig, trainval_accuracy_ax = plt.subplots()
        trainval_accuracy_fig.set_size_inches(16, 9) 
        trainval_accuracy_ax.set_title('Resultados de accuracy de entrenamiento/validación')
        trainval_accuracy_ax.set_xlabel('Epochs de entrenamiento/validación')
        trainval_accuracy_ax.set_ylabel('Valor sobre conjuntos de entrenamiento/validación')
        
        #Se decide si el valor que se está graficando es el mejor o no para agregar el mejor gráfico al resto
        if vector_idx == best_val_acc_idx:
            #Se renderizan los resultados del entrenamiento en escala semi-logarítmica
            epochs = np.linspace(0, len(train_acc_vector)-1, len(train_acc_vector))
            train_accuracy_line, = train_accuracy_ax.semilogx(epochs, train_acc_vector)
            train_accuracy_line.set_label('Mejor curva de entrenamiento')
            train_accuracy_ax.set_xlim( [1, len(train_acc_vector)] )
            train_accuracy_ax.set_ylim( [0, 1] )
            train_accuracy_ax.legend()
            
            for x in range(1, int( np.log10( len( val_acc_vector ) ) ) + 1 ):
                train_accuracy_ax.axvline(x = 10**x, color = 'k', linestyle= '--')
            #train_accuracy_ax.axvline(x = 100, color = 'k', linestyle= '--')
            
            
            val_accuracy_line, = val_accuracy_ax.semilogx(epochs, val_acc_vector)
            val_accuracy_line.set_label('Mejor curva de validación')
            val_accuracy_ax.set_xlim( [1, len(val_acc_vector)] )
            val_accuracy_ax.set_ylim( [0, 1] )
            val_accuracy_ax.legend()
            
            for x in range(1, int( np.log10( len( val_acc_vector ) ) ) + 1 ):
                val_accuracy_ax.axvline(x = 10**x, color = 'k', linestyle= '--')
            #val_accuracy_ax.axvline(x = 100, color = 'k', linestyle= '--')
            
            valtrain_accuracy_line, = trainval_accuracy_ax.semilogx(epochs, val_acc_vector)
            valtrain_accuracy_line.set_label('Curva de validación')
            trainval_accuracy_line, = trainval_accuracy_ax.semilogx(epochs, train_acc_vector)
            trainval_accuracy_line.set_label('Curva de entrenamiento')
            
            trainval_accuracy_ax.set_xlim( [1, len(val_acc_vector)] )
            trainval_accuracy_ax.set_ylim( [0, 1] )
            trainval_accuracy_ax.legend()  
            
            for x in range(1, int( np.log10( len( val_acc_vector ) ) ) + 1 ):
                trainval_accuracy_ax.axvline(x = 10**x, color = 'k', linestyle= '--')
            #trainval_accuracy_ax.axvline(x = 100, color = 'k', linestyle= '--')
            
                               
        else:
            epochs = np.linspace(1, len(train_acc_vector), len(train_acc_vector))
            epochs_best = np.linspace(1, len(best_train_acc_vector), len(best_train_acc_vector))
            train_accuracy_line, = train_accuracy_ax.semilogx(epochs, train_acc_vector)
            train_accuracy_line.set_label('Curva de entrenamiento')
            best_train_accuracy_line, = train_accuracy_ax.semilogx(epochs_best, best_train_acc_vector)
            best_train_accuracy_line.set_label('Mejor curva de entrenamiento')
            train_accuracy_ax.legend()
            train_accuracy_ax.set_xlim( [1, len(train_acc_vector)] )
            train_accuracy_ax.set_ylim( [0, 1] )
            
            for x in range(1, int( np.log10( len( val_acc_vector ) ) ) + 1 ):
                train_accuracy_ax.axvline(x = 10**x, color = 'k', linestyle= '--')
                val_accuracy_ax.axvline(x = 10**x, color = 'k', linestyle= '--')
            
            val_accuracy_line, = val_accuracy_ax.semilogx(epochs, val_acc_vector)
            val_accuracy_line.set_label('Curva de validación')
            best_val_accuracy_line, = val_accuracy_ax.semilogx(epochs_best, best_val_acc_vector)
            best_val_accuracy_line.set_label('Mejor curva de validación')
            val_accuracy_ax.set_xlim( [1, len(val_acc_vector)] )
            val_accuracy_ax.set_ylim( [0, 1] )
            val_accuracy_ax.legend()  
            #val_accuracy_ax.axvline(x = 10, color = 'k', linestyle= '--')
            #val_accuracy_ax.axvline(x = 100, color = 'k', linestyle= '--')
            
            
            valtrain_accuracy_line, = trainval_accuracy_ax.semilogx(epochs, val_acc_vector)
            valtrain_accuracy_line.set_label('Curva de validación')
            trainval_accuracy_line, = trainval_accuracy_ax.semilogx(epochs, train_acc_vector)
            trainval_accuracy_line.set_label('Curva de entrenamiento')
            
            trainval_accuracy_ax.set_xlim( [1, len(val_acc_vector)] )
            trainval_accuracy_ax.set_ylim( [0, 1] )
            trainval_accuracy_ax.legend()  
            for x in range(1, int( np.log10( len( val_acc_vector ) ) ) + 1 ):
                trainval_accuracy_ax.axvline(x = 10**x, color = 'k', linestyle= '--')
            #trainval_accuracy_ax.axvline(x = 100, color = 'k', linestyle= '--')
        
        train_accuracy_fig.savefig(plot_destiny + '/' + str(vector_idx) + '_iteration_train.pdf')
        val_accuracy_fig.savefig(plot_destiny + '/' + str(vector_idx) + '_iteration_val.pdf')
        trainval_accuracy_fig.savefig(plot_destiny + '/' + str(vector_idx) + '_iteration_trainval.pdf')
        plt.close('all')
    
#Función para generar la matriz de confusión y entregar alguna otra estadística que se estime conveniente    
def conf_matrix(model, model_type, X, Y, feat_mean = [], class_names = ['grass', 'trifolium repens', 'lactuca virosa'], destiny = '', show_bool = False, pred = True, C_in = '', title = '', CI_calc = False, nfolds = 5):
    
    #Si se pide predecir se calcula la matriz de confusión, si no, se toma el valor de C
    if pred:
        #Si el modelo ingresado corresponde a una CNN, se crea la matriz de confusión
        if model_type == 'CNN': Y_pred = np.reshape(np.argmax(model.predict(X, batch_size = 8), axis = 1), (X.shape[0], 1))
        #Si el modelo es una CSN, se calculan primero las características y luego se clasifica cada dato
        elif model_type == 'CSN':
            class_n_list = list(set(np.ravel(Y).tolist()))
            sem_model = Sequential()
            feat_model = Sequential()
            for idx, layer in enumerate(model.layers):
                if isinstance(layer,Model):#not isinstance(layer, Flatten) and not isinstance(layer, Lambda) and not isinstance(layer, Dense):
                    for in_layer in layer.layers:
                        if not isinstance(in_layer, Flatten) and not isinstance(in_layer, Lambda) and not isinstance(in_layer, Dense): 
                            sem_model.add(in_layer)
                            feat_model.add(in_layer)
                        elif isinstance(in_layer, Dense) or isinstance(in_layer, Flatten): feat_model.add(in_layer)
            semantic_X = np.repeat(feat_model.predict(X, batch_size = 4)[:,np.newaxis,:], len(class_n_list), axis = 1)
            d_mat =  np.sqrt( np.sum(np.square(semantic_X-feat_mean), axis = 2) )
            pred_mat = 1-d_mat
            Y_pred = np.argmax(pred_mat, axis = 1)
        #Se calcula la matriz de confusión y se normalizan las filas para la cantidad relativa de aciertos
        C = confusion_matrix(Y, Y_pred).astype(float) 
    
    #Si el valor ingresado corresponde a una lista
    if type(C_in) == list: C , C_sd = C_in[0], C_in[1]
    elif type(C_in) != list and not pred: C, C_sd = C_in, np.zeros(C_in.shape)
    #Se normaliza el valor para guardar la matriz normalizada y por número de ocurrencia
    C_norm = np.copy(C)
    C_sd_norm = np.copy(C_sd)
    for class_n in range(C_norm.shape[0]):
        C_norm[class_n,:] = C_norm[class_n, :] / np.sum(C_norm[class_n], axis = 0)
        C_sd_norm[class_n,:] = C_sd[class_n, :] / np.sum(C_norm[class_n], axis = 0)
        if CI_calc:
            C_sd_norm[class_n,:] = np.sqrt( C_norm[class_n, :] * (1-C_norm[class_n, :]) / ( nfolds * np.where(Y == class_n)[0].shape[0] ) ) / np.sum(C_norm[class_n], axis = 0) 
    #Se crea la imagen de la matriz de confusión calculada SIN NORMALIZAR
    C_fig, C_ax = plt.subplots()
    C_fig.set_size_inches(16, 9) 
    if title: C_ax.set_title(title, fontsize = 25, fontweight = 'bold')
    C_ax.matshow(C, cmap = 'Purples', interpolation='none',  vmin=-0.1, vmax=1)
    C_ax.set_xlabel('Predicted', fontsize = 18, fontweight = 'bold', y = -.1)
    C_ax.set_ylabel('Actual', fontsize = 18, fontweight = 'bold')
    plt.gca().xaxis.tick_bottom()
    C_ax.set_xticklabels(['']+class_names)
    plt.yticks(rotation = 90)
    C_ax.set_yticklabels(['']+class_names)
    C_ax.tick_params(axis=u'both', which=u'both',length=0, labelbottom = True, bottom=False, top = False, labeltop=False, labelsize = 12)
    #C_ax.xaxis.labelpad = 8
    #C_ax.yaxis.labelpad = 8

    for (i, j), z in np.ndenumerate(C):
        C_sd_ij = C_sd[i,j] if type(C_in) == list else 0
        text = ('{:0.1f}'.format(z)) + u"\u00B1" + ('{:0.1f}'.format(C_sd_ij)) if type(C_in) == list else ('{:0.2f}'.format(z)) 
        C_ax.text(j, i, text, ha='center', va='center', fontsize = 20, color = 'black' if z <.5 else 'white') 
        
    #Si se ingresó una dirección para guardar la imagen se hace eso, si no, solo se muestra la matriz
    if destiny:
        os.makedirs(destiny, exist_ok= True)
        plt.savefig(destiny + '/C.pdf', bbox_inches='tight', pad_inches=0.5)
    else: 
        if show_bool: plt.show(C_fig)
    
    #Se crea la imagen de la matriz de confusión calculada AHORA NORMALIZADA
    C_norm_fig, C_norm_ax = plt.subplots()
    C_norm_fig.set_size_inches(16, 9) 
    if title: C_norm_ax.set_title(title, fontsize = 25, fontweight = 'bold')
    C_norm_ax.matshow(C_norm, cmap = 'Purples', interpolation='none',  vmin=-0.1, vmax=1)
    C_norm_ax.set_xlabel('Predicted', fontsize = 18, fontweight = 'bold', y = -.1)
    C_norm_ax.set_ylabel('Actual', fontsize = 18, fontweight = 'bold')
    plt.gca().xaxis.tick_bottom()
    C_norm_ax.set_xticklabels(['']+class_names)
    plt.yticks(rotation = 90)
    C_norm_ax.set_yticklabels(['']+class_names, va = 'center' )
    C_norm_ax.tick_params(axis=u'both', which=u'both',length=0, labelbottom = True, bottom=False, top = False, labeltop=False, labelsize = 16)
    C_norm_ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    C_norm_ax.set_yticks(np.arange(-.5, 3, 1), minor=True)

    # Gridlines based on minor ticks
    C_norm_ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    #C_norm_ax.xaxis.labelpad = 8
    #C_norm_ax.yaxis.labelpad = 8

    for (i, j), z in np.ndenumerate(C_norm):
        C_sd_ij = C_sd_norm[i,j]*100 if type(C_in) == list else 0
        text = ('{:0.1f}'.format(z*100)) + u"\u00B1" + ('{:0.1f}'.format(C_sd_ij)) +('%') if type(C_in) == list else ('{:0.2f}'.format(z*100) + ('%')) 
        C_norm_ax.text(j, i, text, ha='center', va='center', fontsize = 20, color = 'black' if z <.5 else 'white')
        
    #Si se ingresó una dirección para guardar la imagen se hace eso, si no, solo se muestra la matriz
    if destiny:
        os.makedirs(destiny, exist_ok= True)
        plt.savefig(destiny + '/C_norm.pdf', bbox_inches='tight', pad_inches=0.5)
    else: 
        if show_bool: plt.show(C_norm_fig)
    
    return C, C_norm
    
#Función para probar el efecto del cambio de proporción de datos de entrenamiento en el rendimiento de una red (o modelo SVM) específica            
def set_size_iterator(origin_folder_list, destiny_dir, CNN_params_obj, CNN_train_params_obj,  img_shape,\
     train_ratio_limits = [0.05, 0.95], train_num = 2, val_test_ratio = 0.5,\
         destiny = 'modelo', model_name = 'generico', class_layers_range = 1,\
              class_neuron_range = [5, 5], last_check = False,  ntimes = 1, tf_seed = 1,\
                  lbp_r = 1, lbp_points = 8, \
            hog_block_x = 3, hog_block_y = 3, hog_pixels_x = 16, hog_pixels_y = 16, hog_orientations = 4,\
                SVM_kernel_list = ['linear', 'poly', 'rbf'], SVM_gamma_list = [0.01, 10, 0.01], SVM_degrees = [2, 3, 4 ], \
            SVM_max_iter = 1000, best_clf_metric = 'avg',  SVM_clean = True, SVM_norm = True,  SVM_sfs_list = [5, 25, 1], SVM_pca_list = [5, 15, 1], SVM_last_check = True):
    
    #Se crea el vector de límites para crear los conjuntos de entrenamiento, prueba y validación
    limit_vec = np.around( np.linspace(train_ratio_limits[0], train_ratio_limits[1], train_num ), decimals = 2)
    train_start_bool = True  
    
    #Se comprueba si existe el diccionario de train_rates probados y se eliminan los elementos repetidos
    #OJO: LA ADICIÓN DEL CNN_CLASS_NAME PERMITE GUARDAR DATOS DE CNN Y CSN POR SEPARADO
    if isinstance(CNN_params_obj, str): CNN_class_name = 'SVM'
    
    else: CNN_class_name = CNN_params_obj.CNN_class
    
    CNN_class_name_destiny = destiny + '_' +  CNN_class_name
    if os.path.isfile(destiny_dir + '/' + CNN_class_name_destiny + '/ratio_dic.pickle'):

        with open(destiny_dir + '/' + CNN_class_name_destiny + '/ratio_dic.pickle', 'rb') as handle: ratio_dic = pickle_load(handle)
            
        train_rates_done = ratio_dic [ 'train_ratio_list' ]
        limit_vec = [item for item in limit_vec if item not in train_rates_done]
        train_start_bool = ratio_dic['bool']
    
    #Se itera para cada razón de entrenamiento para encontrar los valores que se obtiene entrenando el modelo especificado
    for train_ratio in limit_vec:
        
        print(train_ratio)
        #Para el caso de entrenar SVM y no red neuronal se ejecuta el siguiente código
        if CNN_class_name == 'SVM':
            
            SVM_class_name_destiny = destiny_dir + '/' + CNN_class_name_destiny
            X_train, Y_train, X_val, Y_val, X_test, Y_test = feature_set_generator(origin_folder_list, destiny_dir, train_ratio, val_test_ratio, lbp_r = lbp_r, lbp_points = lbp_points, \
            hog_block_x = hog_block_x, hog_block_y = hog_block_y, hog_pixels_x = hog_pixels_x, hog_pixels_y = hog_pixels_y, hog_orientations = hog_orientations, img_shape = img_shape)
            best_clf, best_X_test = SVM_set_iterator (destiny_dir, SVM_class_name_destiny + '/clf', SVM_kernel_list = SVM_kernel_list,\
                SVM_gamma_list = SVM_gamma_list, SVM_degrees = SVM_degrees, \
            SVM_max_iter = SVM_max_iter, best_clf_metric = best_clf_metric,  SVM_clean = SVM_clean, SVM_norm = SVM_norm,\
                SVM_sfs_list = SVM_sfs_list, SVM_pca_list = SVM_pca_list, SVM_last_check = SVM_last_check)
            
            accuracy, TNR, TPR, F1 = SVM_train_test( X_train, Y_train, X_val, Y_val, best_X_test, Y_test, train = False, SVM_max_iter = SVM_max_iter, clf = best_clf)
            
        #Para redes neuronales el que sigue  
        else:
            
            if CNN_class_name == 'TL':
                
                #Se crea el conjunto de entrenamiento y validación/prueba para cada caso
                sources_list, class_list = multiclassparser(origin_folder_list)
                X, Y, class_dict = multiclass_preprocessing(sources_list, class_list)
                multiclass_set_creator(X, Y, class_dict, destiny_dir,  train_test_rate = train_ratio, test_val_rate = val_test_ratio, TL = True, \
                     TL_model_name = 'VGG')
                
            else:
                
                #Se crea el conjunto de entrenamiento y validación/prueba para cada caso
                sources_list, class_list = multiclassparser(origin_folder_list)
                X, Y, class_dict = multiclass_preprocessing(sources_list, class_list)
                multiclass_set_creator(X, Y, class_dict, destiny_dir, train_test_rate = train_ratio, test_val_rate = val_test_ratio, CNN_class = CNN_class_name)    

            #Se entrena la red con el modelo especificado
            CNN_train_iterator(CNN_params_obj, CNN_train_params_obj,  img_shape, destiny_dir, destiny = CNN_class_name_destiny, model_name = model_name,\
            class_layers_range = class_layers_range, class_neuron_range = class_neuron_range, last_check = last_check,\
                ntimes = ntimes, tf_seed = tf_seed)
            
            #Se encuentra el mejor modelo para el entrenamiento propuesto
            iter_dir = destiny_dir + '/' + CNN_class_name_destiny
            model, session = best_model_finder(iter_dir, CNN_class_name = CNN_class_name)
            
            #Se calculan los datos de prueba para el mejor modelo encontrado  
            accuracy, TNR, TPR, F1  = CNN_train_test(destiny_dir, session, model, CNN_train_params_obj, best_val_acc = 0, destiny = CNN_class_name_destiny,\
                model_name = model_name, CNN_class_name = CNN_class_name, train_bool = False)    
        
        #Se crea un diccionario con los valores de testeo asociados a un tran_ratio
        if train_start_bool:
            
            ratio_dic = {'accuracy' : [accuracy],'TNR' : [TNR], 'TPR' : [TPR],\
                'F1': [F1], 'train_ratio_list': [train_ratio], 'CNN_class': CNN_class_name, 'bool': False}
            train_start_bool = False
            
        else:
            
            ratio_dic['accuracy'].append(accuracy)
            ratio_dic['TNR'].append(TNR)
            ratio_dic['TPR'].append(TPR)
            ratio_dic['F1'].append(F1)
            ratio_dic['train_ratio_list'].append(train_ratio)
        
        with open(destiny_dir + '/' + CNN_class_name_destiny + '/ratio_dic.pickle', 'wb') as handle: pickle_dump(ratio_dic, handle, protocol=pickle_HIGHEST_PROTOCOL)  
                
#Función que genera gráficos a partir de datos obtenidos en la iteración de tamaños de entrenamiento
def set_size_plot(set_size_dir_list, fig_destiny):
    
    #Lista que contendrá el nombre de los archivos contendores de datos
    file_name_list = []

    #Se verifica que la entrada sea una lista, en caso contrario se ocupa el único nombre para extraer los datos
    if isinstance(set_size_dir_list, list):
        
        #Se lee y agrega cada uno de los nombres de archivo de datos contenidos en la lista de carpetas entregada
        for set_size_dir in set_size_dir_list: file_name_list = [set_size_dir + '/' + filename for filename in os.listdir(set_size_dir) if filename.endswith('ratio_dic.pickle')]
                    
    else: file_name_list = [set_size_dir + '/' + filename for filename in os.listdir(set_size_dir) if filename.endswith('ratio_dic.pickle')]                
                
    #Se crean las figuras y ejes para graficar los resultados
    accuracy_fig, accuracy_ax = plt.subplots()
    accuracy_fig.set_size_inches(16, 9) 
    accuracy_ax.set_title('Resultados de accuracy')
    accuracy_ax.set_xlabel('Tamaño relativo de conjunto de entrenamiento')
    accuracy_ax.set_ylabel('Valor sobre conjunto de prueba')
    accuracy_ax.set_xlim( [0, 1] )
    accuracy_ax.set_ylim( [0, 1] )
    
    TNR_fig, TNR_ax = plt.subplots()
    TNR_fig.set_size_inches(16, 9) 
    TNR_ax.set_title('Resultados de TNR')
    TNR_ax.set_xlabel('Tamaño relativo de conjunto de entrenamiento')
    TNR_ax.set_ylabel('Valor sobre conjunto de prueba')
    TNR_ax.set_xlim( [0, 1] )
    TNR_ax.set_ylim( [0, 1] )
    
    TPR_fig, TPR_ax = plt.subplots()
    TPR_fig.set_size_inches(16, 9)
    TPR_ax.set_title('Resultados de TPR') 
    TPR_ax.set_xlabel('Tamaño relativo de conjunto de entrenamiento')
    TPR_ax.set_ylabel('Valor sobre conjunto de prueba')
    TPR_ax.set_xlim( [0, 1] )
    TPR_ax.set_ylim( [0, 1] )
    
    F1_fig, F1_ax = plt.subplots()
    F1_fig.set_size_inches(16, 9) 
    F1_ax.set_title('Resultados de F1')
    F1_ax.set_xlabel('Tamaño relativo de conjunto de entrenamiento')
    F1_ax.set_ylabel('Valor sobre conjunto de prueba')
    F1_ax.set_xlim( [0, 1] )
    F1_ax.set_ylim( [0, 1] )
    
    #Para cada archivo de datos de tamaño de entrenamiento se obtiene el diccionario y cada valor individual    
    for file_name in file_name_list:
    
        with open(file_name, 'rb') as handle: ratio_dic = pickle_load(handle)
            
        accuracy = np.asarray( ratio_dic['accuracy'] ) 
        TNR = np.asarray( ratio_dic['TNR'] )
        TPR = np.asarray( ratio_dic['TPR'] ) 
        F1 = np.asarray ( ratio_dic['F1'] )
        train_ratio_list = np.asarray ( ratio_dic['train_ratio_list'] )
        CNN_class = ratio_dic['CNN_class']
        
        #Se dibujan los resultados de cada archivo en el plot correspondiente
        accuracy_ax.plot(train_ratio_list, accuracy, label = CNN_class)
        accuracy_ax.legend()
        
        TNR_ax.plot(train_ratio_list, TNR, label = CNN_class)
        TNR_ax.legend()
        
        TPR_ax.plot(train_ratio_list, TPR, label = CNN_class)
        TPR_ax.legend()
        
        F1_ax.plot(train_ratio_list, F1, label = CNN_class)
        F1_ax.legend()
        
    #Se crea la lista de xticks para todos los gráficos, incluyendo el 0 y 1 en caso de no estar    
    x_ticks = np.concatenate( ( np.array([0]), train_ratio_list, np.array([1]) ) )    
    
    #Se imponen los xticks y se guardan las imágenes     
    os.makedirs(fig_destiny , exist_ok = True)
    
    accuracy_ax.set_xticks(x_ticks)
    accuracy_fig.savefig(fig_destiny + '/accuracy.pdf')
    
    TNR_ax.set_xticks(x_ticks)
    TNR_fig.savefig(fig_destiny + '/TNR.pdf')
    
    TPR_ax.set_xticks(x_ticks)
    TPR_fig.savefig(fig_destiny + '/TPR.pdf')
    
    F1_ax.set_xticks(x_ticks)
    F1_fig.savefig(fig_destiny + '/F1.pdf')
    
#Función para generar gráficos (y encontrar el mejor valor) para distintos valores de margen
def margin_iterator(CNN_params_obj, CNN_train_params_obj, img_shape, data_dir, destiny_dir,  destiny = 'margin_iter', ntimes = 10, tf_seed = 1,\
    train_ratio = 0.25, val_test_ratio = 0.5, margin_interval = [0.5, 5, 10], distance = 0.5):
    
    #Se crea el conjunto de entrenamiento y validación/prueba para cada caso
    sources_list, class_list = multiclassparser(data_dir)
    X, Y, class_dict = multiclass_preprocessing(sources_list, class_list)
    multiclass_set_creator(X, Y, class_dict, destiny_dir, train_test_rate = train_ratio, test_val_rate = val_test_ratio, CNN_class = 'CSN')    
    epochs = CNN_train_params_obj.epochs
    
    #Se crea el vector para iterar sobre los márgenes
    margin_vector = np.linspace(margin_interval[0], margin_interval[1], margin_interval[2])   
    val_acc_iter = 0
    margin_acc_vec = []
    margin_val_acc_vec = []
    
    #Se crea el diccionario de los margenes
    margin_bool = False
    
    #Se crea el modelo para el margen pedido
    for margin in margin_vector:
        
        print(margin)                
        #Se crea el objeto que contiene las funciones para la pérdida contrastiva para cada margen y la distancia especificada
        contrastive_loss_fcn_obj = contrastive_loss_fcn(margin, distance)
        
        #Se instancian los valores randomizados con la semilla especificada
        set_random_seed(tf_seed)
        margin_val_acc = 0
        
        #Para cada una de las ntimes que se especificaron
        for _ in range(ntimes):
        
            session, _, CNN_class = fully_CNN_creator(CNN_params_obj, img_shape, contrastive_loss_fcn_obj = contrastive_loss_fcn_obj )
            val_acc, sub_fold, full_destiny, acc_vec, val_acc_vec = CNN_train_test(destiny_dir, session, CNN_class, CNN_train_params_obj, best_val_acc = margin_val_acc,\
                destiny = destiny, CNN_class_name = 'CSN', train_bool = True, acc_vec = True)

            if val_acc > margin_val_acc:
                            
                margin_val_acc = val_acc
                margin_acc_vec = acc_vec
                margin_val_acc_vec = val_acc_vec
         
        #Si es la primera iteración se crea el diccionario con los datos del entrenamiento     
        if margin_bool == False:
            
            margin_dic = {'epochs': epochs, 'margin_list': [margin], 'val_acc_list': [margin_val_acc], 'acc_vec_list': [margin_acc_vec],\
                'val_acc_vec_list' : [margin_val_acc_vec]}
            margin_bool = True
            
        
        #Si no, se agregan los nuevos resultados
        else:
            
            margin_dic['margin_list'].append(margin) 
            margin_dic['val_acc_list'].append(margin_val_acc)
            margin_dic['acc_vec_list'].append(margin_acc_vec)
            margin_dic['val_acc_vec_list'].append(margin_val_acc_vec)
                        
        #Se guarda el diccionario con los datos obtenidos
        margin_dic_file = destiny_dir + '/'  + destiny + '/margin_dic.pickle'
        
        with open(margin_dic_file, 'wb') as handle: pickle_dump(margin_dic, handle, protocol=pickle_HIGHEST_PROTOCOL)

#Función que lee el diccionario de iteración de márgenes y genera gráficos a partir de ello
def margin_iter_plot(margin_dic_dir, fig_destiny):
    #Se lee el directorio donde está el diccionario
    with open(margin_dic_dir, 'rb') as handle: margin_dic = pickle_load(handle)
    #Se asignan los valores de vectores de accuracy y se crea el eje x en escala semi-log para las epochs
    margin_list = margin_dic['margin_list']
    val_acc_vec_list = margin_dic['val_acc_vec_list']
    acc_vec_list = margin_dic['acc_vec_list']
    epochs_end = len(acc_vec_list[0])
    epochs_semi_log = np.log10( np.linspace(1, epochs_end, epochs_end) )
    #Se crean las figuras y ejes para graficar los resultados
    accuracy_fig, accuracy_ax = plt.subplots()
    accuracy_fig.set_size_inches(16, 9) 
    accuracy_ax.set_title('Resultados de accuracy de entrenamiento')
    accuracy_ax.set_xlabel('Epochs de entrenamiento')
    accuracy_ax.set_ylabel('Valor sobre conjunto de entrenamiento')
    accuracy_ax.set_xlim( [ 0, epochs_semi_log[-1] ]  )
    accuracy_ax.set_ylim( [0, 1] )
    
    val_accuracy_fig, val_accuracy_ax = plt.subplots()
    val_accuracy_fig.set_size_inches(16, 9) 
    val_accuracy_ax.set_title('Resultados de accuracy de validación')
    val_accuracy_ax.set_xlabel('Epochs de entrenamiento')
    val_accuracy_ax.set_ylabel('Valor sobre conjunto de validación')
    val_accuracy_ax.set_xlim( [ 0, epochs_semi_log[-1] ]  )
    val_accuracy_ax.set_ylim( [0, 1] )
    
    #Para cada margen probado se generan los gráficos de accuracy de entrenamiento y validación
    for i in range(0, len(margin_list)):       
        accuracy_now = acc_vec_list[i]
        val_accuracy_now = val_acc_vec_list[i]
        margin_now = 'Margen: ' +  str( '{0:.2f}'.format ( margin_list[i] ) )
        #Se dibujan los resultados de cada archivo en el plot correspondiente
        accuracy_ax.plot(epochs_semi_log, accuracy_now, label = margin_now)
        accuracy_ax.legend()
        val_accuracy_ax.plot(epochs_semi_log, val_accuracy_now, label = margin_now)
        val_accuracy_ax.legend()
    #Se guardan las imágenes en el destino que les corresponde
    os.makedirs(fig_destiny , exist_ok = True)
    accuracy_fig.savefig(fig_destiny + '/accuracy.pdf'), val_accuracy_fig.savefig(fig_destiny + '/val_accuracy.pdf')
    
#Function to evaluate the performance of the algorithm with respect to hand-labeled images
def seg_performance(hand_labeled_dir, seg_labeled_dir, IOU = .5, imshow = True, only_intersection = False, confidence_level = 1.96, stat_info = False):
    #The files containing the segmentation values are loaded
    if os.path.isfile(hand_labeled_dir + '/seg_dict.pickle'):
        with open(hand_labeled_dir + '/seg_dict.pickle', 'rb') as handle: hand_labeled_segdict =  pickle_load(handle)
    else:
        raise ValueError('The directory of hand-labeled images does not contain valid information')
    if os.path.isfile(seg_labeled_dir + '/seg_dict.pickle'):
        with open(seg_labeled_dir + '/seg_dict.pickle', 'rb') as handle: seg_labeled_segdict =  pickle_load(handle)
    else:
        raise ValueError('The directory of images labeled by the algorithm does not contain valid information')
    name_list = [hand_labeled_dir + '/' +  s for s in os.listdir(hand_labeled_dir) if s.endswith('.jpg')]
    #Both dictionaries are checked to see if they contain the same number of images and if not, an error flag is raised
    hand_labeled_imgsizelist = hand_labeled_segdict['img_size']
    seg_labeled_imgsizelist = seg_labeled_segdict['img_size']
    if len(hand_labeled_imgsizelist) != len(seg_labeled_imgsizelist): raise ValueError('Las carpetas NO contienen la misma cantidad de información para imágenes')
    #The data corresponding to the LV bounding boxes are loaded
    hand_labeled_multirescoords = hand_labeled_segdict['multires_coords']
    hand_labeled_multireswh = hand_labeled_segdict['multires_wh']
    seg_labeled_multirescoords = seg_labeled_segdict['multires_coords']
    seg_labeled_multireswh = seg_labeled_segdict['multires_wh']
    multires_coords = []
    F1_total, LRP_total, FP_WL_total, FN_WL_total, TP_WL_total, TN_WL_total = 0, 0, 0, 0, 0, 0
    FP_WL_total_vec, FN_WL_total_vec, TP_WL_total_vec, TN_WL_total_vec = [], [], [], []
    F1_ct, LRP_ct = 0,0
    #The data corresponding to the heat maps are loaded
    hand_labeled_hmlist = hand_labeled_segdict['bin_hm']
    seg_labeled_hmlist = seg_labeled_segdict['bin_hm']
    error_hm, TP_hm, FP_hm, FN_hm, TN_hm = 0, 0, 0, 0, 0
    pixel_error_hm, pixel_TP_hm, pixel_FP_hm, pixel_FN_hm, pixel_TN_hm = 0, 0, 0, 0, 0
    pixel_error_WL, pixel_TP_WL, pixel_FP_WL, pixel_FN_WL, pixel_TN_WL = 0, 0, 0, 0, 0
    total_pixels, total_pixels_vec = 0, []
    pixel_error_hm_vec, pixel_TP_hm_vec, pixel_FP_hm_vec, pixel_FN_hm_vec, pixel_TN_hm_vec = [], [], [], [], []
    pixel_error_WL_vec, pixel_TP_WL_vec, pixel_FP_WL_vec, pixel_FN_WL_vec, pixel_TN_WL_vec = [], [], [], [], []
    GTP_hm, GTN_hm = 0, 0    
    #Performance over LV regions and heat maps are obtained for each image
    for img_idx, img_name in enumerate(name_list):
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2HSV)
        multires_img = np.copy(img)
        total_pixels += img.shape[0] * img.shape[1]
        total_pixels_vec.append(img.shape[0] * img.shape[1])
        #Initially, all hand-labeled instances of LV objects are considered false negatives ("not found by algorithm"). Conversely, every LV 
        #detection made by the algorithm is considered a false positive ("falsely detected by algorithm")
        LRP_IOU,  FN_WL, FP_WL, TP_WL = 0, len(hand_labeled_multirescoords[img_idx]), len(seg_labeled_multirescoords[img_idx]), 0
        handlabeledimg_multires_wh, handlabeledimg_multires_coords, segimg_multires_wh, segimg_multires_coords =\
            hand_labeled_multireswh[img_idx], hand_labeled_multirescoords[img_idx],seg_labeled_multireswh[img_idx], seg_labeled_multirescoords[img_idx]
        intersection_map = np.zeros(( img.shape[0], img.shape[1] ))
        pixel_intersection_map = np.copy(intersection_map)
        #Every LV object labeled by the algorithm is compared with ground truth 
        for hand_multires_idx, hand_multires_coord in enumerate(handlabeledimg_multires_coords):
            #Creation of coordinates for the hand-labeled bounding box
            hand_multires_wh = handlabeledimg_multires_wh[hand_multires_idx]
            hand_labeled_bb = [ np.max([0, int( hand_multires_coord[0] - hand_multires_wh[0]/2 )] ), np.min([int( hand_multires_coord[0] + hand_multires_wh[0]/2 ), img.shape[0]]),\
                np.max([0, int( hand_multires_coord[1] - hand_multires_wh[1]/2 )]), np.min([int( hand_multires_coord[1] + hand_multires_wh[1]/2 ), img.shape[1]]) ]
            intersection_map[hand_labeled_bb[0]:hand_labeled_bb[1], hand_labeled_bb[2]:hand_labeled_bb[3]] = 1
            pixel_intersection_map[hand_labeled_bb[0]:hand_labeled_bb[1], hand_labeled_bb[2]:hand_labeled_bb[3]] = 1
            #Comparison between algorithm results and ground truth
            for seg_multires_idx, seg_multires_coord in enumerate(segimg_multires_coords):
                seg_multires_wh = segimg_multires_wh[seg_multires_idx]
                seg_labeled_bb = [ np.max([0, int( seg_multires_coord[0] - seg_multires_wh[0]/2 )] ), np.min([int( seg_multires_coord[0] + seg_multires_wh[0]/2 ), img.shape[0]]),\
                np.max([0, int( seg_multires_coord[1] - seg_multires_wh[1]/2 )]), np.min([int( seg_multires_coord[1] + seg_multires_wh[1]/2 ), img.shape[1]]) ]
                #If the IOU is greater than input threshold, the detection is declared as a True Positive, and False Positives and False Negatives are subtracted by one
                intersection_map[seg_labeled_bb[0]:seg_labeled_bb[1], seg_labeled_bb[2]:seg_labeled_bb[3]] += 1
                pixel_intersection_map[seg_labeled_bb[0]:seg_labeled_bb[1], seg_labeled_bb[2]:seg_labeled_bb[3]] -= 1
                bb_I, bb_IOU = np.sum(intersection_map>1)/(hand_multires_wh[0]*hand_multires_wh[1]), np.sum(intersection_map>1)/np.sum(intersection_map>=1)
                IOU_vector = [bb_IOU, bb_I]
                if IOU_vector[only_intersection]>IOU: 
                    #Calculation of Localization Recall Precision Error (Oksuz, 2018)
                    TP_WL, FN_WL, FP_WL = TP_WL + 1, FN_WL - 1, FP_WL - 1
                    LRP_IOU += (1-bb_IOU)/(1-IOU+np.spacing(1))
                #The intersection map is resetted to only contain ground truth for potential future iterations
                intersection_map[seg_labeled_bb[0]:seg_labeled_bb[1], seg_labeled_bb[2]:seg_labeled_bb[3]] -= 1
                
        #F1 and LRP metrics are calcuated ONLY if they are not undetermined, otherwise they are set as a NoneType
        if FN_WL <0: FN_WL = 0
        FN_WL_total += FN_WL
        TN_WL_total += 1-FP_WL
        TP_WL_total += TP_WL
        FP_WL_total += FP_WL
        FN_WL_total_vec.append(FN_WL)
        TN_WL_total_vec.append(1-FP_WL)
        TP_WL_total_vec.append(TP_WL)
        FP_WL_total_vec.append(FP_WL)
        pixel_error_WL += np.sum(pixel_intersection_map != 0)
        pixel_FN_WL += pixel_intersection_map[pixel_intersection_map>0].shape[0]
        pixel_FP_WL += pixel_intersection_map[pixel_intersection_map<0].shape[0]
        pixel_TN_WL += pixel_intersection_map[(pixel_intersection_map == 0) & (intersection_map == 0)].shape[0]
        pixel_TP_WL += pixel_intersection_map[(pixel_intersection_map == 0) & (intersection_map == 1)].shape[0]
        pixel_error_WL_vec.append(np.sum(pixel_intersection_map != 0))
        pixel_FN_WL_vec.append(pixel_intersection_map[pixel_intersection_map>0].shape[0])
        pixel_FP_WL_vec.append(pixel_intersection_map[pixel_intersection_map<0].shape[0])
        pixel_TN_WL_vec.append(pixel_intersection_map[(pixel_intersection_map == 0) & (intersection_map == 0)].shape[0])
        pixel_TP_WL_vec.append(pixel_intersection_map[(pixel_intersection_map == 0) & (intersection_map == 1)].shape[0])
        #Metrics are now calculated for the heat map class
        hm_img = np.copy(multires_img)
        hand_labeled_hmlist = hand_labeled_segdict['bin_hm']
        seg_labeled_hmlist = seg_labeled_segdict['bin_hm']
        #Each of the heat maps is loaded and transformed for subtraction
        hand_labeled_hm = hand_labeled_hmlist[img_idx].astype(np.int)
        seg_labeled_hm = seg_labeled_hmlist[img_idx].astype(np.int)
        #Heat maps are subtracted
        diff_hm = hand_labeled_hm - seg_labeled_hm
        #Ground truth positives and negatives are added as pixel-wise values, the same is done for the subtracted heatmap
        GTP_hm += np.sum(hand_labeled_hm)
        GTN_hm += np.sum(1-hand_labeled_hm)
        pixel_error_hm += np.sum(diff_hm != 0)
        pixel_FN_hm += diff_hm[diff_hm>0].shape[0]
        pixel_FP_hm += diff_hm[diff_hm<0].shape[0]
        pixel_TN_hm += diff_hm[(diff_hm == 0) & (hand_labeled_hm==0)].shape[0]
        pixel_TP_hm += diff_hm[(diff_hm == 0) & (hand_labeled_hm==1)].shape[0]
        pixel_error_hm_vec.append(np.sum(diff_hm != 0))
        pixel_FN_hm_vec.append(diff_hm[diff_hm>0].shape[0])
        pixel_FP_hm_vec.append(diff_hm[diff_hm<0].shape[0])
        pixel_TN_hm_vec.append(diff_hm[(diff_hm == 0) & (hand_labeled_hm==0)].shape[0])
        pixel_TP_hm_vec.append(diff_hm[(diff_hm == 0) & (hand_labeled_hm==1)].shape[0])
        #FN are blue, FP are red, all the correct labels are green.
        hm_img[diff_hm<0,0] = 0
        hm_img[diff_hm>0,0] = 120
        hm_img[(diff_hm == 0) , 0] = 90
        #Parts classified as LV are returned to the original values
        for multires_coord in multires_coords: hm_img[multires_coord[0]:multires_coord[1], multires_coord[2]:multires_coord[3]] = multires_img[multires_coord[0]:multires_coord[1], multires_coord[2]:multires_coord[3]]
        hm_img = cv2.cvtColor(hm_img, cv2.COLOR_HSV2BGR)
        #If it was chosen to display performance images (UNDER DEVELOPMENT)
        if imshow:
            cv2.imshow('diff', cv2.resize( (diff_hm + 2).astype(np.uint8)*120, None, fx = .2, fy=.2 ))
            cv2.imshow('multires', cv2.cvtColor(cv2.resize( multires_img, None, fx = .2, fy=.2 ), cv2.COLOR_HSV2BGR))
            cv2.imshow('Imagen segmentada', cv2.resize(hm_img, None, fx = .2, fy = .2))
            cv2.waitKey(0)
    #Aggregated scores are calculated for the entirety of the set compared
    pixel_error_hm_vec, pixel_FN_hm_vec, pixel_FP_hm_vec = np.array(pixel_error_hm_vec), np.array(pixel_FN_hm_vec), np.array(pixel_FP_hm_vec)
    pixel_TN_hm_vec, pixel_TP_hm_vec = np.array(pixel_TN_hm_vec), np.array(pixel_TP_hm_vec)
    FN_WL_total_vec, TN_WL_total_vec, TP_WL_total_vec, FP_WL_total_vec =\
        np.array(FN_WL_total_vec), np.array(TN_WL_total_vec), np.array(TP_WL_total_vec), np.array(FP_WL_total_vec), 
    pixel_error_WL_vec, pixel_FN_WL_vec, pixel_FP_WL_vec = np.array(pixel_error_WL_vec), np.array(pixel_FN_WL_vec), np.array(pixel_FP_WL_vec)
    pixel_TN_WL_vec, pixel_TP_WL_vec = np.array(pixel_TN_WL_vec), np.array(pixel_TP_WL_vec)
    total_pixels_vec = np.array(total_pixels_vec)
    #First on heatmap class
    TP_hm, FP_hm = TP_hm/(img_idx+1), FP_hm/(img_idx+1)
    TPR_hm = pixel_TP_hm /( pixel_TP_hm + pixel_FN_hm)
    FPR_hm =  pixel_FP_hm /( pixel_FP_hm + pixel_TN_hm)
    TNR_hm = pixel_TN_hm /( pixel_TN_hm + pixel_FP_hm)
    FNR_hm = pixel_FN_hm /( pixel_TP_hm + pixel_FN_hm)
    error_hm_total = (pixel_FP_hm + pixel_FN_hm ) / (pixel_FP_hm + pixel_FN_hm + pixel_TP_hm + pixel_TN_hm)
    error_hm_std = np.std(np.divide(pixel_FP_hm_vec + pixel_FN_hm_vec, (pixel_FP_hm_vec + pixel_FN_hm_vec + pixel_TP_hm_vec + pixel_TN_hm_vec)))
    conf_int_hm = error_hm_std * confidence_level / np.sqrt(img_idx+1)
    balanced_acc_hm = TPR_hm*.5+TNR_hm*.5
    F1_hm_total = pixel_TP_hm / ( pixel_TP_hm + .5*( pixel_FP_hm + pixel_FN_hm ) )
    precision_hm, recall_hm = pixel_TP_hm / (pixel_TP_hm + pixel_FP_hm + np.spacing(1)), pixel_TP_hm / (pixel_TP_hm + pixel_FN_hm + np.spacing(1))
    #Then on LV class
    TPR_WL = pixel_TP_WL /( pixel_TP_WL + pixel_FN_WL)
    FPR_WL =  pixel_FP_WL /( pixel_FP_WL + pixel_TN_WL)
    TNR_WL = pixel_TN_WL /( pixel_TN_WL + pixel_FP_WL)
    FNR_WL = pixel_FN_WL /( pixel_FN_WL + pixel_TP_WL)
    F1_WL_total = TP_WL_total / (TP_WL_total + .5* ( FP_WL_total + FN_WL_total ) )
    error_WL_total = (pixel_FP_WL + pixel_FN_WL ) / (pixel_FP_WL + pixel_FN_WL + pixel_TP_WL + pixel_TN_WL)
    error_WL_std = np.std(np.divide(pixel_FP_WL_vec + pixel_FN_WL_vec, (pixel_FP_WL_vec + pixel_FN_WL_vec + pixel_TP_WL_vec + pixel_TN_WL_vec)))
    conf_int_WL = error_WL_std * confidence_level / np.sqrt(img_idx+1)
    balanced_acc_WL = TPR_WL*.5+TNR_WL*.5
    accuracy_WL_total = (pixel_TN_WL + pixel_TP_WL) / (pixel_FN_WL + pixel_FP_WL + pixel_TP_WL + pixel_TN_WL)
    precision_WL, recall_WL = TP_WL_total / (TP_WL_total + FP_WL_total + np.spacing(1)), TP_WL_total / (TP_WL_total + FN_WL_total + np.spacing(1))
    LRP_total = (LRP_IOU + FP_WL_total + FN_WL_total) / ( TP_WL_total + FP_WL_total + FN_WL_total ) if TP_WL_total + FP_WL_total + FN_WL_total != 0 else -1
    #The overall F1 metric is calculated using the F1-macro method.
    P_macro = precision_hm*.5 + precision_WL*.5#( (TP_WL_total / ( TP_WL_total + FP_WL_total + np.spacing(1) ) ) + (pixel_TP_hm / ( pixel_TP_hm + pixel_FP_hm + np.spacing(1)) ) ) / 2
    R_macro = recall_hm*.5 + recall_WL*.5#( (TP_WL_total / ( TP_WL_total + FN_WL_total + np.spacing(1) ) ) + (pixel_TP_hm / ( pixel_TP_hm + pixel_FN_hm + np.spacing(1)) ) ) / 2
    P_macro_vec, R_macro_vec = np.zeros(pixel_TP_hm_vec.shape[0]), np.zeros(pixel_TP_hm_vec.shape[0])
    P_WL_vec, R_WL_vec = np.zeros(TP_WL_total_vec.shape[0]), np.zeros(TP_WL_total_vec.shape[0])
    P_hm_vec, R_hm_vec = np.zeros(pixel_TP_hm_vec.shape[0]), np.zeros(pixel_TP_hm_vec.shape[0])
    pixel_hm_precision, pixel_hm_recall, pixel_WL_precision, pixel_WL_recall = 0, 0, 0, 0
    #Values for valid precision and recall divisors are obtained
    for el_idx in range(pixel_TP_hm_vec.shape[0]):
        #If both divisors are non-zero for precision 
        if (pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])>0 and (FP_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])>0:
            #P_macro_vec[el_idx] = pixel_TP_hm_vec[el_idx]/( pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx] )*.5 + TP_WL_total_vec[el_idx]/(FP_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])*.5
            P_WL_vec[el_idx] = TP_WL_total_vec[el_idx]/(FP_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])
            P_hm_vec[el_idx] = pixel_TP_hm_vec[el_idx]/( pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx] )
            pixel_hm_precision += pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx]
            pixel_WL_precision += pixel_FP_WL_vec[el_idx] + pixel_TP_WL_vec[el_idx]
            P_macro_vec[el_idx] = .5 * (P_WL_vec[el_idx] + P_hm_vec[el_idx])
        #If only the divisor for hm-class is non-zero for precision
        elif (pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])>0 and (FP_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])==0:
            P_WL_vec[el_idx] = float("nan")
            P_hm_vec[el_idx] = pixel_TP_hm_vec[el_idx]/( pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx] )
            pixel_hm_precision += pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx]
            P_macro_vec[el_idx] = P_hm_vec[el_idx]
        #If only the divisor for WL-class is non-zero for precision
        elif (pixel_FP_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])==0 and (FP_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])>0:
            P_WL_vec[el_idx] = TP_WL_total_vec[el_idx]/(FP_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])
            pixel_WL_precision += pixel_FP_WL_vec[el_idx] + pixel_TP_WL_vec[el_idx]
            P_hm_vec[el_idx] = float("nan")
            P_macro_vec[el_idx] = P_WL_vec[el_idx]
        #If no divisors are non-zero for precision
        else:
            P_WL_vec[el_idx] = float("nan")
            P_hm_vec[el_idx] = float("nan")
            P_macro_vec[el_idx] = float("nan")
        #If both divisors are non-zero for recall
        if (FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])>0 and (pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])>0:
            R_WL_vec[el_idx] = TP_WL_total_vec[el_idx]/( FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx] )
            R_hm_vec[el_idx] = pixel_TP_hm_vec[el_idx]/(pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])
            pixel_hm_recall += pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx]
            pixel_WL_recall += FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx]
            R_macro_vec[el_idx] = .5 * (R_WL_vec[el_idx] + R_hm_vec[el_idx])
        #If only the divisor for hm-class is non-zero for recall
        elif (FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])==0 and (pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])>0:
            R_WL_vec[el_idx] = float("nan")
            R_hm_vec[el_idx] = pixel_TP_hm_vec[el_idx]/(pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])
            pixel_hm_recall += pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx]
            R_macro_vec[el_idx] = R_hm_vec[el_idx]
        #If only the divisor for WL-class is non-zero for recall
        elif (FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx])>0 and (pixel_FN_hm_vec[el_idx] + pixel_TP_hm_vec[el_idx])==0:
            R_WL_vec[el_idx] = TP_WL_total_vec[el_idx]/( FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx] )
            R_hm_vec[el_idx] = float("nan")
            pixel_WL_recall += FN_WL_total_vec[el_idx] + TP_WL_total_vec[el_idx]
            R_macro_vec[el_idx] = R_WL_vec[el_idx]
        #If no divisors are non-zero for recall
        else:
            R_WL_vec[el_idx] = float("nan")
            R_hm_vec[el_idx] = float("nan")
            R_macro_vec[el_idx] = float("nan")
    #Values with their corresponding Confidence Interval (C.I.) are obtained for Precision-macro, Recall-macro and F1-macro
    print(pixel_hm_precision)
    F1_macro_vec = 2*  ( P_macro_vec * R_macro_vec ) / ( P_macro_vec + R_macro_vec ) 
    P_WL_vec, P_hm_vec, P_macro_vec = P_WL_vec[np.invert(np.isnan(P_macro_vec))],P_hm_vec[np.invert(np.isnan(P_macro_vec))],P_macro_vec[np.invert(np.isnan(P_macro_vec))]
    R_macro_vec = R_macro_vec[np.invert(np.isnan(R_macro_vec))]
    F1_macro_vec = F1_macro_vec[np.invert(np.isnan(F1_macro_vec))]
    precision_WL_confint, precision_hm_confint = confidence_level * np.sqrt(precision_WL*(1-precision_WL) / P_WL_vec.shape[0]), confidence_level * np.sqrt(precision_hm*(1-precision_hm) / P_hm_vec.shape[0])
    print(recall_hm)
    recall_WL_confint, recall_hm_confint = confidence_level * np.sqrt(recall_WL*(1-recall_WL) / R_WL_vec.shape[0]), confidence_level * np.sqrt(recall_hm*(1-recall_hm) / ( R_hm_vec.shape[0] * 1 ) )
    print(precision_WL_confint, precision_hm_confint, recall_WL_confint, recall_hm_confint, pixel_hm_precision)
    P_macro_avg, R_macro_avg = np.mean(P_macro_vec), np.mean(R_macro_vec)
    P_macro_std, R_macro_std = np.sqrt( ( P_macro_avg * (1-P_macro_avg) )), np.sqrt( ( R_macro_avg * (1-R_macro_avg) )) #np.std(P_macro_vec), np.std(R_macro_vec)
    P_macro_confint, R_macro_confint = confidence_level*P_macro_std / np.sqrt(P_macro_vec.shape[0]), confidence_level*R_macro_std / np.sqrt(R_macro_vec.shape[0])
    F1_macro_avg = 2* ( (P_macro*R_macro) / ( (P_macro+np.spacing(1)) + (R_macro+np.spacing(1)) ) )#np.mean(F1_macro_vec)
    F1_macro_std = np.sqrt( ( F1_macro_avg * (1-F1_macro_avg) ))#np.std(F1_macro_vec)
    F1_macro_confint = confidence_level*F1_macro_std / np.sqrt(F1_macro_vec.shape[0])
    F1_macro = 2* ( (P_macro*R_macro) / ( (P_macro+np.spacing(1)) + (R_macro+np.spacing(1)) ) )
    #Values with their corresponding Confidence Interval (C.I.) are obtained for TPR-macro, TNR-macro and balanced accuracy
    TPR_macro = .5*pixel_TP_hm / (pixel_TP_hm + pixel_FN_hm)  + .5*pixel_TP_WL / (pixel_TP_WL + pixel_FN_WL)
    TNR_macro = .5*pixel_TN_hm / (pixel_TN_hm + pixel_FP_hm) + .5*pixel_TN_WL / (pixel_TN_WL + pixel_FP_WL)
    balanced_acc_macro = TPR_macro*.5 + TNR_macro*.5
    TPR_macro_vec = np.divide(pixel_TP_hm_vec + pixel_TP_WL_vec, pixel_FN_WL_vec + pixel_TP_WL_vec + pixel_FN_hm_vec + pixel_TP_hm_vec  )
    TNR_macro_vec = np.divide(pixel_TN_hm_vec + pixel_TN_WL_vec, pixel_FP_WL_vec + pixel_TN_WL_vec + pixel_FP_hm_vec + pixel_TN_hm_vec  )
    balanced_acc_macro_vec = .5* (TPR_macro_vec + TNR_macro_vec) 
    balanced_acc_macro_vec = balanced_acc_macro_vec[np.invert(np.isnan(balanced_acc_macro_vec))]
    balanced_acc_macro = np.mean(balanced_acc_macro_vec)
    balanced_acc_macro_std =  np.sqrt( ( balanced_acc_macro * (1-balanced_acc_macro) ) )#np.std(balanced_acc_macro_vec )
    conf_int_balanced_acc_macro = balanced_acc_macro_std * confidence_level / np.sqrt(balanced_acc_macro_vec.shape[0] )#* total_pixels)
    #Values with their corresponding Confidence Interval (C.I.) are obtained for accuracy-macro and error-macro 
    error_macro = error_WL_total + error_hm_total
    error_macro_std = np.std(np.divide(pixel_FP_WL_vec + pixel_FN_WL_vec + pixel_FN_hm_vec + pixel_FP_hm_vec\
        , (pixel_FP_WL_vec + pixel_FN_WL_vec + pixel_TP_WL_vec + pixel_TN_WL_vec + pixel_FP_hm_vec + pixel_FN_hm_vec + pixel_TP_hm_vec + pixel_TN_hm_vec )))
    conf_int_macro = error_macro_std * confidence_level / np.sqrt(img_idx+1)
    accuracy_hm_vec = np.divide(pixel_TP_hm_vec + pixel_TN_hm_vec ,total_pixels_vec )
    accuracy_hm = np.mean(accuracy_hm_vec)
    accuracy_hm_confint = np.sqrt( ( accuracy_hm * (1-accuracy_hm) ) ) * confidence_level / ( np.sqrt(img_idx+1) )
    accuracy_macro_vec = np.divide(pixel_TP_hm_vec + pixel_TN_hm_vec + pixel_TP_WL_vec + pixel_TN_WL_vec ,\
        2*total_pixels_vec )
    balanced_acc_hm_confint = np.sqrt( ( balanced_acc_hm * (1-balanced_acc_hm) ) ) * confidence_level / ( np.sqrt(img_idx+1) )
    accuracy_macro = np.mean(accuracy_macro_vec)
    accuracy_macro_std = np.sqrt( ( accuracy_macro * (1-accuracy_macro) ) )#np.std(accuracy_macro_vec)
    conf_int_acc_macro = accuracy_macro_std * confidence_level / ( np.sqrt(img_idx+1) )# * total_pixels )
    accuracy_macro = (pixel_TP_hm + pixel_TN_hm + pixel_TN_WL + pixel_TP_WL) / (2*total_pixels)
    F1_hm_confint = np.sqrt( ( F1_hm_total * (1-F1_hm_total) ) ) * confidence_level / ( np.sqrt(img_idx+1) )
    accuracy_WL_confint = confidence_level * np.sqrt(accuracy_WL_total*(1-accuracy_WL_total) / P_WL_vec.shape[0])
    balanced_acc_WL_confint = confidence_level * np.sqrt(balanced_acc_WL*(1-balanced_acc_WL) / P_WL_vec.shape[0])
    F1_WL_confint = np.sqrt( ( F1_WL_total * (1-F1_WL_total) ) ) * confidence_level / ( np.sqrt(img_idx+1) )
    #If confidence intervals were requested
    if stat_info:
        return TPR_WL, FPR_WL, TPR_hm, FPR_hm, F1_hm_total, F1_WL_total, 1-LRP_total, P_macro, R_macro, F1_macro, precision_hm, recall_hm, precision_WL, recall_WL, error_macro,\
            [balanced_acc_macro, conf_int_balanced_acc_macro], [accuracy_macro, conf_int_acc_macro], [P_macro_avg, P_macro_confint], [R_macro_avg, R_macro_confint], [F1_macro_avg, F1_macro_confint],\
                [accuracy_hm, accuracy_hm_confint], [balanced_acc_hm, balanced_acc_hm_confint], [precision_hm, precision_hm_confint] , [recall_hm, recall_hm_confint], [F1_hm_total, F1_hm_confint],\
                    [accuracy_WL_total, accuracy_WL_confint], [balanced_acc_WL, balanced_acc_WL_confint], [precision_WL, precision_WL_confint] , [recall_WL, recall_WL_confint], [F1_WL_total, F1_WL_confint]
    else:
        return TPR_WL, FPR_WL, TPR_hm, FPR_hm, F1_hm_total, F1_WL_total, 1-LRP_total, P_macro, R_macro, F1_macro, precision_hm, recall_hm, precision_WL, recall_WL, error_macro
        
#Function to obtain the performance metrics of a two-step algorithm based on hand-labeled images
def ID_performance(hand_labeled_dir, first_model, second_model ,class_dict_reg, class_dict_sld, frame_size = (256, 256), feat_mean_first = [], feat_mean_second = [], overlap_factor_first_step = 5e-2,\
    overlap_factor_second_step = 5e-2, multi_res_win_size = (1280, 1280), multi_res_name = 'wild lettuce', IOU_multires_step = 5e-2, IOU_hm_step = 5e-2, heat_map_class = 'trebol', heat_map_display = True, bg_class = 'pasto',\
        class_mask_display = True, method = 'region_sld', model_type = ['CNN', 'CNN'], savedir = '', overwrite = False, pred_batch = [32, 32], r_neighbours = 0 , imsave = True, feat_model_only = True,\
            thresh_step_hm = 1e-2, thresh_step_region = 1e-1,  only_intersection = False, multi_region_thresh = True, seg_imshow = False, selection_dict_sld = '', feats_param_dict_sld = '',\
                selection_dict_reg = '', feats_param_dict_reg = ''):
    #Iteration vectors are created for each hyperparameter of the algorithm given the step entered for each one
    thresh_vector_hm = np.round( np.linspace(5e-3, 995e-3, num = int(1/thresh_step_hm)+1), decimals = 3) if ( thresh_step_hm!= 0 and  model_type[1] != 'SVM') else np.array([.5])
    thresh_vector_region = np.round( np.linspace(5e-3, 995e-3, num = int(1/thresh_step_region)+1), decimals = 3) if ( thresh_step_region!= 0 and  model_type[0] != 'SVM') else np.array([.5])
    overlap_factor_first_vec = np.round( np.linspace(.25, .85, num = int(.6/overlap_factor_first_step)+1), decimals = 3) if overlap_factor_first_step!= 0 else np.array([.85])
    overlap_factor_second_vec = np.round( np.linspace(.5, .75, num = int(.25/overlap_factor_second_step)+1), decimals = 3) if overlap_factor_second_step!= 0 else np.array([.75])
    IOU_multires_vec = np.round( np.linspace(.25, .75, num = int(.5/IOU_multires_step)+1), decimals = 3) if IOU_multires_step!= 0 else np.array([.25])
    IOU_hm_vec = np.round( np.linspace(.25, .75, num = int(.5/IOU_hm_step)+1), decimals = 3) if IOU_hm_step!= 0 else np.array([.75])
    TP_hm_vec, FP_hm_vec, F1_hm_vec, TP_wl_vec, FP_wl_vec, F1_wl_vec, LRP_vec, P_macro_vec, R_macro_vec, F1_macro_vec, P_hm_vec, R_hm_vec, P_wl_vec, R_wl_vec, mean_time_vec, error_macro_vec\
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    full_iter_idx = thresh_vector_hm.shape[0]*thresh_vector_region.shape[0]*overlap_factor_first_vec.shape[0]*overlap_factor_second_vec.shape[0]*\
        IOU_multires_vec.shape[0]*IOU_hm_vec.shape[0]
    iter_idx = 0
    #For each hyperparameter the code iterates (in case of choosing multiple region thresholds)
    if multi_region_thresh:
        full_iter_idx*=thresh_vector_region.shape[0]
        for thresh_wl_region in thresh_vector_region:
            for thresh_hm_region in thresh_vector_region[::-1]:
                for thresh_hm in thresh_vector_hm:
                    for overlap_factor_first in overlap_factor_first_vec:
                        for overlap_factor_second in overlap_factor_second_vec:
                            for IOU_multires in IOU_multires_vec:
                                for IOU_hm in IOU_hm_vec:
                                    thresh_savedir = savedir + '/temp'
                                    dict_savedir = thresh_savedir + '/' + model_type[0] + '-' + model_type[1]
                                    #Segmentation data is calculated within the input folder
                                    mean_time = folder_pipeline(hand_labeled_dir,first_model, second_model ,class_dict_reg, class_dict_sld, frame_size =frame_size, feat_mean_first = feat_mean_first,\
                                        feat_mean_second = feat_mean_second, overlap_factor_first = overlap_factor_first, overlap_factor_second = overlap_factor_second, multi_res_win_size = multi_res_win_size\
                                            , multi_res_name = multi_res_name, IOU_multires = IOU_multires, IOU_hm = IOU_hm, heat_map_class = heat_map_class, heat_map_display = heat_map_display, bg_class = bg_class,\
                                        class_mask_display = class_mask_display, method = method, model_type = model_type, savedir = thresh_savedir, overwrite = overwrite, pred_batch = pred_batch, r_neighbours = r_neighbours ,\
                                        imsave = imsave, feat_model_only = feat_model_only, hm_thresh = thresh_hm, region_wl_thresh = thresh_wl_region, region_hm_thresh = thresh_hm_region,\
                                            selection_dict_sld = selection_dict_sld, feats_param_dict_sld = feats_param_dict_sld,\
                                                selection_dict_reg = selection_dict_reg, feats_param_dict_reg = feats_param_dict_reg)
                                    TP_wl, FP_wl, TP_hm, FP_hm, F1_hm, F1_wl, LRP_total, P_macro, R_macro, F1_macro, P_hm, R_hm, P_wl, R_wl, error_macro \
                                        = seg_performance(hand_labeled_dir, dict_savedir, IOU = .5, imshow = seg_imshow)
                                    TP_hm_vec.append(TP_hm), FP_hm_vec.append(FP_hm), F1_hm_vec.append(F1_hm), TP_wl_vec.append(TP_wl), FP_wl_vec.append(FP_wl),F1_wl_vec.append(F1_wl), LRP_vec.append(LRP_total), mean_time_vec.append(mean_time)
                                    P_macro_vec.append(P_macro), R_macro_vec.append(R_macro), F1_macro_vec.append(F1_macro), P_hm_vec.append(P_hm), R_hm_vec.append(R_hm), P_wl_vec.append(P_wl), R_wl_vec.append(R_wl), error_macro_vec.append(error_macro)
                                    print(thresh_hm),print('TP_hm ' + str(TP_hm)), print('FP_hm ' + str(FP_hm)), print('TP_wl ' + str(TP_wl)), print('FP_wl ' + str(FP_wl)), print('P_macro ' + str(P_macro)), print('R_macro ' + str(R_macro)), print('F1_macro ' + str(F1_macro))
                                    iter_idx += 1
                                    print('Iteration number ' + str(iter_idx) + ' of ' + str(full_iter_idx))
        #The vectors are formatted to be stored in a dictionary with the variable order of the iterations
        TP_hm_vec = np.reshape( TP_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        FP_hm_vec = np.reshape( FP_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        F1_hm_vec = np.reshape( F1_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        TP_wl_vec = np.reshape( TP_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        FP_wl_vec = np.reshape( FP_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        F1_wl_vec = np.reshape( F1_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        LRP_vec = np.reshape(LRP_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        P_macro_vec = np.reshape( P_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        R_macro_vec = np.reshape( R_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        F1_macro_vec = np.reshape( F1_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        P_hm_vec = np.reshape( P_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        R_hm_vec = np.reshape( R_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        P_wl_vec = np.reshape( P_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        R_wl_vec = np.reshape( R_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        mean_time_vec = np.reshape(mean_time_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        error_macro_vec = np.reshape(error_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        metric_dict = {'parameter_name_list': ['thresh_vector_region_wl', 'thresh_vector_region_hm', 'thresh_vector_hm', 'overlap_factor_first_vec', 'overlap_factor_second_vec', 'IOU_multires_vec', 'IOU_hm_vec'],\
            'parameter_list': [thresh_vector_region, thresh_vector_region, thresh_vector_hm, overlap_factor_first_vec, overlap_factor_second_vec, IOU_multires_vec, IOU_hm_vec],\
                'metric_name_list' : ['TP_hm_vec', 'FP_hm_vec', 'F1_hm_vec', 'TP_wl_vec', 'FP_wl_vec', 'F1_wl_vec', 'LRP_vec', 'P_macro_vec', 'R_macro_vec', 'F1_macro_vec', 'P_hm_vec', 'R_hm_vec', 'P_wl_vec','R_wl_vec', 'mean_time_vec', 'error_macro_vec'],\
                    'metric_list': [TP_hm_vec, FP_hm_vec, F1_hm_vec, TP_wl_vec, FP_wl_vec, F1_wl_vec, LRP_vec, P_macro_vec, R_macro_vec, F1_macro_vec, P_hm_vec, R_hm_vec, P_wl_vec, R_wl_vec, mean_time_vec, 1-error_macro_vec]}
    #For each hyperparameter the code iterates (in case of choosing only one region threshold)
    else:
        for thresh_region in thresh_vector_region:
            for thresh_hm in thresh_vector_hm:
                for overlap_factor_first in overlap_factor_first_vec:
                    for overlap_factor_second in overlap_factor_second_vec:
                        for IOU_multires in IOU_multires_vec:
                            for IOU_hm in IOU_hm_vec:
                                thresh_savedir = savedir + '/temp'
                                dict_savedir = thresh_savedir + '/' + model_type[0] + '-' + model_type[1]
                                #Segmentation data is calculated within the input folder
                                mean_time = folder_pipeline(hand_labeled_dir,first_model, second_model ,class_dict_reg, class_dict_sld, frame_size =frame_size, feat_mean_first = feat_mean_first,\
                                    feat_mean_second = feat_mean_second, overlap_factor_first = overlap_factor_first, overlap_factor_second = overlap_factor_second, multi_res_win_size = multi_res_win_size\
                                        , multi_res_name = multi_res_name, IOU_multires = IOU_multires, IOU_hm = IOU_hm, heat_map_class = heat_map_class, heat_map_display = heat_map_display, bg_class = bg_class,\
                                    class_mask_display = class_mask_display, method = method, model_type = model_type, savedir = thresh_savedir, overwrite = overwrite, pred_batch = pred_batch, r_neighbours = r_neighbours ,\
                                    imsave = imsave, feat_model_only = feat_model_only, hm_thresh = thresh_hm, region_wl_thresh = thresh_region, region_hm_thresh = thresh_region, selection_dict_sld = selection_dict_sld, feats_param_dict_sld = feats_param_dict_sld,\
                                    selection_dict_reg = selection_dict_reg, feats_param_dict_reg = feats_param_dict_reg)
                                TP_wl, FP_wl, TP_hm, FP_hm, F1_hm, F1_wl, LRP_total, P_macro, R_macro, F1_macro, P_hm, R_hm, P_wl, R_wl, error_macro\
                                        = seg_performance(hand_labeled_dir, dict_savedir, IOU = .5, imshow = seg_imshow)
                                TP_hm_vec.append(TP_hm), FP_hm_vec.append(FP_hm), F1_hm_vec.append(F1_hm), TP_wl_vec.append(TP_wl), FP_wl_vec.append(FP_wl), F1_wl_vec.append(F1_wl), LRP_vec.append(LRP_total), mean_time_vec.append(mean_time)
                                P_macro_vec.append(P_macro), R_macro_vec.append(R_macro), F1_macro_vec.append(F1_macro), P_hm_vec.append(P_hm), R_hm_vec.append(R_hm), P_wl_vec.append(P_wl), R_wl_vec.append(R_wl), error_macro_vec.append(error_macro)
                                print(thresh_hm),print('TP_hm ' + str(TP_hm)), print('FP_hm ' + str(FP_hm)), print('TP_wl ' + str(TP_wl)), print('FP_wl ' + str(FP_wl))
                                iter_idx += 1
                                print('Iteration number ' + str(iter_idx) + ' of ' + str(full_iter_idx))
        #The vectors are formatted to be stored in a dictionary with the variable order of the iterations
        TP_hm_vec = np.reshape( TP_hm_vec,  ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        FP_hm_vec = np.reshape( FP_hm_vec,  ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        F1_hm_vec = np.reshape( F1_hm_vec,  ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        TP_wl_vec = np.reshape( TP_wl_vec,  ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        FP_wl_vec = np.reshape( FP_wl_vec,  ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        F1_wl_vec = np.reshape( F1_wl_vec,  ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        LRP_vec = np.reshape(LRP_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ))
        P_macro_vec = np.reshape( P_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        R_macro_vec = np.reshape( R_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        F1_macro_vec = np.reshape( F1_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        P_hm_vec = np.reshape( P_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        R_hm_vec = np.reshape( R_hm_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        P_wl_vec = np.reshape( P_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        R_wl_vec = np.reshape( R_wl_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        mean_time_vec = np.reshape( mean_time_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        error_macro_vec = np.reshape( error_macro_vec, ( thresh_vector_region.shape[0], thresh_vector_hm.shape[0], overlap_factor_first_vec.shape[0], overlap_factor_second_vec.shape[0], IOU_multires_vec.shape[0], IOU_hm_vec.shape[0] ) )
        metric_dict = {'parameter_name_list': ['thresh_vector_region', 'thresh_vector_hm', 'overlap_factor_first_vec', 'overlap_factor_second_vec', 'IOU_multires_vec', 'IOU_hm_vec'],\
            'parameter_list': [thresh_vector_region, thresh_vector_hm, overlap_factor_first_vec, overlap_factor_second_vec, IOU_multires_vec, IOU_hm_vec],\
                'metric_name_list' : ['TP_hm_vec', 'FP_hm_vec', 'F1_hm_vec', 'TP_wl_vec', 'FP_wl_vec', 'F1_wl_vec', 'LRP_vec', 'P_macro_vec', 'R_macro_vec', 'F1_macro_vec', 'P_hm_vec', 'R_hm_vec', 'P_wl_vec','R_wl_vec', 'mean_time_vec', 'error_macro_vec'],\
                    'metric_list': [TP_hm_vec, FP_hm_vec, F1_hm_vec, TP_wl_vec, FP_wl_vec, F1_wl_vec, LRP_vec, P_macro_vec, R_macro_vec, F1_macro_vec, P_hm_vec, R_hm_vec, P_wl_vec, R_wl_vec, mean_time_vec, 1-error_macro_vec]}
    #The dictionary containing the metrics obtained is saved
    destiny_dir = savedir + '/metrics/' + model_type[0] + '-' + model_type[1] 
    os.makedirs(destiny_dir, exist_ok=True)
    with open(destiny_dir + '/metrics_dic.pickle', 'wb') as handle: pickle_dump(metric_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)

#Function to find the two-step detector that obtains the best performance based on an inputted metric
def best_ID_finder(metric_dir, metric_name = 'F1_macro', multigraph = True, all_models_par = 'thresh_vector_hm'):
    print('Starting ID graph construction...')
    original_metric_name = '' + metric_name
    metric_name = metric_name + '_vec'
    #The dictionary(s) containing the calculated metrics name(s) is(are) loaded
    dict_names = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(metric_dir)
    for f in files if f.endswith('metrics_dic.pickle')]
    dir_names = [dirpath
    for dirpath, dirnames, files in os.walk(metric_dir)
    for f in files if f.endswith('metrics_dic.pickle')]
    if not dict_names: raise ValueError('Please input directory that contains metric info')    
    all_model_names, all_metrics_dict, par_max_idx_vec, model_metric_vec, model_max_vec, model_best_metric_vec = [], [], [], [], [], []
    #For each dictionary found, the code iterates
    for dict_idx, dict_name in enumerate(dict_names):
        model_name = dir_names[dict_idx].rsplit('\\', 1)[-1]
        all_model_names.append(model_name)
        with open(dict_name, 'rb') as handle: metrics_dict = pickle_load(handle)
        #If the name of the metric is not in the loaded dictionary variables, an error message is displayed
        if metric_name not in metrics_dict['metric_name_list']: raise ValueError('The input metric was not calculated')
        all_metrics_dict.append(metrics_dict)
        metric_index = metrics_dict['metric_name_list'].index(metric_name)
        if dict_idx == 0:
            all_P_macro_vec, all_R_macro_vec, all_P_hm_vec, all_R_hm_vec =\
            [None]*len(dict_names)*len(metrics_dict['parameter_name_list']), [None]*len(dict_names)*len(metrics_dict['parameter_name_list']),\
                [None]*len(dict_names)*len(metrics_dict['parameter_name_list']), [None]*len(dict_names)*len(metrics_dict['parameter_name_list'])
        #The best metric value (and its index) is obtained for each model
        best_metric = metrics_dict['metric_list'][metric_index]
        max_metric_index = np.array(np.unravel_index(best_metric.argmax(), best_metric.shape))
        model_max_vec.append(max_metric_index)
        P_macro_vec, R_macro_vec = metrics_dict['metric_list'][metrics_dict['metric_name_list'].index('P_macro_vec')], metrics_dict['metric_list'][metrics_dict['metric_name_list'].index('R_macro_vec')]
        P_hm_vec, R_hm_vec = metrics_dict['metric_list'][metrics_dict['metric_name_list'].index('P_hm_vec')], metrics_dict['metric_list'][metrics_dict['metric_name_list'].index('R_hm_vec')]
        F1_macro_vec =  metrics_dict['metric_list'][metrics_dict['metric_name_list'].index('F1_macro_vec')]
        new_P_macro_vec, new_R_macro_vec, new_P_hm_vec, new_R_hm_vec = P_macro_vec, R_macro_vec, P_hm_vec, R_hm_vec
        #For each parameter iterated on, Precision-Recall plots are constructed.
        all_fig_destiny = metric_dir + '/figs/' + 'all_models/'
        os.makedirs(all_fig_destiny, exist_ok=True)
        #If it was chosen to create multiple curves containing the Precision-Recall information for several overlapping threshold values for regions
        if multigraph:
            #The vectors that will be used to create the curves are extracted
            region_P_macro_vec, region_R_macro_vec, region_P_hm_vec, region_R_hm_vec = np.copy(P_macro_vec), np.copy(R_macro_vec), np.copy(P_hm_vec), np.copy(R_hm_vec)
            region_F1_macro_vec = np.copy(F1_macro_vec)
            region_max_metric_index = np.copy(max_metric_index)
            hm_idx = [metrics_dict['parameter_name_list'].index('thresh_vector_hm')]
            if 'thresh_vector_region' in metrics_dict['parameter_name_list']: region_idx =  [metrics_dict['parameter_name_list'].index('thresh_vector_region')]
            else: region_idx =  [metrics_dict['parameter_name_list'].index('thresh_vector_region_wl'), metrics_dict['parameter_name_list'].index('thresh_vector_region_hm')]
            #If the information contained in the dictionary corresponds to iterations over one threshold value for region
            if len(region_idx) <= 1:
                for n_idx, par_idx in enumerate(region_max_metric_index):
                    if n_idx != region_idx[0] and n_idx != hm_idx[0]:
                        region_P_macro_vec, region_R_macro_vec = region_P_macro_vec[:,:,region_max_metric_index[n_idx]], region_R_macro_vec[:,:,region_max_metric_index[n_idx]]
                        region_P_hm_vec, region_R_hm_vec =  region_P_hm_vec[:,:,region_max_metric_index[n_idx]], region_R_hm_vec[:,:,region_max_metric_index[n_idx]]
                        region_F1_macro_vec = region_F1_macro_vec[:,:,region_max_metric_index[n_idx]]
                n_region_macro_pr_fig, n_region_macro_pr_ax = plt.subplots()
                n_region_macro_pr_fig.set_size_inches(16, 9)
                #The curves are plotted for all the region values over which iteration was performed
                for n_region in range(len(metrics_dict['parameter_list'][metrics_dict['parameter_name_list'].index('thresh_vector_region')])):
                    n_region_P_macro_vec, n_region_R_macro_vec = region_P_macro_vec[n_region], region_R_macro_vec[n_region]
                    n_region_R_macro_vec, n_region_P_macro_vec = np.concatenate( (n_region_R_macro_vec, [0]) ), np.concatenate( (n_region_P_macro_vec, [np.max(n_region_P_macro_vec)]) )
                    n_region_F1_macro_vec = region_F1_macro_vec[n_region]
                    n_region_auc = auc(n_region_R_macro_vec, n_region_P_macro_vec)
                    n_region_macro_pr_ax.plot(n_region_R_macro_vec, n_region_P_macro_vec, label = model_name + ', AUC: ' + f"{n_region_auc:.2f}")
                    n_region_macro_pr_ax.set_title('Precision-Recall curve for both classes of weed', fontsize = 20)
                    n_region_macro_pr_ax.set_xlabel('Recall', fontsize = 20)
                    n_region_macro_pr_ax.set_ylabel('Precision', fontsize = 20)
                    n_region_macro_pr_ax.set_xlim( [ 0, 1]  )
                    n_region_macro_pr_ax.set_ylim( [0, 1] )
                    n_region_macro_pr_ax.legend()
                #WIP
                fig_destiny = metric_dir + '/figs/' + model_name + '/'
                os.makedirs(fig_destiny, exist_ok = True)
                n_region_macro_pr_fig.savefig(fig_destiny + 'overlap_macro_auc.pdf')
            #If the information contained in the dictionary corresponds to iterations over two threshold values for region
            else:
                #The iteration is differentiated in such a way that the value of one of the region thresholds is fixed at the global maximum obtained
                for region_sub_idx in region_idx:
                    region_max_metric_index = np.copy(max_metric_index)
                    region_P_macro_vec, region_R_macro_vec, region_P_hm_vec, region_R_hm_vec = np.copy(P_macro_vec), np.copy(R_macro_vec), np.copy(P_hm_vec), np.copy(R_hm_vec)
                    if region_sub_idx == 0: 
                        region_P_macro_vec, region_R_macro_vec = region_P_macro_vec[:,region_max_metric_index[1],:,:,:,:,:], region_R_macro_vec[:,region_max_metric_index[1],:,:,:,:,:]
                        region_P_hm_vec, region_R_hm_vec =  region_P_hm_vec[:,region_max_metric_index[1],:,:,:,:,:], region_R_hm_vec[:,region_max_metric_index[1],:,:,:,:,:]
                        region_max_metric_index = np.delete(region_max_metric_index, 1)
                    else:
                        region_P_macro_vec, region_R_macro_vec = region_P_macro_vec[region_max_metric_index[0],:,:,:,:,:,:], region_R_macro_vec[region_max_metric_index[0],:,:,:,:,:,:]
                        region_P_hm_vec, region_R_hm_vec =  region_P_hm_vec[region_max_metric_index[0],:,:,:,:,:,:], region_R_hm_vec[region_max_metric_index[0],:,:,:,:,:,:]
                        region_max_metric_index = np.delete(region_max_metric_index, 0)
                    #The curves are plotted for all the region thresholds values over which iteration was performed
                    for n_idx, par_idx in enumerate(region_max_metric_index):
                        if n_idx >1 :
                            region_P_macro_vec, region_R_macro_vec = region_P_macro_vec[:,:,region_max_metric_index[n_idx]], region_R_macro_vec[:,:,region_max_metric_index[n_idx]]
                            region_P_hm_vec, region_R_hm_vec =  region_P_hm_vec[:,:,region_max_metric_index[n_idx]], region_R_hm_vec[:,:,region_max_metric_index[n_idx]]
                    if region_sub_idx == 0:
                        n_region_hm_pr_fig, n_region_hm_pr_ax = plt.subplots()
                        n_region_hm_pr_fig.set_size_inches(16, 9)
                        
                        n_region_macro_pr_fig, n_region_macro_pr_ax = plt.subplots()
                        n_region_macro_pr_fig.set_size_inches(16, 9)
                        for n_region in range(len(metrics_dict['parameter_list'][metrics_dict['parameter_name_list'].index('thresh_vector_region_wl')])):
                            n_region_P_hm_vec, n_region_R_hm_vec = region_P_hm_vec[n_region], region_R_hm_vec[n_region]
                            n_region_R_hm_vec, n_region_P_hm_vec = np.concatenate( (n_region_R_hm_vec, np.array([0])) ), np.concatenate( (n_region_P_hm_vec, [np.max(n_region_P_hm_vec)]) )
                            n_region_R_hm_vec, n_region_P_hm_vec = np.concatenate( ([np.max(n_region_R_hm_vec)],n_region_R_hm_vec ) ), np.concatenate( ([0], n_region_P_hm_vec) )
                            n_region_hm_auc = auc(n_region_R_hm_vec, n_region_P_hm_vec) 
                            n_region_hm_pr_ax.plot(n_region_R_hm_vec, n_region_P_hm_vec, label = model_name + ', AUC: ' + f"{n_region_hm_auc:.2f}")
                            n_region_hm_pr_ax.set_title('Precision-Recall curve for heat-map class', fontsize = 20)
                            n_region_hm_pr_ax.set_xlabel('Recall', fontsize = 20)
                            n_region_hm_pr_ax.set_ylabel('Precision', fontsize = 20)
                            n_region_hm_pr_ax.set_xlim( [ 0, 1]  )
                            n_region_hm_pr_ax.set_ylim( [0, 1] )
                            n_region_hm_pr_ax.legend()
                            
                            n_region_P_macro_vec, n_region_R_macro_vec = region_P_macro_vec[n_region], region_R_macro_vec[n_region]
                            n_region_R_macro_vec, n_region_P_macro_vec = np.concatenate( (n_region_R_macro_vec, np.array([0])) ), np.concatenate( (n_region_P_macro_vec, [np.max(n_region_P_macro_vec)]) )
                            n_region_R_macro_vec, n_region_P_macro_vec = np.concatenate( ([np.max(n_region_R_macro_vec)],n_region_R_macro_vec ) ), np.concatenate( ([0], n_region_P_macro_vec) )
                            n_region_F1_macro_vec = region_F1_macro_vec[n_region]
                            n_region_macro_auc = auc(n_region_R_macro_vec, n_region_P_macro_vec) 
                            n_region_macro_pr_ax.plot(n_region_R_macro_vec, n_region_P_macro_vec, label = model_name + ', AUC: ' + f"{n_region_macro_auc:.2f}")
                            n_region_macro_pr_ax.set_title('Macro Precision-Recall curve for both classes of weed', fontsize = 20)
                            n_region_macro_pr_ax.set_xlabel('Macro Recall', fontsize = 20)
                            n_region_macro_pr_ax.set_ylabel('Macro Precision', fontsize = 20)
                            n_region_macro_pr_ax.set_xlim( [ 0, 1]  )
                            n_region_macro_pr_ax.set_ylim( [0, 1] )
                            n_region_macro_pr_ax.legend()
                    else:
                        n_region_hm_pr_fig, n_region_hm_pr_ax = plt.subplots()
                        n_region_hm_pr_fig.set_size_inches(16, 9)
                        
                        n_region_macro_pr_fig, n_region_macro_pr_ax = plt.subplots()
                        n_region_macro_pr_fig.set_size_inches(16, 9)
                        for n_region in range(len(metrics_dict['parameter_list'][metrics_dict['parameter_name_list'].index('thresh_vector_region_hm')])):
                            n_region_P_hm_vec, n_region_R_hm_vec = region_P_hm_vec[n_region], region_R_hm_vec[n_region]
                            n_region_R_hm_vec, n_region_P_hm_vec = np.concatenate( (n_region_R_hm_vec, np.array([0])) ), np.concatenate( (n_region_P_hm_vec, [np.max(n_region_P_hm_vec)]) )
                            n_region_R_hm_vec, n_region_P_hm_vec = np.concatenate( ([np.max(n_region_R_hm_vec)],n_region_R_hm_vec ) ), np.concatenate( ([0], n_region_P_hm_vec) )
                            n_region_hm_auc = auc(n_region_R_hm_vec, n_region_P_hm_vec) 
                            n_region_hm_pr_ax.plot(n_region_R_hm_vec, n_region_P_hm_vec, label = model_name + ', AUC: ' + f"{n_region_hm_auc:.2f}")
                            n_region_hm_pr_ax.set_title('Precision-Recall curve for heat-map class', fontsize = 20)
                            n_region_hm_pr_ax.set_xlabel('Recall', fontsize = 20)
                            n_region_hm_pr_ax.set_ylabel('Precision', fontsize = 20)
                            n_region_hm_pr_ax.set_xlim( [ 0, 1]  )
                            n_region_hm_pr_ax.set_ylim( [0, 1] )
                            n_region_hm_pr_ax.legend()
                            
                            n_region_P_macro_vec, n_region_R_macro_vec = region_P_macro_vec[n_region], region_R_macro_vec[n_region]
                            n_region_R_macro_vec, n_region_P_macro_vec = np.concatenate( (n_region_R_macro_vec, [0]) ), np.concatenate( (n_region_P_macro_vec, [np.max(n_region_P_macro_vec)]) )
                            n_region_F1_macro_vec = region_F1_macro_vec[n_region]
                            n_region_auc = auc(n_region_R_macro_vec, n_region_P_macro_vec) 
                            n_region_macro_pr_ax.plot(n_region_R_macro_vec, n_region_P_macro_vec, label = model_name + ', AUC: ' + f"{n_region_auc:.2f}")
                            n_region_macro_pr_ax.set_title('Macro Precision-Recall curve for both classes of weed', fontsize = 20)
                            n_region_macro_pr_ax.set_xlabel('Macro Recall', fontsize = 20)
                            n_region_macro_pr_ax.set_ylabel('Macro Precision', fontsize = 20)
                            n_region_macro_pr_ax.set_xlim( [ 0, 1]  )
                            n_region_macro_pr_ax.set_ylim( [0, 1] )
                            n_region_macro_pr_ax.legend()
                    #WIP
                    fig_destiny = metric_dir + '/figs/' + model_name + '/'
                    os.makedirs(fig_destiny, exist_ok = True)
                    n_region_macro_pr_fig.savefig(fig_destiny + 'overlap_macro_auc_' + str(region_sub_idx) + '.pdf'), n_region_hm_pr_fig.savefig(fig_destiny + 'overlap_hm_auc_' + str(region_sub_idx) + '.pdf')
                    #plt.show()
                    #plt.close('all')
        for par_idx in range(len(metrics_dict['parameter_name_list'])):
            best_metric = metrics_dict['metric_list'][metric_index]
            new_P_macro_vec, new_R_macro_vec, new_P_hm_vec, new_R_hm_vec = P_macro_vec, R_macro_vec, P_hm_vec, R_hm_vec
            par_name = metrics_dict['parameter_name_list'][par_idx]
            par_vec_idx = [idx for idx in range(len(metrics_dict['parameter_name_list'])) if idx != par_idx]
            for vec_idx in par_vec_idx: 
                if vec_idx<par_idx: new_P_macro_vec, new_R_macro_vec, new_P_hm_vec, new_R_hm_vec = new_P_macro_vec[max_metric_index[vec_idx],:], new_R_macro_vec[max_metric_index[vec_idx],:], new_P_hm_vec[max_metric_index[vec_idx],:], new_R_hm_vec[max_metric_index[vec_idx],:]
                else: new_P_macro_vec = new_P_macro_vec, new_R_macro_vec, new_P_hm_vec, new_R_hm_vec = new_P_macro_vec[:, max_metric_index[vec_idx]], new_R_macro_vec[:, max_metric_index[vec_idx]], new_P_hm_vec[:, max_metric_index[vec_idx]], new_R_hm_vec[:, max_metric_index[vec_idx]]
            #Precision and Recall values (heat map and macro) are assigned to build the classification model overlapping graphs
            all_R_macro_vec[len(metrics_dict['parameter_name_list'])*dict_idx + par_idx] = new_R_macro_vec
            all_P_macro_vec[len(metrics_dict['parameter_name_list'])*dict_idx + par_idx] = new_P_macro_vec
            all_R_hm_vec[len(metrics_dict['parameter_name_list'])*dict_idx + par_idx] = new_R_hm_vec
            all_P_hm_vec[len(metrics_dict['parameter_name_list'])*dict_idx + par_idx] = new_P_hm_vec
            #The vectors of the selected metric and LRP_IOU are reduced until the absolute maximum is obtained
            par_max_idx = max_metric_index[par_idx]
            par_max_idx_vec.append(par_max_idx)
            max_LRP = metrics_dict['metric_list'][metrics_dict['metric_name_list'].index('LRP_vec')]
            for max_idx in max_metric_index[:-1]: best_metric, max_LRP = best_metric[max_idx, :], max_LRP[max_idx,:]
            max_LRP = max_LRP[max_metric_index[-1]]
            best_metric = best_metric[max_metric_index[-1]]
            if par_idx == 0: model_metric_vec.append(best_metric)
            print('max_LRP', max_LRP)
            #Precision-Recall curves are constructed and plotted for both classes and for heat map separately
            hm_pr_fig, hm_pr_ax = plt.subplots()
            macro_pr_fig, macro_pr_ax = plt.subplots()
            hm_pr_fig.set_size_inches(16, 9), macro_pr_fig.set_size_inches(16, 9)
            hm_pr_ax.set_title('Precision-Recall curve for heat map class', fontsize = 20), macro_pr_ax.set_title('Precision-Recall curve for both classes of weed', fontsize = 20)
            hm_pr_ax.set_xlabel('Recall', fontsize = 20), macro_pr_ax.set_xlabel('Recall', fontsize = 20)
            hm_pr_ax.set_ylabel('Precision', fontsize = 20), macro_pr_ax.set_ylabel('Precision', fontsize = 20)
            hm_pr_ax.set_xlim( [ 0, 1]  ), macro_pr_ax.set_xlim( [ 0, 1]  )
            hm_pr_ax.set_ylim( [0, 1] ), macro_pr_ax.set_ylim( [0, 1] )
            hm_pr_ax.plot(new_R_hm_vec, new_P_hm_vec), macro_pr_ax.plot(new_R_macro_vec, new_P_macro_vec)
            hm_pr_ax.annotate('Max metric score point: ' + str( round( best_metric, 2 ) ), (new_R_hm_vec[par_max_idx] + 1e-2, new_P_hm_vec[par_max_idx] + 1e-2) )
            hm_pr_ax.annotate('Recall: '+ str(round(new_R_hm_vec[par_max_idx], 2 ) ) + ', Precision: '+ str( round( new_P_hm_vec[par_max_idx], 2 ) ), (new_R_hm_vec[par_max_idx] + 1e-2, new_P_hm_vec[par_max_idx] - 1e-2 ) )
            hm_pr_ax.scatter(new_R_hm_vec[par_max_idx], new_P_hm_vec[par_max_idx], s = 100,  c = 'gold')
            macro_pr_ax.annotate('Max metric score point: ' + str( round( best_metric, 2 ) ), (new_R_macro_vec[par_max_idx] + 1e-2, new_P_macro_vec[par_max_idx] + 1e-2) )
            macro_pr_ax.annotate('Recall: '+ str(round(new_R_macro_vec[par_max_idx], 2 ) ) + ', Precision: '+ str( round( new_P_macro_vec[par_max_idx], 2 ) ), (new_R_macro_vec[par_max_idx] + 1e-2, new_P_macro_vec[par_max_idx] - 1e-2 ) )
            macro_pr_ax.scatter(new_R_macro_vec[par_max_idx], new_P_macro_vec[par_max_idx], s = 100,  c = 'gold')
            fig_destiny = metric_dir + '/figs/' + model_name + '/'
            #The created graphics are saved
            os.makedirs(fig_destiny, exist_ok=True)
            hm_pr_fig.savefig(fig_destiny + '/hm_pr_' + par_name + '.pdf'), macro_pr_fig.savefig(fig_destiny + '/macro_pr_' + par_name + '.pdf')
            plt.close('all')  
    #The data is iterated over to plot the results of all classification models tested
    for par_idx, par_name in enumerate(metrics_dict['parameter_name_list']):
        #Figures are created to plot the precision-recall curves
        model_macro_fig, model_macro_ax = plt.subplots()
        model_hm_fig, model_hm_ax = plt.subplots()
        model_macro_fig.set_size_inches(16, 9), model_hm_fig.set_size_inches(16, 9)
        for model_idx, model_name in enumerate(all_model_names):
            #The best values are loaded to annotate the graphs
            max_metric_index = model_max_vec[model_idx]
            best_metric = model_metric_vec[model_idx]
            par_max_idx = par_max_idx_vec[len(metrics_dict['parameter_name_list'])*model_idx+par_idx]
            #Precision-Recall curves are constructed for the heat map class and for the two classes
            par_model_R_macro_vec, par_model_P_macro_vec = all_R_macro_vec[len(metrics_dict['parameter_name_list'])*model_idx+par_idx], all_P_macro_vec[len(metrics_dict['parameter_name_list'])*model_idx+par_idx]
            par_model_R_hm_vec, par_model_P_hm_vec = all_R_hm_vec[len(metrics_dict['parameter_name_list'])*model_idx+par_idx], all_P_hm_vec[len(metrics_dict['parameter_name_list'])*model_idx+par_idx]
            par_model_R_macro_vec, par_model_P_macro_vec =  np.concatenate( (par_model_R_macro_vec, [0]) ), np.concatenate( (par_model_P_macro_vec, [np.max(par_model_P_macro_vec)]) )
            if par_idx == metrics_dict['parameter_name_list'].index(all_models_par) and par_model_R_hm_vec.shape[0] > 1: 
                hm_par_auc, macro_par_auc = round(auc(par_model_R_hm_vec, par_model_P_hm_vec), 2), round(auc(par_model_R_macro_vec, par_model_P_macro_vec), 2)
                model_macro_ax.plot(par_model_R_macro_vec, par_model_P_macro_vec, label = model_name + ', AUC: ' + str(macro_par_auc), zorder = 10), model_macro_ax.legend(loc = 'lower left')
                model_hm_ax.plot(par_model_R_hm_vec, par_model_P_hm_vec, label = model_name + ', AUC: ' + str(hm_par_auc), zorder = 10), model_hm_ax.legend(loc = 'lower left')
            else:
                model_macro_ax.plot(par_model_R_macro_vec, par_model_P_macro_vec, label = model_name, zorder = 10) , model_macro_ax.legend()
                model_hm_ax.plot(par_model_R_hm_vec, par_model_P_hm_vec, label = model_name, zorder = 10) , model_hm_ax.legend()
            model_hm_ax.text( par_model_R_hm_vec[par_max_idx] + 1e-2, par_model_P_hm_vec[par_max_idx] + 2e-2, 'Max ' + original_metric_name + ' score: ' + str( round( best_metric, 2 ) ) + '\r\n' + \
                ' Recall: '+ str(round(par_model_R_hm_vec[par_max_idx], 2 ) ) + ', Precision: '+ str( round( par_model_P_hm_vec[par_max_idx], 2 ) ),\
                ha="left", va="bottom", size=10,bbox=dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2), zorder = 1)
            #model_hm_ax.annotate('Recall: '+ str(round(par_model_R_hm_vec[par_max_idx], 2 ) ) + ', Precision: '+ str( round( par_model_P_hm_vec[par_max_idx], 2 ) ), (par_model_R_hm_vec[par_max_idx] + 1e-2, par_model_P_hm_vec[par_max_idx] - .5e-2 ) )
            model_hm_ax.scatter(par_model_R_hm_vec[par_max_idx], par_model_P_hm_vec[par_max_idx], s = 100,  c = 'gold')
            model_macro_ax.text( par_model_R_macro_vec[par_max_idx] + 1e-2, par_model_P_macro_vec[par_max_idx] + 2e-2, 'Max ' + original_metric_name + ' score: ' + str( round( best_metric, 2 ) ) + '\r\n' + \
                ' Recall: '+ str(round(par_model_R_macro_vec[par_max_idx], 2 ) ) + ', Precision: '+ str( round( par_model_P_macro_vec[par_max_idx], 2 ) ),\
                ha="left", va="bottom", size=10,bbox=dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=2), zorder = 1)
            #model_macro_ax.annotate('Max metric score point: ' + str( round( best_metric, 2 ) ), (par_model_R_macro_vec[par_max_idx] + 1e-2, par_model_P_macro_vec[par_max_idx] + 2e-2) )
            #model_macro_ax.annotate('Recall: '+ str(round(par_model_R_macro_vec[par_max_idx], 2 ) ) + ', Precision: '+ str( round( par_model_P_macro_vec[par_max_idx], 2 ) ), (par_model_R_macro_vec[par_max_idx] + 1e-2, par_model_P_macro_vec[par_max_idx] - .5e-2 ) )
            model_macro_ax.scatter(par_model_R_macro_vec[par_max_idx], par_model_P_macro_vec[par_max_idx], s = 100,  c = 'gold') 
        model_macro_ax.set_title('Precision-Recall curve for both classes of weed', fontsize = 20), model_hm_ax.set_title('Precision-Recall curve for heat map class', fontsize = 20)
        model_macro_ax.set_xlabel('Recall', fontsize = 20), model_hm_ax.set_xlabel('Recall', fontsize = 20)
        model_macro_ax.set_ylabel('Precision', fontsize = 20), model_hm_ax.set_ylabel('Precision', fontsize = 20)
        model_macro_ax.set_xlim( [ 0, 1]  ), model_hm_ax.set_xlim( [ 0, 1]  )
        model_macro_ax.set_ylim( [0, 1] ), model_hm_ax.set_ylim( [0, 1] )
        #The created graphics are saved
        os.makedirs(all_fig_destiny, exist_ok=True)
        model_macro_fig.savefig(all_fig_destiny + 'macro_pr_' + par_name + '.pdf'), model_hm_fig.savefig(all_fig_destiny + 'hm_pr_' + par_name + '.pdf')
        plt.close('all')
        with open(all_fig_destiny + 'seg_dict.pickle', 'wb') as handle: pickle_dump(model_max_vec, handle, protocol = pickle_HIGHEST_PROTOCOL)
    print('Graphs created!'), print(model_max_vec, all_model_names, metrics_dict['parameter_name_list'], all_metrics_dict[0]['parameter_list'])
    return model_max_vec

#Function to choose specific images to obtain metrics    
def seg_dict_selector(hand_labeled_dir, img_list = [], overwrite = True ):
    #If the directory does not contain a file with a valid dictionary name, an error flag is raised.
    if os.path.isfile(hand_labeled_dir + '/seg_dict.pickle'):
        with open(hand_labeled_dir + '/seg_dict.pickle', 'rb') as handle: hand_labeled_segdict =  pickle_load(handle)
    else: raise ValueError('The directory of hand-tagged images does not contain valid information')
    new_dir = hand_labeled_dir + '/new_hand_labeled_dir'
    if os.path.isdir(new_dir) and overwrite: rmtree(new_dir)
    os.makedirs(new_dir, exist_ok = True)
    hand_labeled_len = len(hand_labeled_segdict['img_size'])
    new_img_index = [idx for idx in range(hand_labeled_len) if (idx in img_list or len(img_list)==0)]
    #A dictionary is created containing only the data to be copied to the new folder
    new_seg_dict = {'img_size':[], 'multires_coords':[], 'multires_wh':[], 'bin_hm':[], 'est_dens':[], 'param_dict':[]}
    new_seg_dict['img_size'] = [hand_labeled_segdict['img_size'][idx] for idx in new_img_index]
    new_seg_dict['multires_coords'] = [hand_labeled_segdict['multires_coords'][idx] for idx in new_img_index]
    new_seg_dict['multires_wh'] = [hand_labeled_segdict['multires_wh'][idx] for idx in new_img_index]
    new_seg_dict['bin_hm'] = [hand_labeled_segdict['bin_hm'][idx] for idx in new_img_index]
    new_seg_dict['est_dens'] = [hand_labeled_segdict['est_dens'][idx] for idx in new_img_index]
    new_seg_dict['param_dict'] = [hand_labeled_segdict['param_dict'][idx] for idx in new_img_index]
    #The images that correspond to the chosen indexes are saved in a new directory
    new_img_name_list, new_img_name_list_wodir = [], []
    img_name_list = [s for s in os.listdir(hand_labeled_dir) if s.endswith('.jpg')]
    new_img_name_list = [hand_labeled_dir + '/' + img_name for img_idx, img_name in enumerate(img_name_list) if img_idx in new_img_index]
    new_img_name_list_wodir = [img_name for img_idx, img_name in enumerate(img_name_list) if img_idx in new_img_index]
    for img_idx, img_name in enumerate(new_img_name_list): copyfile(img_name, new_dir + '/' + new_img_name_list_wodir[img_idx])
    copyfile(new_img_name_list[-1], new_dir + '/' + new_img_name_list_wodir[-1])
    #The new dictionary is saved in its new directory
    with open(new_dir + '/seg_dict.pickle', 'wb') as handle: pickle_dump(new_seg_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)

