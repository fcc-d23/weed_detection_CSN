#DATA PROCESSING LIBRARY FOR USE IN NEURAL NETWORKS
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Import of external libraries
import time
import numpy as np
import cv2
import os, sys
from pickle import load as pickle_load, dump as pickle_dump, HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL
from numpy.random import seed, shuffle

##KERAS##
from   keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.models import Model

##
#Import of external libraries
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from CNN.cnn_configuration import model_load
from Interfaz.interfaces import YNC_prompt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Implementation of own functions

#Function to parse and automatically obtain the class of the images depending on the name of the folder where they are located, and numerical assignment depending on the number of objects
def multiclassparser(multiclass_folder_list):
    sources_list = []
    class_name_list = []
    #If the classes are in more than one root folder, separate directories must be added
    if isinstance(multiclass_folder_list, list):
        for folder in multiclass_folder_list:
            for name in os.listdir(folder):
                if os.path.isdir(folder + '/' + name): 
                    sources_list.append(folder + '/' + name)
                    class_name_list.append(name) 
    #Otherwise, the directories and class names are directly appended           
    else:
        sources_list = [multiclass_folder_list + '/' + name for name in os.listdir(multiclass_folder_list) if os.path.isdir(multiclass_folder_list + '/' + name)]
        class_name_list = [name for name in os.listdir(multiclass_folder_list) if os.path.isdir(multiclass_folder_list + '/' + name)]
    return sources_list, class_name_list

#Function for image preprocessing to arrays of X and Y for all classes within a list
def multiclass_preprocessing(sources_list, class_name_list, image_size = (64,64,3), weed_list_bool = False):
    #If any of the entries do not correspond to a list or they have different lengths, an error message is raised
    if ( not isinstance(sources_list, list) ) or ( not isinstance(class_name_list, list) ): raise ValueError('One of the entries does not correspond to a list')
    else: 
        if len(sources_list) != len(class_name_list): raise ValueError('Lists are not the same length')             
    #The name of each class is asked whether it corresponds to weeds or non-weeds
    weed_list = []
    if weed_list_bool:
        for class_name in class_name_list:
            YN_class = YNC_prompt('¿Does: "' + class_name + '" belong to weed class?')
            weed_list.append(YN_class.exec_())
    #All classes are counted and values are assigned to each one
    class_dict = {'class_name_list' : list(dict.fromkeys(class_name_list)), 'class_n_list' : list (range(0, len( list(dict.fromkeys(class_name_list)) ) )), 'weed_class_list' : weed_list }
    #The matrix is created for the images with the chosen size, with its corresponding vector of classes
    X = np.zeros( ( 1, image_size[0], image_size[1], image_size[2] ), dtype = np.uint8)
    Y = np.zeros( ( 1, 1 ), dtype = np.uint8) 
    #All the folders provided are traversed and converted into X and Y matrices
    for source_count, source in enumerate(sources_list):
        #The value assigned to the corresponding class is obtained
        class_n = class_dict['class_name_list'].index(class_name_list[source_count])
        #The names of the images contained in the processed folder are obtained, and the images are transformed into matrices
        img_name_list = [ source + '/' + s for s in os.listdir(source) if (s.endswith('.jpg') or s.endswith('.jpeg') or s.endswith('.png') )]
        for img_name in img_name_list:
            X = np.concatenate( (X, np.reshape( cv2.resize( cv2.imread(img_name), (image_size[0], image_size[1]) ) , (1, image_size[0], image_size[1], image_size[2])) ), axis = 0)
            Y = np.concatenate( (Y, np.array([[class_n]])), axis = 0 )
    #The first value of the matrix and vector (since they are zero) are eliminated
    X = np.concatenate((X[1:-1,:,:,:], np.reshape ( X[-1,:,:,:], (1, image_size[0], image_size[1], image_size[2]))), axis = 0)
    Y = np.concatenate((Y[1:-1,:], np.reshape ( Y[-1,:], (1, 1))), axis = 0)
    return X, Y, class_dict

#Function to create training, test and validation sets given by the specified ratio
def multiclass_CNN_set_creator(X, Y, class_dict, data_dir, train_test_rate = 0.75, test_val_rate = 0.5, TL = False, TL_model_name = 'VGG'):
    #It is verified that X and Y contain the same amount of examples
    if X.shape[0] != Y.shape[0]: raise ValueError ('Please enter corresponding sets')
    #The values assigned to the classes are loaded from the corresponding dictionary and the arrays containing the sets are created
    class_n_list = class_dict['class_n_list']
    X_train = np.zeros( (1, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
    Y_train = np.zeros((1,  Y.shape[1]))
    X_val = np.zeros( (1, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
    Y_val = np.zeros((1, Y.shape[1]))
    X_test = np.zeros( (1, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
    Y_test = np.zeros((1, Y.shape[1]))
    
    #The sets corresponding to each class are randomly separated with the same seed
    for class_n in class_n_list:
        seed(1)   
        Y_index = np.where(Y == class_n)[0]
        np.random.shuffle(Y_index)
        #Corresponding sets are created
        Y_train = np.concatenate( ( Y_train, Y[Y_index[ 0 : int( np.round( train_test_rate * Y_index.shape[0]) ) ], :] ), axis = 0 )
        X_train = np.concatenate( ( X_train, X[Y_index[ 0 : int( np.round( train_test_rate * Y_index.shape[0] ) ) ] , : , : , :] ), axis = 0 )
        Y_val = np.concatenate( ( Y_val, Y[Y_index[ int( np.round( train_test_rate * Y_index.shape[0]) ) : \
            int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) ], :] ), axis = 0 )
        X_val = np.concatenate( ( X_val, X[Y_index[ int( np.round( train_test_rate * Y_index.shape[0]) ) : \
            int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) ] , : , : , :] ), axis = 0 )
        Y_test = np.concatenate( ( Y_test, Y[Y_index[ int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) : \
             -1], :] ), axis = 0 )
        X_test = np.concatenate( ( X_test, X[Y_index[ int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) : \
             -1] , : , : , :] ), axis = 0 )
        Y_test = np.concatenate( (Y_test, np.reshape( Y[Y_index[-1], :], (1, Y.shape[1]) ) ), axis = 0 )
        X_test = np.concatenate( (X_test, np.reshape( X[Y_index[-1], :, :, :] ,( 1, X.shape[1], X.shape[2], X.shape[3] ) ) ), axis = 0 )
    #The value -1 is added, which is left out of the arrays by the NumPy method
    X_train = np.concatenate( ( X_train[1:-1,:,:,:], np.reshape(X_train[-1,:,:,:], [1, X_train.shape[1], X_train.shape[2], X_train.shape[3]]) ) , axis = 0)
    Y_train = np.concatenate( (Y_train[1:-1,:], np.reshape(Y_train[-1,:], [1, Y_train.shape[1]])), axis = 0 )
    X_val = np.concatenate( ( X_val[1:-1,:,:,:], np.reshape(X_val[-1,:,:,:], [1, X_val.shape[1], X_val.shape[2], X_val.shape[3]]) ) , axis = 0)
    Y_val = np.concatenate( (Y_val[1:-1,:], np.reshape(Y_val[-1,:], [1, Y_val.shape[1]])), axis = 0 )
    X_test = np.concatenate( ( X_test[1:-1,:,:,:], np.reshape(X_test[-1,:,:,:], [1, X_test.shape[1], X_test.shape[2], X_test.shape[3]]) ) , axis = 0)
    Y_test = np.concatenate( (Y_test[1:-1,:], np.reshape(Y_test[-1,:], [1, Y_test.shape[1]])), axis = 0 )
    #If it was also chosen to save data from a Transfer Learning network, the transformation of the images to features is carried out
    if TL == True:
        #The model is loaded and the inputs are formatted to match the chosen image size
        TL_model, _ = model_load ((X.shape[1], X.shape[2], X.shape[3]), model_name = TL_model_name, fine_tuning = False)
        new_TL_model = Flatten() ( TL_model(TL_model.inputs ) )
        new_TL_model = Model( input = TL_model.inputs, output = new_TL_model )
        X_train = new_TL_model.predict( X_train )
        X_val = new_TL_model.predict( X_val )
        X_test = new_TL_model.predict( X_test )        
        os.makedirs(data_dir, exist_ok = True)
        with open(data_dir + '/data_dic.pickle', 'wb') as handle: pickle_dump({'X_train' : X_train, 'X_val' : X_val, 'X_test' : X_test,\
            'Y_train' : Y_train, 'Y_val' : Y_val, 'Y_test' : Y_test, 'class_dict' : class_dict, 'img_shape' : (X.shape[1], X.shape[2])} , handle, protocol=pickle_HIGHEST_PROTOCOL)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict
    #A dictionary is saved with the created sets
    os.makedirs(data_dir, exist_ok = True)
    with open(data_dir + '/data_dic.pickle', 'wb') as handle: pickle_dump({'X_train' : X_train, 'X_val' : X_val, 'X_test' : X_test, 'Y_train' : Y_train, 'Y_val' : Y_val, 'Y_test' : Y_test, 'class_dict' : class_dict}\
        , handle, protocol=pickle_HIGHEST_PROTOCOL)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict

#Image matching function for use in CSN type networks
def CSN_pair_creator(X, Y, class_dict, pair_all = True):
    #It is verified that X and Y contain the same amount of examples
    if X.shape[0] != Y.shape[0]: raise ValueError ('Por favor ingrese conjuntos correspondientes')
    #Necessary variables are initialized
    class_n_list = class_dict['class_n_list']
    CSN_X = np.zeros((2, 1, X.shape[1], X.shape[2], X.shape[3]), dtype = np.uint8)
    CSN_Y = np.zeros((1, Y.shape[1]))
    #All classes are traversed to create the pairs
    for class_n in class_n_list:
        #Matching pairs are found and concatenated and assigned the value '1' to designate that they belong to the same class
        class_n_index = np.where(Y == class_n)[0]
        seed(1)
        shuffle(class_n_index)
        for i in range(0, int(class_n_index.shape[0]/2)):
            X_1 = np.reshape(X[class_n_index[2*i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
            X_2 = np.reshape(X[class_n_index[2*i+1],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
            X_i = np.concatenate( (X_1, X_2), axis = 0 )
            CSN_X = np.concatenate( (CSN_X, X_i), axis = 1 )
            CSN_Y = np.concatenate( ( CSN_Y, np.array( [ [1] ] ) ), axis = 0 )
        #Now the negative examples are matched, starting with the current class and continuing with the following ones
        not_class_list = list(range(class_n+1, class_n_list[-1]+1))
        for not_class_n in not_class_list:
            not_class_n_index = np.where(Y == not_class_n)[0]
            shuffle(class_n_index)
            #If it was chosen to match all examples
            if pair_all:
                #If the current class contains more examples than the 'current non-class', iterate until the 'class' is finished and 'restart' the 'non-class'
                not_idx = 0
                if len(not_class_n_index) < len(class_n_index):
                    for i in range( len(class_n_index)):
                        X_1 = np.reshape(X[class_n_index[i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_2 = np.reshape(X[not_class_n_index[not_idx],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_i = np.concatenate( (X_1, X_2), axis = 0 )
                        CSN_X = np.concatenate( (CSN_X, X_i), axis = 1 )
                        CSN_Y = np.concatenate( ( CSN_Y, np.array( [ [0] ] ) ), axis = 0 )
                        #If the number of examples of the 'non-class' is exceeded, we start from 0 in this list, otherwise the counter is increased by 1
                        not_idx = not_idx + 1 if not_idx < len(not_class_n_index)-1 else 0          
                #If the 'current non-class' contains more examples than the current class, iterate until the 'non-class' is finished and eventually reset the class counter
                else:
                    class_idx = 0
                    for i in range(0, len(not_class_n_index)):
                        X_1 = np.reshape(X[class_n_index[class_idx],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_2 = np.reshape(X[not_class_n_index[i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_i = np.concatenate( (X_1, X_2), axis = 0 )
                        CSN_X = np.concatenate( (CSN_X, X_i), axis = 1 )
                        CSN_Y = np.concatenate( ( CSN_Y, np.array( [ [0] ] ) ), axis = 0 )
                        #If the number of examples of the class is exceeded, the list starts from 0, otherwise the counter is increased by 1
                        class_idx = class_idx + 1 if class_idx < len(class_n_index)-1 else 0   
            #If it is only matched up to the example of the class with the least data
            else:
                #If the current class contains more examples than the 'current non-class', iterate until the 'non-class' is finished
                not_idx = 0
                if len(not_class_n_index) < len(class_n_index):
                    for i in range( len(not_class_n_index)):
                        X_1 = np.reshape(X[class_n_index[i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_2 = np.reshape(X[not_class_n_index[i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_i = np.concatenate( (X_1, X_2), axis = 0 )
                        CSN_X = np.concatenate( (CSN_X, X_i), axis = 1 )
                        CSN_Y = np.concatenate( ( CSN_Y, np.array( [ [0] ] ) ), axis = 0 )                                            
                #If the 'current non-class' contains more examples than the current class, iterate until the 'class' is finished
                else:
                    class_idx = 0
                    for i in range(0, len(class_n_index)):
                        X_1 = np.reshape(X[class_n_index[i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_2 = np.reshape(X[not_class_n_index[i],:,:,:], (1, 1, X.shape[1], X.shape[2], X.shape[3]))
                        X_i = np.concatenate( (X_1, X_2), axis = 0 )
                        CSN_X = np.concatenate( (CSN_X, X_i), axis = 1 )
                        CSN_Y = np.concatenate( ( CSN_Y, np.array( [ [0] ] ) ), axis = 0 )    
    #X and Y variables are created and returned, and the classification dictionary is updated to 'equal' or 'different'
    CSN_X = np.concatenate( ( CSN_X[:, 1:-1, :, :, :], np.reshape( CSN_X[:, -1, :, :, :] , (2, 1, X.shape[1], X.shape[2], X.shape[3])) ), axis = 1 )
    CSN_Y = np.concatenate( ( CSN_Y[1:-1, :], np.reshape( CSN_Y[-1,:],  (1, 1) ) ), axis = 0 )
    CSN_class_dict = {'class_n_list' : [0, 1], 'class_name_list' :  ['different, equal']}
    return CSN_X, CSN_Y, class_dict, CSN_class_dict

#Function to create training, validation and test sets for CSN
def multiclass_CSN_set_creator(X, Y, class_dict, data_dir, train_test_rate = 0.25, test_val_rate = 0.5, pair_all = True):
    #The data paired as equal and different is loaded
    CSN_X, CSN_Y, class_dict, CSN_class_dict = CSN_pair_creator(X, Y, class_dict, pair_all = pair_all)
    CSN_class_list = CSN_class_dict['class_n_list']
    CSN_X_train = np.zeros( (CSN_X.shape[0], 1, CSN_X.shape[2], CSN_X.shape[3], CSN_X.shape[4]), dtype = np.uint8)
    CSN_Y_train = np.zeros((1,  CSN_Y.shape[1]))
    CSN_X_val = np.zeros( (CSN_X.shape[0], 1, CSN_X.shape[2], CSN_X.shape[3], CSN_X.shape[4]), dtype = np.uint8)
    CSN_Y_val = np.zeros((1, CSN_Y.shape[1]))
    CSN_X_test = np.zeros( (CSN_X.shape[0], 1, CSN_X.shape[2], CSN_X.shape[3], CSN_X.shape[4]), dtype = np.uint8)
    CSN_Y_test = np.zeros((1, CSN_Y.shape[1]))
    #Iterate over the two classes of data to create the training, validation and test sets
    for CSN_class_n in CSN_class_list:
        #For each class the examples are randomly ordered with the same seed 
        seed(1)   
        CSN_Y_index = np.where(CSN_Y == CSN_class_n)[0]
        shuffle(CSN_Y_index)
        #Corresponding sets are created
        CSN_Y_train = np.concatenate( ( CSN_Y_train, CSN_Y[CSN_Y_index[ 0 : int( np.round( train_test_rate * CSN_Y_index.shape[0]) ) ], :] ), axis = 0 )
        CSN_X_train = np.concatenate( ( CSN_X_train, CSN_X[:,CSN_Y_index[ 0 : int( np.round( train_test_rate * CSN_Y_index.shape[0] ) ) ] , : , : , :] ), axis = 1 )
        CSN_Y_val = np.concatenate( ( CSN_Y_val, CSN_Y[CSN_Y_index[ int( np.round( train_test_rate * CSN_Y_index.shape[0]) ) : \
            int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * CSN_Y_index.shape[0] ) ) ], :] ), axis = 0 )
        CSN_X_val = np.concatenate( ( CSN_X_val, CSN_X[:,CSN_Y_index[ int( np.round( train_test_rate * CSN_Y_index.shape[0]) ) : \
            int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * CSN_Y_index.shape[0] ) ) ] , : , : , :] ), axis = 1 )
        CSN_Y_test = np.concatenate( ( CSN_Y_test, CSN_Y[CSN_Y_index[ int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) *CSN_Y_index.shape[0] ) ) : \
             -1], :] ), axis = 0 )
        CSN_X_test = np.concatenate( ( CSN_X_test, CSN_X[:,CSN_Y_index[ int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * CSN_Y_index.shape[0] ) ) : \
             -1] , : , : , :] ), axis = 1 )  
        CSN_Y_test = np.concatenate( (CSN_Y_test, np.reshape( CSN_Y[CSN_Y_index[-1], :], (1, CSN_Y.shape[1]) ) ), axis = 0 )
        CSN_X_test = np.concatenate( (CSN_X_test, np.reshape( CSN_X[:, CSN_Y_index[-1], :, :, :] ,( 2, 1, X.shape[1], X.shape[2], X.shape[3] ) ) ), axis = 1 )
    #The value -1 is added, which is left out of the arrays by the NumPy method
    CSN_X_train = np.concatenate( ( CSN_X_train[:,1:-1,:,:,:], np.reshape(CSN_X_train[:,-1,:,:,:], [2, 1, CSN_X_train.shape[2], CSN_X_train.shape[3], CSN_X_train.shape[4]]) ) , axis = 1)
    CSN_Y_train = np.concatenate( (CSN_Y_train[1:-1,:], np.reshape(CSN_Y_train[-1,:], [1, CSN_Y_train.shape[1]])), axis = 0 )
    CSN_X_val = np.concatenate( ( CSN_X_val[:,1:-1,:,:,:], np.reshape(CSN_X_val[:,-1,:,:,:], [2, 1, CSN_X_val.shape[2], CSN_X_val.shape[3], CSN_X_val.shape[4]]) ) , axis = 1)
    CSN_Y_val = np.concatenate( (CSN_Y_val[1:-1,:], np.reshape(CSN_Y_val[-1,:], [1, CSN_Y_val.shape[1]])), axis = 0 )
    CSN_X_test = np.concatenate( ( CSN_X_test[:,1:-1,:,:,:], np.reshape(CSN_X_test[:,-1,:,:,:], [2, 1, CSN_X_test.shape[2], CSN_X_test.shape[3], CSN_X_test.shape[4]]) ) , axis = 1)
    CSN_Y_test = np.concatenate( (CSN_Y_test[1:-1,:], np.reshape(CSN_Y_test[-1,:], [1, CSN_Y_test.shape[1]])), axis = 0 )
    #A dictionary is saved with the created sets
    os.makedirs(data_dir, exist_ok = True)
    with open(data_dir + '/data_dic.pickle', 'wb') as handle: pickle_dump({'X_train' : CSN_X_train, 'X_val' : CSN_X_val, 'X_test' : CSN_X_test, 'Y_train' : CSN_Y_train,\
        'Y_val' : CSN_Y_val, 'Y_test' : CSN_Y_test, 'class_dict' : class_dict, 'CSN_class_dict' : CSN_class_dict}, handle, protocol=pickle_HIGHEST_PROTOCOL)
    return CSN_X_train, CSN_X_val, CSN_X_test, CSN_Y_train, CSN_Y_val, CSN_Y_test, class_dict

#Function to load training, validation and test sets from pre-built folders
def folder_set_creator(folder_list, destiny_folder, image_size = (64,64,3), CNN_type = 'CNN', strat_folder = ['train', 'val', 'test'], pair_all = True):
    #It is checked that the folders contain the requested sets
    if len(folder_list) != len(strat_folder): raise ValueError('Las carpetas no corresponden a los conjuntos pedidos')
    X_train = np.zeros([0])
    X_val = np.zeros([0])
    X_test = np.zeros([0])
    Y_train = np.zeros([0])
    Y_val = np.zeros([0])
    Y_test = np.zeros([0])
    #For each folder the requested sets are created
    for folder_idx, folder in enumerate(folder_list):
        sources_list, class_name_list = multiclassparser(folder)
        X, Y, class_dict = multiclass_preprocessing(sources_list, class_name_list, image_size = image_size)
        #If the target classifier is CNN, values are simply returned. If not, the pairs are constructed for CSN
        if CNN_type == 'CSN': X, Y, class_dict, CSN_class_dict = CSN_pair_creator(X, Y, class_dict, pair_all = pair_all)
        #If the folder is for training
        if strat_folder[folder_idx] == 'train':
            if X_train.shape[0] == 0:
                X_train = X
                Y_train = Y
            else:
                X_train = np.concatenate( ( X_train, X ), axis = 0)
                Y_train = np.concatenate( ( Y_train, Y ), axis = 0)   
        #If the folder is for validation 
        elif strat_folder[folder_idx] == 'val':  
            if X_val.shape[0] == 0:
                X_val = X
                Y_val = Y
            else:
                X_val = np.concatenate( ( X_val, X ), axis = 0)
                Y_val = np.concatenate( ( Y_val, Y ), axis = 0) 
        #If the folder is for testing
        elif strat_folder[folder_idx] == 'test':  
            if X_test.shape[0] == 0:
                X_test = X
                Y_test = Y
            else:
                X_test = np.concatenate( ( X_test, X ), axis = 0)
                Y_test = np.concatenate( ( Y_test, Y ), axis = 0) 
    #A dictionary is saved with the created sets
    os.makedirs(destiny_folder, exist_ok = True)
    with open(destiny_folder + '/data_dic.pickle', 'wb') as handle: \
        pickle_dump({'X_train' : X_train, 'X_val' : X_val, 'X_test' : X_test, 'Y_train' : Y_train, 'Y_val' : Y_val, 'Y_test' : Y_test, 'class_dict' : class_dict}\
        , handle, protocol=pickle_HIGHEST_PROTOCOL)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict

#General function to create sets, either for CNN or CSN
def multiclass_set_creator(X, Y, class_dict, data_dir, train_test_rate, test_val_rate, CNN_class = 'CNN', TL = False, TL_model_name = 'VGG', pair_all = True):
    #The set is created depending on the chosen name
    if CNN_class == 'CNN':
        X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = \
        multiclass_CNN_set_creator(X, Y, class_dict, data_dir, train_test_rate = train_test_rate, test_val_rate = test_val_rate, TL = TL, TL_model_name = TL_model_name)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict
    elif CNN_class == 'CSN':
        CSN_X_train, CSN_X_val, CSN_X_test, CSN_Y_train, CSN_Y_val, CSN_Y_test, class_dict = \
        multiclass_CSN_set_creator(X, Y, class_dict, data_dir, train_test_rate = train_test_rate, test_val_rate = test_val_rate, pair_all = pair_all)
        return CSN_X_train, CSN_X_val, CSN_X_test, CSN_Y_train, CSN_Y_val, CSN_Y_test, class_dict

#Function to load data created for training/testing of neural networks
#BASED ON DOMINGO MERY CODE, (c) D.Mery, 2019
def load_data(st_file, train = True, TL = False):
    print('loading training/testing data from '+ st_file +' ...')
    with open(st_file, 'rb') as handle: data = pickle_load(handle)
    X_train  = data['X_train']
    Y_train  = data['Y_train']
    if Y_train.shape[0] != 0: Y_train  = to_categorical(Y_train)
    print('X train size: {}'.format(X_train.shape))
    print('y train size: {}'.format(Y_train.shape))
    X_val   = data['X_val']
    Y_val   = data['Y_val']
    if Y_val.shape[0] != 0: Y_val   = to_categorical(Y_val)
    print('X val  size: {}'.format(X_val.shape))
    print('y val  size: {}'.format(Y_val.shape))
    X_test   = data['X_test']
    Y_test   = data['Y_test']
    if Y_test.shape[0] != 0: Y_test   = to_categorical(Y_test)
    print('X test  size: {}'.format(X_test.shape))
    print('y test  size: {}'.format(Y_test.shape))
    classes  = list(range(0, Y_train.shape[1]))
    #If you choose the data option for Transfer Learning
    if TL == True: return X_train, Y_train, X_val, Y_val, X_test, Y_test, classes, data['img_shape']
    else: return X_train, Y_train, X_val, Y_val, X_test, Y_test, classes, [], []