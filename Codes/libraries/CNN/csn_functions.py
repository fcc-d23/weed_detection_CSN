#LIBRARY FOR THE CREATION OF AUXILIARY FUNCTIONS FOR CONVOLUTIONAL SIAMESE  NETWORKS 
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Import of external libraries
import numpy as np
import os, sys
from pickle import dump as pickle_dump, HIGHEST_PROTOCOL as pickle_HIGHEST_PROTOCOL, load as pickle_load

##KERAS##
from keras import backend as K
from keras.models import load_model, Model, Sequential
from keras.optimizers import RMSprop

##
##Import of own libraries
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from data_handling.img_processing import slide_window_creator
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Implementation of own functions

#Class that allows creating contrastive loss and accuracy calculation functions for CSN with adjustable margin (m) and distance
class contrastive_loss_fcn():
    def __init__(self, margin, distance):
        self.m, self.d = margin, distance
    def contrastive_loss(self, y_true, y_pred):
        square_pred, margin_square = K.square(y_pred), K.square(K.maximum(self.m - y_pred, 0))
        return K.mean(y_true * square_pred / 2 + (1 - y_true) * margin_square / 2)
    def CSN_accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
        
#Contrastive loss function, source code: https://keras.io/examples/mnist_siamese/
##NOTE: THE LOSS FUNCTION MARGIN IS FIXED AT m = 1 IN THIS CONFIGURATIONN##
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred / 2 + (1 - y_true) * margin_square / 2)

#Source code: https://keras.io/examples/mnist_siamese/
def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

#Source code: https://keras.io/examples/mnist_siamese/
def CSN_accuracy(y_true, y_pred):
    
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

#OJO: Esta función está extremadamente mal escrita, ver si se ocupa
#Función para transformar las distancias calculadas por una CSN a predicciones de clase forma 1 y 0
def CSN_dist2pred(Y_dist, dist = 0.5):
    #Vector donde se guardarán los valores de las predicciones de distancia
    Y_pred = np.zeros( ( Y_dist.shape[0], 1 ) )
    #Para cada distancia se obtiene un 1 o 0
    for i in range(Y_dist.shape[0]) :
        distance = Y_dist[i]
        #Si la distancia es mayor al umbral se asigna un 0, si no, un 1
        Y_pred[i] = np.array([ 0 ]) if distance > 0.5 else np.array([ 1 ])  
    return(Y_pred)

#Function to find the centroid of features extracted by the final layer of the CSNs, prior to the distance calculation.
#WARNING with X_train and Y_train, they are images by themselves and the class they belong to, NOT pairs of CSN_X_train examples of equals (1's) and differents (0's)
def CSN_centroid_finder(X_train, Y_train, model, class_dict, save_data = False, save_dir = ''):
    semantic_net = Sequential()
    #The feature extraction model is built prior to the distance calculation
    for layer in model.layers[0:-1]: semantic_net.add(layer)   
    semantic_net = Model( inputs = semantic_net.inputs, output = semantic_net.outputs )
    semantic_net.compile(loss = contrastive_loss, optimizer=RMSprop(), metrics=[CSN_accuracy])
    #Predictions are calculated for the training set to create the feature centroids
    feat_pred = semantic_net.predict(X_train, batch_size = 16)    
    #The values of feat_pred are summed and averaged by class to determine a centroid for each one
    class_n_list = class_dict['class_n_list']
    feat_mean = np.zeros( ( len(class_n_list), feat_pred.shape[1] ) )
    for class_n in class_n_list:
        class_Y_indexes = np.where(Y_train == class_n)[0]
        feat_mean[class_n] = np.mean(feat_pred[class_Y_indexes,:], axis=0)
    #A dictionary is created at the requested address in case the data is to be saved
    if save_data:
        if not save_dir: raise ValueError('Ingrese una dirección de guardado')
        feat_dict = {'feat_mean': feat_mean, 'class_dict': class_dict}
        with open(save_dir + '/' + '/feat_dict.pickle', 'wb') as handle: pickle_dump(feat_dict, handle, protocol=pickle_HIGHEST_PROTOCOL)  
    return feat_mean, class_dict, semantic_net

#Function that assigns the class of the regions within a frame array for an image
def CSN_centroid_pred(sld_win_array, frame_array_tot, model, X_train, Y_train, class_dict, win_size = (256, 256),  thresh = 'mean_half'):
    #The average of the features and the semantic model are extracted
    feat_mean, class_dict, semantic_net = CSN_centroid_finder(X_train, Y_train, model, class_dict)
    #The features of the image pairs are predicted and the distance is calculated for each set of frames
    pred_feat_central = semantic_net.predict(sld_win_array[0, :, :, :], batch_size = 8)
    pred_feat_next = semantic_net.predict(sld_win_array[1, :, :, :], batch_size = 8)
    distance = np.sqrt ( np.sum( np.square(np.subtract(pred_feat_central, pred_feat_next)), axis = 1 ) )
    distance = distance / np.max(distance)
    pred_feat_tot = semantic_net.predict(frame_array_tot, batch_size = 8)
    #It is determined whether each pair of frames corresponds to the same segmentation class depending on the type of threshold chosen
    if thresh == 'mean_half': one_thresh = 0.5 * (0.5 + np.mean(distance) )
    elif thresh == 'fixed': one_thresh = 0.5
    elif thresh == 'mean': one_thresh = np.mean(distance)
    Y_pred = np.reshape (np.asarray([1 if d < one_thresh else 0 for d in distance]), (distance.shape[0], 1) )
    #In addition, specific classes (e.g. weeds, soil, grass) are assigned to the image frames entered
    Class_pred = np.zeros((0))
    distance_pred = np.zeros((0))
    for i in range(pred_feat_tot.shape[0] ):
        pred_feat_concat = np.zeros((0))
        for _ in range(feat_mean.shape[0]-1):
            if pred_feat_concat.shape[0] == 0: pred_feat_concat = np.concatenate( ( np.reshape (pred_feat_tot[i], (1, pred_feat_tot[i].shape[0] ) ),\
                    np.reshape (pred_feat_tot[i], (1, pred_feat_tot[i].shape[0] ) )  ), axis = 0 )
            else: pred_feat_concat = np.concatenate( ( pred_feat_concat,  np.reshape ( pred_feat_tot[i], (1, pred_feat_tot[i].shape[0] ) ) ), axis = 0 )
        if Class_pred.shape[0] == 0:
            Class_pred = np.array ( [np.argmin( np.sqrt ( np.sum( np.square( np.subtract( pred_feat_concat, feat_mean ) ), axis = 1 ) ) ) ]) 
            distance_pred = np.reshape( np.sqrt ( np.sum( np.square( np.subtract( pred_feat_concat, feat_mean ) ), axis = 1 ) ), (1, np.sqrt ( np.sum( np.square( np.subtract( pred_feat_concat, feat_mean ) ), axis = 1 ) ).shape[0]) )
        elif Class_pred.shape[0] > 0:
            Class_pred = np.concatenate( (Class_pred, np.array([np.argmin( np.sqrt ( np.sum( np.square( np.subtract( pred_feat_concat, feat_mean ) ), axis = 1 ) ) ) ] ) ) , axis = 0 )
            distance_pred = np.concatenate( ( distance_pred, \
                np.reshape( np.sqrt ( np.sum( np.square( np.subtract( pred_feat_concat, feat_mean ) ), axis = 1 ) ), (1, np.sqrt ( np.sum( np.square( np.subtract( pred_feat_concat, feat_mean ) ), axis = 1 ) ).shape[0]) )), axis=0) 
    return Y_pred, Class_pred, distance_pred