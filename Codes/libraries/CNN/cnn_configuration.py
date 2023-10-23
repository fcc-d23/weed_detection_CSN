#LIBRARY FOR THE CONFIGURATION OF NEURAL NETWORKS
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Import of external libraries
import os, sys
import h5py

##KERAS##
from keras import backend as K
from keras.engine import saving
from keras import Input
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, MaxPooling2D, Dropout, Lambda, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop
from   keras.callbacks import ModelCheckpoint, EarlyStopping

##TENSORFLOW##
from tensorflow.compat.v1 import ConfigProto, GPUOptions, Session

##
##Import of own libraries
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from CNN.csn_functions import contrastive_loss, CSN_accuracy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Implementation of own functions

#Tensorflow session initialization function with or without GPU growth
def tensor_init ():
    config  = ConfigProto( gpu_options = GPUOptions(allow_growth=False) )
    session = Session(config = config)
    K.tensorflow_backend.set_session(session)
    return session

#Class that generates CNN_layer_options objects, with a method to add layers
class CNN_params():
    #The object is initialized with tuple of empty layers and number of layers 0, in addition to instantiating the type of network to be used and the number of labels for the architecture
    def __init__(self, CNN_class, class_n = 2, learning_rate = 0.001):
        self.conv2dlayer_tuple = []
        self.conv2dlayer_count = 0
        self.classlayer_list = []
        self.classlayer_count = 0
        self.CNN_class = CNN_class
        self.dropout_rate = 0.1
        self.TL_model = 'VGG'
        self.TL_ft = False
        self.class_n = class_n
        self.learning_rate = learning_rate  
    #Method for adding convolutional layers and increasing the layer counter
    def conv2dlayer_shape_adder(self, layer_shape):
        self.conv2dlayer_tuple.append(layer_shape)
        self.conv2dlayer_count = self.conv2dlayer_count + 1 
    #Method for deleting the convolution layer tuple
    def conv2dlayer_shape_clear(self):
        self.conv2dlayer_tuple = []
        self.conv2dlayer_count = 0
    #Method for adding classification layers
    def classlayer_shape_adder(self, layer_shape):
        self.classlayer_list.append(layer_shape)
        self.classlayer_count = self.classlayer_count + 1
    #Method for deleting the list of classification layers
    def classlayer_shape_clear(self):
        self.classlayer_list = []
        self.classlayer_count = 0
    #Method for changing the model to be used for Transfer Learning and the possibility of Fine-Tuning
    def TL_att_change(self, new_TL_model, new_TL_ft):
        self.TL_model = new_TL_model
        self.TL_ft = new_TL_ft

#Function to create the categorical focal loss definition for use in neural network training
"""
Created on Fri Oct 19 08:20:58 2018
@OS: Ubuntu 18.04
@IDE: Spyder3
@author: Aldi Faizal Dimara (Steam ID: phenomos)
"""
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss

#Class for generating CNN_train_options objects
class CNN_train_params():
    def __init__(self, batch_size, epochs, verbose, shuffle, patience, min_epochs = 100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.shuffle = shuffle
        self.patience = patience
        self.min_epochs = min_epochs

#SOURCE CODE: https://www.kaggle.com/ekkus93/keras-models-as-datasets-test
def load_split_weights(model, model_path_pattern='model_%d.h5', memb_size=102400000):  
    """Loads weights from split hdf5 files.
    
    Parameters
    ----------
    model : keras.models.Model
        Your model.
    model_path_pattern : str
        The path name should have a "%d" wild card in it.  For "model_%d.h5", the following
        files will be expected:
        model_0.h5
        model_1.h5
        model_2.h5
        ...
    memb_size : int
        The number of bytes per hdf5 file.  
    """
    model_f = h5py.File(model_path_pattern, "r", driver="family", memb_size=memb_size)
    saving.load_weights_from_hdf5_group_by_name(model_f, model.layers)
    
    return model

#Function to load the CNN model to be used for transfer learning
def model_load (img_shape, model_name = 'VGG', fine_tuning = False):
    #The selected model is loaded, by default VGG19
    if model_name == 'VGG':
        from keras.applications.vgg19 import VGG19
        model = VGG19(include_top=False, input_shape= img_shape, weights = None)
        keras_models_dir = 'keras_models'
        model_path_pattern = keras_models_dir + "/vgg19_weights_tf_dim_ordering_tf_kernels_%d.h5" 
        model = load_split_weights(model, model_path_pattern = model_path_pattern)
        output_VGG = model(model.inputs)
        shape_list = output_VGG.shape.as_list()
        VGG_num = shape_list[-1] * shape_list[-2] * shape_list[-3]
    #All convolutional layers are frozen to train only on the final NN layer if fine-tuning is not desired
    if fine_tuning == False: 
        for layer in model.layers: layer.trainable = False   
    else:
        for layer in model.layers:layer.trainable = True
    return model, VGG_num

#Creador de redes fully CNN, con la opción de ser CNN, CSN o basadas en TL
def fully_CNN_creator(CNN_params_obj, img_shape, contrastive_loss_fcn_obj = [], autoencoder = False, GAP = False):
    session = tensor_init()
    #The layers of the CNN_options object are extracted
    conv2dlayers_shape = CNN_params_obj.conv2dlayer_tuple
    classlayers_shape = CNN_params_obj.classlayer_list
    CNN_class = CNN_params_obj.CNN_class
    droprate = CNN_params_obj.dropout_rate
    class_n = CNN_params_obj.class_n
    learning_rate = CNN_params_obj.learning_rate
    input_shape = img_shape
    #Variables to be used during the creation of the CNN's
    kernel_size_list = []
    kernel_size_ctr = 0
    model_ctr = 0
    input_shape_ctr = 0
    first_transposed = 0
    first_layer = True
    #If Transfer-Learning was chosen, the loading of the model and the addition of classification layers follows
    if CNN_class == 'TL':
        #The model to be used for Transfer Learning is loaded
        CNN_model_name = CNN_params_obj.TL_model
        fine_tuning = CNN_params_obj.TL_ft
        model, _ = model_load (img_shape, model_name = CNN_model_name, fine_tuning = fine_tuning)
        #The output layer is obtained prior to the classification stage
        output_VGG = Flatten()(model(model.inputs))
        semantic_model = Model(inputs = model.inputs, output=output_VGG)
        #A layer is created that has the shape of the Flattened output of the loaded pre-trained model
        new_class = Sequential()
        output_VGG_dim =  model(model.inputs).shape.as_list()
        VGG_num = output_VGG_dim[-1] * output_VGG_dim[-2] * output_VGG_dim[-3]
        new_class_input = Input(shape = (VGG_num,) )
        new_class = new_class( new_class_input )
        #The classification layers are added to the pre-trained model, after which the complete model is compiled for training
        for layer_shape in classlayers_shape: new_class = Dense(layer_shape, activation='relu')(new_class)
        output = Dense(class_n, activation='softmax')(new_class)
        class_model = Model(inputs = new_class_input, output=output)
        class_model.compile(loss        = categorical_crossentropy,
                        optimizer   = RMSprop(lr = learning_rate),
                        metrics     = ['accuracy'])
    #If Transfer-Learning was not chosen, a custom CNN is built
    else:  
        #The convolutional stage to create the CNN is analyzed layer by layer
        for layer_shape in conv2dlayers_shape:
            #The characteristics of each layer are extracted and the default activation function of all layers is instantiated as RELU
            filter_n = layer_shape[-1]
            kernel_size =  input_shape[0] - layer_shape[0] + 1
            conv_act_str = 'relu'
            #If it is part of the ENCODER structure, a convolutional layer is added
            if input_shape[0] >= layer_shape[0] :
                kernel_size_list.append(kernel_size)
                current_layer = Sequential()
                current_layer.add( Conv2D(filter_n, kernel_size, activation = conv_act_str, input_shape = input_shape) )
                current_layer.add( MaxPooling2D () ) 
                current_layer.add( Dropout( droprate ) )      
                #The output of the current layer is set as the new input form, which due to MaxPooling corresponds to half the size of the conv2D layer output
                input_shape = [ int(layer_shape[0]/2),  int(layer_shape[0]/2), filter_n]
                first_layer = False
            #If it is part of the DECODER structure, a transposed convolutional or "deconvolutional" layer is added
            #NOTE: a deconvolutional layer is added only if the network has been chosen to have an ENCODER-DECODER structure
            else:
                if autoencoder:
                    #Arregla error de forma por el iterador input_shape_ctr
                    if first_transposed == 0:
                        input_shape_ctr = input_shape_ctr - 1
                        first_transposed = 1
                    kernel_size =  layer_shape[0] - input_shape[0] +  1
                    current_layer = Sequential()
                    current_layer.add( Conv2DTranspose(filter_n, kernel_size, activation = conv_act_str, input_shape = input_shape,kernel_initializer='glorot_uniform',\
                        bias_initializer = 'random_normal' ) )
                    current_layer.add( MaxPooling2D() )
                    current_layer.add( Dropout( droprate ) )
                    kernel_size_ctr = kernel_size_ctr + 1
                    #The output of the current layer is set as the new input form, again taking into account the MaxPooling effect
                    input_shape = [int(layer_shape[0]/2), int(layer_shape[0]/2), filter_n]
            #If it is the first layer created, the model corresponds only to that structure
            if model_ctr == 0:
                model_ctr = 1
                model_input = Input(shape=(img_shape[0],img_shape[1], img_shape[2]))
                model_in = current_layer(model_input) 
            #If not, it is added to the previous layers
            else: model_in = current_layer ( model_in )    
        if autoencoder:
            #The output layer is created and returned to the original shape of the image for semantic segmentation
            kernel_size =  -input_shape[0] + img_shape[0] +  1
            model_out = Conv2DTranspose(img_shape[-1], kernel_size, activation = conv_act_str, input_shape = input_shape,kernel_initializer='glorot_uniform',\
                bias_initializer = 'random_normal')
            model_out = model_out ( model_in )
        else: model_out = model_in
        #If a CNN has been chosen, the model created without the classification stage is compiled
        if CNN_class == 'CNN':
            semantic_model = Model( inputs = model_input, output = model_out )
            semantic_model.compile(loss        = categorical_focal_loss(gamma=2.0, alpha=0.25),
                                optimizer   = RMSprop(lr = learning_rate),
                                metrics     = ['accuracy'])
            #The first flattened layer is created to classify
            #If it was chosen to use Global Average Pooling
            if GAP: flat_in = GlobalAveragePooling2D()(semantic_model(semantic_model.inputs))
            #If it was not chosen to use Global Average Pooling
            else: flat_in = Flatten()(semantic_model(semantic_model.inputs))
            #For each layer that has been instantiated the corresponding tensor is added
            class_act_str = 'relu'
            new_layer = flat_in
            for layer in classlayers_shape: new_layer =  Dense ( layer, activation = class_act_str,bias_initializer='zeros' ) ( new_layer )
            #Finally, the layer of "n" neurons is added to calculate the probability of class membership
            class_out = Dense( class_n, activation='softmax' )( new_layer ) 
            
            class_model =  Model( inputs = semantic_model.inputs, output = class_out )
            class_model.compile(loss        = categorical_crossentropy,
                                optimizer   = RMSprop(lr = learning_rate),
                                metrics     = ['accuracy'])
        #On the other hand, if a CSN was chosen, it is created as follows
        else:
            semantic_model = Model( inputs = model_input, output = model_out )
            #The first flattened layer is created to classify
            #If it was chosen to use Global Average Pooling
            if GAP: flat_in = GlobalAveragePooling2D()(semantic_model(semantic_model.inputs))
            #If it was not chosen to use Global Average Pooling
            else: flat_in = Flatten()(semantic_model(semantic_model.inputs))
            #For each layer that has been instantiated the corresponding tensor is added
            class_act_str = 'relu'
            new_layer = flat_in
            for layer in classlayers_shape: new_layer =  Dense ( layer, activation = class_act_str, bias_initializer='zeros' ) ( new_layer )
            #In this case the last layer is omitted
            class_model =  Model( inputs = semantic_model.inputs, output = new_layer )
            #The "left" and "right" channels are created through which the images will enter the model
            left_input = Input(shape = img_shape)
            right_input = Input(shape = img_shape)
            
            #The output shape of the pure features is used
            semantic_model =  Model( inputs = semantic_model.inputs, output = flat_in )
            encoded_l =  class_model (left_input)
            encoded_r = class_model (right_input)
            #The similarity function between the features to be obtained is created
            #If L2 norm is chosen
            L2_layer = Lambda(lambda tensors:K.sqrt ( K.sum ( K.square ( tensors[0] - tensors[1] ), axis=1, keepdims=True ) ) )
            L2_distance = L2_layer([encoded_l, encoded_r])
            siamese_net = Model([left_input, right_input], L2_distance)
            rms = RMSprop(lr = learning_rate)
            #The model compilation is differentiated based on whether a contrastive loss with custom margin is chosen or not
            if contrastive_loss_fcn_obj == []: siamese_net.compile(loss = contrastive_loss, optimizer=rms, metrics=[CSN_accuracy])    
            else: siamese_net.compile(loss = contrastive_loss_fcn_obj.contrastive_loss, optimizer=rms, metrics = [contrastive_loss_fcn_obj.accuracy])
            return session, semantic_model, siamese_net     
    return session, semantic_model, class_model

#Callbacks for neural network training are defined
#BASED ON DOMINGO MERY CODE, (c) D.Mery, 2019
def defineCallBacks(model_file, patience, CSN = False):
    if CSN:
        callbacks = [
            EarlyStopping(
                monitor        = 'val_CSN_accuracy', 
                patience       = patience,
                mode           = 'max',
                verbose        = 0),
            ModelCheckpoint(model_file,
                monitor        = 'val_CSN_accuracy', 
                save_best_only = True, 
                mode           = 'max',
                verbose        = 1)
        ]    
    else:
        callbacks = [
            EarlyStopping(
                monitor        = 'val_acc', 
                patience       = patience,
                mode           = 'max',
                verbose        = 0),
            ModelCheckpoint(model_file,
                monitor        = 'val_acc', 
                save_best_only = True, 
                mode           = 'max',
                verbose        = 1)
        ]
    return callbacks