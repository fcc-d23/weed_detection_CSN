#LIBRERÍA PARA LA EXTRACCIÓN DE CARACTERÍSTICAS PARA SU USO DE SVM
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
from shutil import rmtree
from itertools import permutations
from skimage.feature import hog, local_binary_pattern as lbp
from sklearn.svm import SVC

##PYBALU##
from pybalu import feature_selection as fs
from pybalu import feature_transformation as ft

##
##Importe de librerías propias
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

#Parser para obtener automáticamente la clase de las imágenes dependiendo del nombre de la carpeta donde se encuentren, y asignación numérica dependiendo de la cantidad de objetos
def multiclassparser(multiclass_folder_list):

    sources_list = []
    class_name_list = []
    
    #Si las clases están en más de una carpeta raíz, se deben añadar los directorios aparte
    if isinstance(multiclass_folder_list, list):
        
        for folder in multiclass_folder_list:
            for name in os.listdir(folder):
                if os.path.isdir(folder + '/' + name): 
                    
                    sources_list.append(folder + '/' + name)
                    class_name_list.append(name)
                    
    #En caso contrario, se apendizan los directorios y nombres de clase directamentes           
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

#TODO: de momento esta función solo puede funcionar (valga la redundancia) con nri uniforme, con 8 vecinos y radio 1. Esto se debe al número 'k' de descriptores posibles que no es fácilmente calculable, 
#averiguar luego cómo generalizar esto para extender el uso a más de 58 descriptores
#Función que calcula los valores de LBP        
def lbp_feats_calc(img, frame_size = [10, 10], lbp_type = 'nri_uniform', lbp_neighbours = 8, lbp_radius = 1):
     
    #Se lee el tamaño de la imagen y se obtiene la cantidad de cuadros que se recorrerán dado el tamaño de frame ingresado
    img_shape = img.shape
    frames_x = int (img_shape[1] / frame_size[1])
    frames_y = int (img_shape[0] / frame_size[0] )
    lbp_mat = np.zeros( (58, frames_y, frames_x ) )
    lbp_vec = []
    
    #Se recorre la imagen en el tamaño de frame especificado
    for j in range(0, frames_x):
        
        for i in range(0, frames_y):
            
            lbp_vec = []
            
            #Se crea el vector de lbp con el método especificado
            lbp_frame = np.ravel(lbp(img[i*frame_size[0]:(i+1)*frame_size[0], j*frame_size[1]:(j+1)*frame_size[1]], lbp_neighbours, lbp_radius, method= lbp_type))
                
            for k in range(0, 58):
                    
                lbp_vec.append( len ( np.where(lbp_frame==k)[0] ) )
                                
            lbp_vec = list (lbp_vec / ( np.linalg.norm(lbp_vec ) + 1e-6 ) )
            lbp_mat[:,i,j] = lbp_vec
                            
    lbp_final = np.array([])
    for k in range(0, 58): lbp_final = np.concatenate ( ( lbp_final, np.ravel( lbp_mat[k,:,:], order='F' ) ) ) 
    return lbp_final

def hog_feats_calc(img, v_cell_size = 8, h_cell_size = 8, v_block_size = 2, h_block_size = 2, orientations_n = 9):
    

    hog_feats = hog(img, orientations = orientations_n, pixels_per_cell = ( h_cell_size, v_cell_size ) ,\
                    cells_per_block = (h_block_size, v_block_size), multichannel = False, visualize = False, feature_vector = True)
    
    return hog_feats

#Extracción y creación de la matriz de features a partir de las imágenes contenidas en la carpeta 'source'
def feature_extraction(source_list, destiny, lbp_r = 1, lbp_points = 8, lbp_type = 'nri_uniform', lbp_frame_size = [10, 10],\
    pixels_x = 8, pixels_y = 8, block_num_x = 2, block_num_y = 2, orientations_n = 9, img_shape = [256, 256]):
    
    #Se leen las carpetas que contienen las imágenes a ser clasificadas
    sources_list, class_name_list = multiclassparser(source_list)
    #Si alguna de las entradas no corresponde a una lista o tienen largos distintos se retorna un mensaje de error
    if ( not isinstance(sources_list, list) ) or ( not isinstance(class_name_list, list) ): raise ValueError('Una de las entradas no corresponde a una lista')
    
    else: 
        if len(sources_list) != len(class_name_list): raise ValueError('Las listas no tienen el mismo largo')             

    #Se cuentan las clases distintas y se asignan valores para cada una
    class_dict = {'class_name_list' : list(dict.fromkeys(class_name_list)), 'class_n_list' : list (range(0, len( list(dict.fromkeys(class_name_list)) ) )) }
    
    #Se recorren todas las carpetas proporcionadas y se convierten en matrices X e Y
    for source_count, source in enumerate(sources_list):
        
        #Se obtiene el valor asignado a la clase correspondiente
        class_n = class_dict['class_name_list'].index(class_name_list[source_count])
        #Se obtienen los nombres de las imágenes contenidas en la carpeta procesada, y se transforman en matriz
        img_name_list = [ source + '/' + s for s in os.listdir(source) if (s.endswith('.jpg') or s.endswith('.jpeg') or s.endswith('.png') )]
        for img_count, img_name in enumerate(img_name_list):
            
            #Transformación a escala de grises y cálculo de características HOG y LBP
            img_gray_eq = cv2.equalizeHist( cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY) ) / 255
            img_gray = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2GRAY)
                        
            lbp_total = np.asarray( lbp_feats_calc(img_gray_eq, frame_size = lbp_frame_size, lbp_type = lbp_type, lbp_neighbours = lbp_points, lbp_radius = lbp_r) )
            hog_total = np.asarray( hog_feats_calc(img_gray_eq, v_cell_size = pixels_y, h_cell_size = pixels_x, v_block_size = block_num_y, h_block_size = block_num_x, orientations_n = orientations_n) )
            #Si es la primera imagen procesada, se crean los vectores de características y de etiquetas
            if img_count == 0 and source_count == 0: 
                
                X = np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1)
                Y = np.array( [ class_n ] )
                
            else:
                
                X = np.concatenate( ( X,  np.concatenate( ( np.reshape(lbp_total, (1, lbp_total.shape[0]) ), np.reshape( hog_total, (1, hog_total.shape[0]) ) ) , axis = 1) ), axis = 0 )
                Y = np.concatenate( (Y, np.array( [ class_n ] ) ), axis = 0 )    
                #cv2.imshow('img_gray_eq', img_gray_eq), cv2.imshow('img_gray', img_gray), print(class_n, img_gray_eq.shape), cv2.waitKey(0)
    
    return X, Y, class_dict

#Creador de conjuntos de entrenamiento, prueba y validación dados por la proporción indicada
def multiclass_SVM_set_creator(X, Y, class_dict, data_dir, train_test_rate = 0.75, test_val_rate = 0.5):
    
    #Se verifica que X e Y contengan la misma cantidad de ejemplos
    if X.shape[0] != Y.shape[0]: raise ValueError ('Por favor ingrese conjuntos correspondientes')
    
    #Se cargan los valores asignados a las clases a partir del diccionario correspondiente y se crean los arrays que contendrán los conjuntos a crear
    class_n_list = class_dict['class_n_list']
    X_train = np.zeros( (1, X.shape[1]), dtype = np.uint8)
    Y_train = np.zeros((1,), dtype = np.uint8)
    X_val = np.zeros( (1, X.shape[1]), dtype = np.uint8)
    Y_val = np.zeros((1,), dtype = np.uint8)
    X_test = np.zeros( (1, X.shape[1]), dtype = np.uint8)
    Y_test = np.zeros((1,), dtype = np.uint8)
    
    #Se separan los conjuntos correspondientes a clase de forma aleatoria con una semilla
    for class_n in class_n_list:
        
        #Para cada clase se ordenan aleatoriamente los ejemplos con la misma semilla
        seed(1)   
        Y_index = np.where(Y == class_n)[0]
        np.random.shuffle(Y_index)
        
        #Se crean los conjuntos correspondientes
        Y_train = np.concatenate( ( Y_train, Y[Y_index[ 0 : int( np.round( train_test_rate * Y_index.shape[0]) ) ]] ), axis = 0 )
        X_train = np.concatenate( ( X_train, X[Y_index[ 0 : int( np.round( train_test_rate * Y_index.shape[0] ) ) ] , : ] ), axis = 0 )
        Y_val = np.concatenate( ( Y_val, Y[Y_index[ int( np.round( train_test_rate * Y_index.shape[0]) ) : \
            int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) ]] ), axis = 0 )
        X_val = np.concatenate( ( X_val, X[Y_index[ int( np.round( train_test_rate * Y_index.shape[0]) ) : \
            int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) ] , :] ), axis = 0 )
        Y_test = np.concatenate( ( Y_test, Y[Y_index[ int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) : \
             -1]] ), axis = 0 )
        X_test = np.concatenate( ( X_test, X[Y_index[ int( np.round( (train_test_rate-train_test_rate*test_val_rate+test_val_rate) * Y_index.shape[0] ) ) : \
             -1] , :] ), axis = 0 )
    
    X_train = np.concatenate( ( X_train[1:-1,:], np.reshape(X_train[-1,:], [1, X_train.shape[1]]) ) , axis = 0)
    Y_train = np.concatenate( (Y_train[1:-1], np.reshape(Y_train[-1], [1])), axis = 0 )
    
    X_val = np.concatenate( ( X_val[1:-1,:], np.reshape(X_val[-1,:], [1, X_val.shape[1]]) ) , axis = 0)
    Y_val = np.concatenate( (Y_val[1:-1], np.reshape(Y_val[-1], [1])), axis = 0 )
    
    X_test = np.concatenate( ( X_test[1:-1,:], np.reshape(X_test[-1,:], [1, X_test.shape[1]]) ) , axis = 0)
    Y_test = np.concatenate( (Y_test[1:-1], np.reshape(Y_test[-1], [1])), axis = 0 )
     
    #Se crea y guarda un diccionario con los conjuntos creados
    os.makedirs(data_dir, exist_ok = True)
    with open(data_dir + '/data_dic.pickle', 'wb') as handle: pickle_dump({'X_train' : X_train, 'X_val' : X_val, 'X_test' : X_test, 'Y_train' : Y_train, 'Y_val' : Y_val, 'Y_test' : Y_test, 'class_dict' : class_dict}\
        , handle, protocol=pickle_HIGHEST_PROTOCOL)
                
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict


##BORRAR DESPUÉS
if __name__ == '__main__':
    
    source_list = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/multiclass'
    destiny = 'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/caca'

    X,Y,class_dict = feature_extraction(source_list, destiny, lbp_r = 1, lbp_points = 8, lbp_type = 'nri_uniform', lbp_frame_size = [10, 10],\
        pixels_x = 8, pixels_y = 8, block_num_x = 2, block_num_y = 2, orientations_n = 9, img_shape = [256, 256])

    X_train, X_val, X_test, Y_train, Y_val, Y_test, class_dict = multiclass_SVM_set_creator(X, Y, class_dict,\
        'C:/Users/calde/Dropbox/PUC/Mg Sc/Python/Codigos/Weed_detection/Tesis/Pasto/extract/caca', train_test_rate = 0.75, test_val_rate = 0.5)


    print(X_train.shape)
    print(Y_train)











