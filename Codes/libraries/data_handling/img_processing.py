#LIBRERÍA PARA EL PROCESAMIENTO DE IMÁGENES
#Felipe Calderara Cea (felipe.calderarac@gmail.com), Mg Sc PUC, 2021

##
#Importe de librerías externas
import numpy as np
from matplotlib import pyplot as plt
from pandas.core import frame
from scipy import ndimage
import cv2
import os, sys
from shutil import rmtree, copyfile
import warnings
import time

##
##Importe de librerías propias
code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]
sys.path.append(main_lib)

from Interfaz.interfaces import YNC_prompt, class_name
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Desarrollo de funciones propias

#Función de despliegue de imágenes
def img_display(img_rgb, title):

    plt.close()
    #####################################################
    ##Esencial para que la figura desplegada quede cómoda
    plt.rcParams["figure.figsize"] = (16* 1.5,9* 1.5) 
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("+0+0")
    #####################################################
    plt.imshow(img_rgb)
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.draw()
    plt.pause(0.01)
    
#Selección de esquinas dentro de una imagen mediante el click del mouse
def corner_selector(img_rgb, old_img = []):

    #Se despliega la imagen y se piden las esquinas que contienen el objeto de interés
    img_display(img_rgb, 'Escoja las dos esquinas que definen el espacio del objeto de interés')
    print('Escoja las dos esquinas que definen el espacio del objeto de interés')
    corners = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)

    #Se ordenan las esquinas escogidas en orden ascendente y se construye la máscara para el recorte de la imagen
    x = [ corners[0,0], corners[1,0] ]
    y = [ corners[0,1], corners[1,1] ]
    x.sort()
    y.sort()
    img_shape = img_rgb.shape

    #Se transforma la imagen a base HSV y se corta la imagen dadas las esquinas escogidas anteriormente
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    new_hsv_black = np.zeros([img_shape[0], img_shape[1], 3], dtype = np.uint8)
    
    #Si se ingresa la imagen sin modificaciones se procede a cortar sobre ella
    if isinstance(old_img, np.ndarray): 
        new_hsv_cut = cv2.cvtColor(old_img, cv2.COLOR_RGB2HSV) [y[0]:y[1],x[0]:x[1],:]
        new_hsv_black[:,:,0:3] = cv2.cvtColor(old_img, cv2.COLOR_RGB2HSV)[:,:,0:3]
        
    #Si no, se corta sobre la imagen ya modificada
    else:
        new_hsv_cut = img_hsv[y[0]:y[1],x[0]:x[1],:]
        new_hsv_black[:,:,0:3] = img_hsv[:,:,0:3]
    
    #Se construyen las nuevas imágenes recortadas
    new_hsv_display = np.zeros([img_shape[0], img_shape[1], 3], dtype = np.uint8)
    
    #Se crea la nueva imagen para desplegar con bounding boxes y la nueva imagen con espacios borrados para crear clases
    new_hsv_display[:,:,0:3] = img_hsv[:,:,0:3]
    
    new_hsv_display[y[0]: y[1],x[0]:x[0]+11,0] = 0
    new_hsv_display[y[0]: y[1],x[1]:x[1]+11,0] = 0
    new_hsv_display[y[0]:y[0]+11, x[0]:x[1],0] = 0
    new_hsv_display[y[1]:y[1]+11, x[0]:x[1],0] = 0
    
    new_hsv_display[y[0]: y[1],x[0]:x[0]+11,1:3] = 255
    new_hsv_display[y[0]: y[1],x[1]:x[1]+11,1:3] = 255
    new_hsv_display[y[0]:y[0]+11, x[0]:x[1],1:3] = 255
    new_hsv_display[y[1]:y[1]+11, x[0]:x[1],1:3] = 255
        
    new_hsv_black[y[0] : y[1], x[0] : x[1],:] = 0
    
    #Se vuelve a convertir de base HSV a RGB
    new_rgb_display = cv2.cvtColor(new_hsv_display, cv2.COLOR_HSV2RGB)
    new_rgb_black = cv2.cvtColor(new_hsv_black, cv2.COLOR_HSV2RGB)
    new_rgb_cut = cv2.cvtColor(new_hsv_cut, cv2.COLOR_HSV2RGB)
    
    return(new_rgb_cut, new_rgb_display, new_rgb_black, x, y)

#Función para etiquetar regiones de objetos de distintas clases
def region_tagging(origin_folder, destinies_folder, frame_size = (1800, 1800, 3), new_shape = (64, 64, 3), overwrite = False, include_crops = False, region_size = (1800, 1800, 3), region_alpha = 1.25,\
    black_fill = True, rand_cut = True):
  
    #Se lee la carpeta de origen de los datos y se adjuntan las imágenes que contengan un formato permitido
    name_list = os.listdir(origin_folder)
    new_name_list = [origin_folder + '/' + s for s in name_list if ( s.endswith('jpg') or s.endswith('png') or s.endswith('jpeg') )]

    #Si se elige sobreescribir (y se verifica que la carpeta de destino existe) se elimina dicho directorio
    if overwrite and os.path.isdir(destinies_folder): rmtree(destinies_folder)  
    
    #Se leen las imágenes y se etiqueta dentro de ellas
    for idx, im_path in enumerate(new_name_list): 
        img = cv2.cvtColor( cv2.imread(im_path), cv2.COLOR_BGR2RGB ) 
        old_img = img[:,:,:]
        ##################################################
        #Si se elige la inclusión de cosechas en el corte#
        ##################################################
        if include_crops:
            #Se pregunta si existe alguna maleza que se quiera etiquetar de la imagen completa
            img_display(img, 'Imagen número  ' + str(idx + 1) + ' de ' + str(len(new_name_list)))
            YN_crop = YNC_prompt('¿Ve una maleza tipo wild lettuce en la imagen?')
            ver_crop = YN_wl.exec_()
            new_frame_crop = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
            new_img = img
            black_img = img
            
            #Mientras que se decida que sí, se extrae el cuadro donde existe la maleza vista
            while ver_crop == 1:
                #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
                cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = black_img)
                img_display(cut_frame, 'Cuadro cortado')
                cut_frame =  np.reshape(  cv2.resize(cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
                ver_good_crop = YN_good_crop.exec_()
                
                #Hasta que no se esté conforme con la selección se pide repetir
                while ver_good_crop == 0:
                    
                    #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
                    cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = black_img)
                    cut_frame =  np.reshape(  cv2.resize(cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                    YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
                    ver_good_crop = YN_good_crop.exec_()
                
                #Si el recorte de la imagen es aceptado, se crea la imagen que contiene la maleza wild lettuce
                if ver_good_crop == 1: 
                    new_frame_crop = np.concatenate( (new_frame_crop, np.reshape (cut_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                    new_img = display_img
                    black_img = cut_img
                else: break
                
                #Se despliega la nueva imagen cortada
                img_display(new_img, 'Imagen procesada')
                
                #Se vuelve a preguntar si existe una maleza que se deba recortar a mano en la imagen global
                YN_crop = YNC_prompt('¿Ve otra maleza tipo wild lettuce en la imagen?')
                ver_crop = YN_crop.exec_()
                img = new_img
                
                #Si no se etiqueta una nueva imagen se pide el nombre de los objetos etiquetados y se guardan en una carpeta con el nombre que corresponda
                if not ver_crop and new_frame_crop.shape[0] > 1:
                    class_name_window = class_name()
                    class_name_window.exec_()
                    crop_name = class_name_window.class_name
                    #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
                    
                    for x in range(1, new_frame_wl.shape[0] ):  
                        crop_destiny_folder = destinies_folder + '/' + crop_name + '/'
                        os.makedirs(crop_destiny_folder, exist_ok = True)
                        
                        crop_name_list = os.listdir(crop_destiny_folder)
                        crop_name_count = len ([crop_destiny_folder + '/' + s for s in crop_name_list if  s.endswith('jpg')])
                        cv2.imwrite(crop_destiny_folder + str(crop_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_crop[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
                
                plt.close('all')
        #Si no se eligen las cosechas, se reinician las imágenes de recorte        
        else:
            new_img = img
            black_img = img
            
        ############################################
        #Se empieza con la maleza tipo wild lettuce#
        ############################################
        
        #Se pregunta si existe alguna maleza que se quiera etiquetar de la imagen completa
        img_display(img, 'Imagen número  ' + str(idx + 1) + ' de ' + str(len(new_name_list)))
        YN_wl = YNC_prompt('¿Ve una maleza tipo wild lettuce en la imagen?')
        ver_wl = YN_wl.exec_()
        new_frame_wl = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        
        
        #Mientras que se decida que sí, se extrae el cuadro donde existe la maleza vista
        while ver_wl == 1:
            
            multi_frame_cut = np.zeros(0, dtype = np.uint8)
            #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
            cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = old_img)
            img_display(cut_frame, 'Cuadro cortado')
            YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
            ver_good_crop = YN_good_crop.exec_()
            
            #Hasta que no se esté conforme con la selección se pide repetir
            while ver_good_crop == 0:
                
                #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
                cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = old_img)
                img_display(cut_frame, 'Cuadro cortado')
                YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
                ver_good_crop = YN_good_crop.exec_()
            
            #Si el recorte de la imagen es aceptado, se crea la imagen que contiene la maleza wild lettuce
            if ver_good_crop == 1: 
                #Se completa la imagen para que cumpla con el tamaño deseado de región (en caso que el corte sea menor al tamaño de región esperado) PARA LAS FILAS
                cut_frame_shape = cut_frame.shape
                
                #Si la imagen tiene menos filas de las pedidas para la región se padea y luego se decide que hacer con las columnas
                if not black_fill:
                    new_cut_frame = cut_frame
                else:
                    if cut_frame_shape[0] < region_size[0]:
                        cut_frame = cv2.copyMakeBorder(cut_frame, int( np.round((region_size[0]-cut_frame_shape[0])/2) ),\
                        int( np.round((region_size[0]-cut_frame_shape[0])/2) ), 0, 0, cv2.BORDER_CONSTANT)
                        
                        #Si además es menor en columnas se padea también en ese eje
                        if cut_frame_shape[1] < region_size[1]: new_cut_frame = cv2.copyMakeBorder(cut_frame, 0, 0,\
                            int( np.round((region_size[1]-cut_frame_shape[1])/2) ), int( np.round((region_size[1]-cut_frame_shape[1])/2) ), cv2.BORDER_CONSTANT)
                        
                        #Si supera el umbral de región extra se crean dos imágenes a lo largo de las columnas
                        elif cut_frame_shape[1] > region_alpha*region_size[1]:
                            for i in range(2):
                                new_cut_frame = cut_frame[:,i*(cut_frame_shape[1] - region_size[1]):cut_frame_shape[1] - (1-i)*(cut_frame_shape[1] - region_size[1]),:]
                                if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                                else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                                
                        #Si solo es mayor debajo del umbral, se corta la imagen
                        else: new_cut_frame = cut_frame[:, int( (cut_frame_shape[1]-region_size[1]) / 2 ): int( cut_frame_shape[1]- (cut_frame_shape[1]-region_size[1]) / 2 ), :]
                    
                    #Si la cantidad de filas de la imagen es mayor que la pedida para la región pero no supera el umbral simplemente se corta y luego se decide que hacer con la cantidad de columnas
                    elif (cut_frame_shape[0] >= region_size[0]) and (cut_frame_shape[0] < region_size[0] * region_alpha):
                        cut_frame = cut_frame[ int( (cut_frame_shape[0]-region_size[0])/2 ) :  int( cut_frame_shape[0] +  (cut_frame_shape[0]-region_size[0])/2 ), :, :]
                        
                        #Si además es menor en columnas se padea también en ese eje
                        if cut_frame_shape[1] < region_size[1]: new_cut_frame = cv2.copyMakeBorder(cut_frame, 0, 0,\
                            int( np.round((region_size[1]-cut_frame_shape[1])/2) ), int( np.round((region_size[1]-cut_frame_shape[1])/2) ), cv2.BORDER_CONSTANT)
                        
                        #Si supera el umbral de región extra se crean dos imágenes a lo largo de las columnas
                        elif cut_frame_shape[1] > region_alpha*region_size[1]:
                            for i in range(2):
                                new_cut_frame = cut_frame[:,i*(cut_frame_shape[1] - region_size[1]):cut_frame_shape[1] - (1-i)*(cut_frame_shape[1] - region_size[1]),:]
                                if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                                else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                                
                        #Si solo es mayor debajo del umbral, se corta la imagen
                        else: new_cut_frame = cut_frame[:, int( (cut_frame_shape[1]-region_size[1]) / 2 ): int( cut_frame_shape[1]- (cut_frame_shape[1]-region_size[1]) / 2 ), :]
                
                    #Si la cantidad de filas supera el umbral de regiones, se decide dependiendo de cada caso que hacer
                    else:
                        for j in range(2):
                
                            #Si además es menor en columnas se padea también en ese eje
                            if cut_frame_shape[1] < region_size[1]:
                                new_cut_frame = cv2.copyMakeBorder(cut_frame[\
                                    j*(cut_frame_shape[0] - region_size[0]):cut_frame_shape[0] - (1-j)*(cut_frame_shape[0] - region_size[0]),:,:]\
                                        , 0, 0,int( np.round((region_size[1]-cut_frame_shape[1])/2) ), int( np.round((region_size[1]-cut_frame_shape[1])/2) ), cv2.BORDER_CONSTANT)
                                
                                if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                                else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                            
                            #Si supera el umbral de región extra se crean dos imágenes a lo largo de las columnas
                            elif cut_frame_shape[1] > region_alpha*region_size[1]:
                                for i in range(2):
                                    new_cut_frame = cut_frame[j*(cut_frame_shape[0] - region_size[0]):cut_frame_shape[0] - (1-j)*(cut_frame_shape[0] - region_size[0]),i*(cut_frame_shape[1] - region_size[1]):cut_frame_shape[1] - (1-i)*(cut_frame_shape[1] - region_size[1]),:]
                                    if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                                    else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                                    
                            #Si solo es mayor debajo del umbral, se corta la imagen
                            else: 
                                new_cut_frame = cut_frame[j*(cut_frame_shape[0] - region_size[0]):cut_frame_shape[0] - (1-j)*(cut_frame_shape[0] - region_size[0]), int( (cut_frame_shape[1]-region_size[1]) / 2 ): int( cut_frame_shape[1]- (cut_frame_shape[1]-region_size[1]) / 2 ), :]
                                if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                                else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                        
                #Si solo se tiene un cuadro por recorte se concatena
                if multi_frame_cut.shape[0] == 0:
                    cut_frame =  np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                    new_frame_wl = np.concatenate( (new_frame_wl, np.reshape (cut_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                
                #Si no, se guardan todas
                else:
                    for i in range(multi_frame_cut.shape[0]):
                        cut_frame =  np.reshape(  cv2.resize(multi_frame_cut[i], (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                        new_frame_wl = np.concatenate( (new_frame_wl, np.reshape (cut_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                
                new_img = display_img
                black_img = cut_img if black_fill else old_img
                

            else: break
               
            #Se despliega la nueva imagen cortada
            img_display(new_img, 'Imagen procesada')
            
            #Se vuelve a preguntar si existe una maleza que se deba recortar a mano en la imagen global
            YN_wl = YNC_prompt('¿Ve otra maleza tipo wild lettuce en la imagen?')
            ver_wl = YN_wl.exec_()
            img = new_img
            
            #Si no se etiqueta una nueva imagen se pide el nombre de los objetos etiquetados y se guardan en una carpeta con el nombre que corresponda
            if not ver_wl and new_frame_wl.shape[0] > 1:
                class_name_window = class_name()
                class_name_window.exec_()
                wl_name = class_name_window.class_name
                #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
                
                for x in range(1, new_frame_wl.shape[0] ):  
                    wl_destiny_folder = destinies_folder + '/' + wl_name + '/'
                    os.makedirs(wl_destiny_folder, exist_ok = True)
                    
                    wl_name_list = os.listdir(wl_destiny_folder)
                    wl_name_count = len ([wl_destiny_folder + '/' + s for s in wl_name_list if  s.endswith('jpg')])
                    cv2.imwrite(wl_destiny_folder + str(wl_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_wl[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
            
            plt.close('all')
            
        ##############
        #Ahora trébol#
        ##############
        
        #Se pregunta si existe alguna maleza que se quiera etiquetar de la imagen completa
        img_display(img, 'Imagen número  ' + str(idx + 1) + ' de ' + str(len(new_name_list)))
        YN_dw = YNC_prompt('¿Ve una maleza tipo trébol en la imagen?')
        ver_dw = YN_dw.exec_()
        new_frame_dw = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        
        #Mientras que se decida que sí, se extrae el cuadro donde existe la maleza vista
        while ver_dw == 1:
            #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
            cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = old_img)
            img_display(cut_frame, 'Cuadro cortado')
            YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
            ver_good_crop = YN_good_crop.exec_()
            
            #Hasta que no se esté conforme con la selección se pide repetir
            while ver_good_crop == 0:
                
                #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
                cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = old_img)
                cut_frame =  np.reshape(  cv2.resize(cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                img_display(cut_frame, 'Cuadro cortado')
                YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
                ver_good_crop = YN_good_crop.exec_()
            
            #Si el recorte de la imagen es aceptado, se crea la imagen que contiene la maleza wild lettuce
            if ver_good_crop == 1: 
                #Se completa la imagen para que cumpla con el tamaño deseado de región (en caso que el corte sea menor al tamaño de región esperado) PARA LAS FILAS
                cut_frame_shape = cut_frame.shape
                multi_frame_cut = np.zeros(0)
                
                #Si la imagen tiene menos filas de las pedidas para la región se padea y luego se decide que hacer con las columnas
                if cut_frame_shape[0] < region_size[0]:
                    #Se decide si se quiere rellenar o no las regiones
                    if not black_fill: 
                        cut_frame = cut_frame
                    else:
                        cut_frame = cv2.copyMakeBorder(cut_frame, int( np.round((region_size[0]-cut_frame_shape[0])/2) ),\
                        int( np.round((region_size[0]-cut_frame_shape[0])/2) ), 0, 0, cv2.BORDER_CONSTANT)
                    
                    #Si además es menor en columnas se padea también en ese eje
                    if cut_frame_shape[1] < region_size[1]: 
                        if not black_fill: 
                            new_cut_frame = cut_frame
                        else:
                            new_cut_frame = cv2.copyMakeBorder(cut_frame, 0, 0,\
                            int( np.round((region_size[1]-cut_frame_shape[1])/2) ), int( np.round((region_size[1]-cut_frame_shape[1])/2) ), cv2.BORDER_CONSTANT)
                    
                    #Si supera el umbral de región extra se crean dos imágenes a lo largo de las columnas
                    elif cut_frame_shape[1] > region_alpha*region_size[1]:
                        for i in range(2):
                            new_cut_frame = cut_frame[:,i*(cut_frame_shape[1] - region_size[1]):cut_frame_shape[1] - (1-i)*(cut_frame_shape[1] - region_size[1]),:]
                            if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                            else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                            
                    #Si solo es mayor debajo del umbral, se corta la imagen
                    else: new_cut_frame = cut_frame[:, int( (cut_frame_shape[1]-region_size[1]) / 2 ): int( cut_frame_shape[1]- (cut_frame_shape[1]-region_size[1]) / 2 ), :]
                    
                #Si la cantidad de filas de la imagen es mayor que la pedida para la región pero no supera el umbral simplemente se corta y luego se decide que hacer con la cantidad de columnas
                elif (cut_frame_shape[0] >= region_size[0]) and (cut_frame_shape[0] < region_size[0] * region_alpha):
                    cut_frame = cut_frame[ int( (cut_frame_shape[0]-region_size[0])/2 ) :  int( cut_frame_shape[0] - (cut_frame_shape[0]-region_size[0])/2 ), :, :]
                    
                    #Si además es menor en columnas se padea también en ese eje
                    if cut_frame_shape[1] < region_size[1]: 
                        if not black_fill: 
                            new_cut_frame = cut_frame
                        else:
                            new_cut_frame = cv2.copyMakeBorder(cut_frame, 0, 0,\
                            int( np.round((region_size[1]-cut_frame_shape[1])/2) ), int( np.round((region_size[1]-cut_frame_shape[1])/2) ), cv2.BORDER_CONSTANT)
                    
                    #Si supera el umbral de región extra se crean dos imágenes a lo largo de las columnas
                    elif cut_frame_shape[1] > region_alpha*region_size[1]:
                        for i in range(2):
                            new_cut_frame = cut_frame[:,i*(cut_frame_shape[1] - region_size[1]):cut_frame_shape[1] - (1-i)*(cut_frame_shape[1] - region_size[1]),:]
                            if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                            else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                            
                    #Si solo es mayor debajo del umbral, se corta la imagen
                    else: new_cut_frame = cut_frame[:, int( (cut_frame_shape[1]-region_size[1]) / 2 ): int( cut_frame_shape[1]- (cut_frame_shape[1]-region_size[1]) / 2 ), :]
            
                #Si la cantidad de filas supera el umbral de regiones, se decide dependiendo de cada caso que hacer
                else:
                    for j in range(2):
            
                        #Si además es menor en columnas se padea también en ese eje
                        if cut_frame_shape[1] < region_size[1]:
                            if not black_fill: 
                                new_cut_frame = cut_frame
                            else:
                                new_cut_frame = cv2.copyMakeBorder(cut_frame[\
                                    j*(cut_frame_shape[0] - region_size[0]):cut_frame_shape[0] - (1-j)*(cut_frame_shape[0] - region_size[0]),:,:]\
                                        , 0, 0,int( np.round((region_size[1]-cut_frame_shape[1])/2) ), int( np.round((region_size[1]-cut_frame_shape[1])/2) ), cv2.BORDER_CONSTANT)
                            
                            if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                            else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                        
                        #Si supera el umbral de región extra se crean dos imágenes a lo largo de las columnas
                        elif cut_frame_shape[1] > region_alpha*region_size[1]:
                            for i in range(2):
                                new_cut_frame = cut_frame[j*(cut_frame_shape[0] - region_size[0]):cut_frame_shape[0] - (1-j)*(cut_frame_shape[0] - region_size[0]),i*(cut_frame_shape[1] - region_size[1]):cut_frame_shape[1] - (1-i)*(cut_frame_shape[1] - region_size[1]),:]
                                if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                                else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                                
                        #Si solo es mayor debajo del umbral, se corta la imagen
                        else: 
                            new_cut_frame = cut_frame[j*(cut_frame_shape[0] - region_size[0]):cut_frame_shape[0] - (1-j)*(cut_frame_shape[0] - region_size[0]), int( (cut_frame_shape[1]-region_size[1]) / 2 ): int( cut_frame_shape[1]- (cut_frame_shape[1]-region_size[1]) / 2 ), :]
                            if multi_frame_cut.shape[0] == 0: multi_frame_cut = np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                            else: multi_frame_cut = np.concatenate( (multi_frame_cut, np.reshape (cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                        
                #Si solo se tiene un cuadro por recorte se concatena
                if multi_frame_cut.shape[0] == 0:
                    cut_frame =  np.reshape(  cv2.resize(new_cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                    new_frame_dw = np.concatenate( (new_frame_dw, np.reshape (cut_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                
                #Si no, se guardan todas
                else:
                    for i in range(multi_frame_cut.shape[0]):
                        cut_frame =  np.reshape(  cv2.resize(multi_frame_cut[i], (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                        new_frame_dw = np.concatenate( (new_frame_dw, np.reshape (cut_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                
                new_img = display_img
                black_img = cut_img if black_fill else old_img
            else: break
               
            #Se despliega la nueva imagen cortada
            img_display(new_img, 'Imagen procesada')
            
            #Se vuelve a preguntar si existe una maleza que se deba recortar a mano en la imagen global
            YN_dw = YNC_prompt('¿Ve otra maleza tipo trébol en la imagen?')
            ver_dw = YN_dw.exec_()
            img = new_img
            
            #Si no se etiqueta una nueva imagen se pide el nombre de los objetos etiquetados y se guardan en una carpeta con el nombre que corresponda
            if not ver_dw and new_frame_dw.shape[0] > 1:
                class_name_window = class_name()
                class_name_window.exec_()
                dw_name = class_name_window.class_name
                #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
                
                for x in range(1, new_frame_dw.shape[0] ):  
                    dw_destiny_folder = destinies_folder + '/' + dw_name + '/'
                    os.makedirs(dw_destiny_folder, exist_ok = True)
                    dw_name_list = os.listdir(dw_destiny_folder)
                    dw_name_count = len ([s for s in dw_name_list if  s.endswith('jpg')])
                    print(dw_name_count)
                    cv2.imwrite(dw_destiny_folder + str(dw_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_dw[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
            
            plt.close('all')
        
        '''
        #TODO: Descomentar el trozo de código arriba y agregar un booleano para elegir la forma de cortar trébol
        #############################################################################
        #Cuadros deslizantes para las zonas exclusivas de pasto (u otro)#
        #############################################################################
        
        new_frame_dw = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        #Se lee la imagen, se despliega un cuadro y se pregunta si existe un objeto de interés dentro del mismo
        frames_r = int ( black_img.shape[0] / frame_size[0] )
        frames_c = int ( black_img.shape[1] / frame_size[1] )
        
        for frame_num_r, frame_r in enumerate(range(0, frames_r)):
            for frame_num_c, frame_c in  enumerate(range(0, frames_c)):
                #Se despliega un cuadro dentro de la imagen
                frame_display = black_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                frame = black_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                frame_orig = np.zeros( (frame.shape[0], frame.shape[1], frame.shape[2]), dtype = np.uint8 )
                frame_orig[:,:,0]= frame[:,:,0]
                frame_orig[:,:,1]= frame[:,:,1]
                frame_orig[:,:,2]= frame[:,:,2]
                
                img_display(frame_display, 'Cuadro: ' + str( frame_r * ( frames_c ) + frame_num_c + 1 ) + ' De: ' + str(frames_r*frames_c))
                if frame_num_c == 0: YN = YNC_prompt('¿Ve una maleza en la imagen? (Primer cuadro de la fila)')
                else: YN = YNC_prompt('¿Ve una maleza en la imagen?')
                
                ver_dw = YN.exec_()
                
                #Si no se ve ninguna maleza en la imagen, se guarda el cuadro como ejemplo negativo. En caso contrario, no se considera guardar
                if ver_dw == 1:
                    
                    dw_frame = cv2.resize(frame, (new_shape[0], new_shape[1]) )
                    new_frame_dw = np.concatenate( (new_frame_dw, np.reshape (dw_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                    black_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :] = 0
                    
                elif ver_dw ==-1 and frame_num_c == 0 : break
        
        #Lo mismo que lo anterior pero para los objetos etiquetados como negativos
        if new_frame_dw.shape[0] > 1:
            
            class_name_window = class_name()
            class_name_window.change_text('Ingrese el nombre de la clase de maleza que acaba de etiquetar')
            class_name_window.exec_()
            dw_name = class_name_window.class_name

            #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
            
            for x in range(1, new_frame_dw.shape[0] ):
                                
                dw_destiny_folder = destinies_folder + '/' + dw_name + '/'
                os.makedirs(dw_destiny_folder, exist_ok = True)
                
                dw_name_list = os.listdir(dw_destiny_folder)
                dw_name_count = len ([dw_destiny_folder + '/' + s for s in dw_name_list if  s.endswith('jpg')])
                
                cv2.imwrite(dw_destiny_folder + str(dw_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_dw[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
                
        plt.close('all')
        
        '''
        #############################################################################
        #Finalmente, cuadros deslizantes para las zonas exclusivas de pasto (u otro)#
        #############################################################################
        
        new_frame_notdw = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        
        #Se padea la imagen de pasto para obtener más imágenes de pasto
        img_shape = black_img.shape
        black_img = cv2.copyMakeBorder(black_img, int( ( np.ceil( img_shape[0]/frame_size[0] ) * frame_size[0] - img_shape[0] ) / 2 ),\
            int( ( np.ceil( img_shape[0]/frame_size[0] ) * frame_size[0] - img_shape[0] ) / 2 ), int( ( np.ceil( img_shape[1]/frame_size[1] ) * frame_size[1] - img_shape[1] ) / 2 )\
                ,int( ( np.ceil( img_shape[1]/frame_size[1] ) * frame_size[1] - img_shape[1] ) / 2 ), cv2.BORDER_CONSTANT)
        
        #Se lee la imagen, se despliega un cuadro y se pregunta si existe un objeto de interés dentro del mismo
        frames_r = int ( black_img.shape[0] / frame_size[0] )
        frames_c = int ( black_img.shape[1] / frame_size[1] )
        
        for frame_num_r, frame_r in enumerate(range(0, frames_r)):
            for frame_num_c, frame_c in  enumerate(range(0, frames_c)):
                #Se despliega un cuadro dentro de la imagen
                frame_display = old_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                frame = old_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                if frame.shape[0] < region_size[0]: frame = old_img[frame_r * frame_size[0] - (region_size[0]-frame.shape[0]) : (frame_r+1) * frame_size[0],\
                    frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                if frame.shape[1] < region_size[1]: frame = old_img[frame_r * frame_size[0]  : (frame_r+1) * frame_size[0],\
                    frame_c * frame_size[1] - (region_size[1]-frame.shape[1]) : (frame_c+1) * frame_size[1], :]
                frame_orig = np.zeros( (frame.shape[0], frame.shape[1], frame.shape[2]), dtype = np.uint8 )
                frame_orig[:,:,0]= frame[:,:,0]
                frame_orig[:,:,1]= frame[:,:,1]
                frame_orig[:,:,2]= frame[:,:,2]
                
                img_display(frame, 'Cuadro: ' + str( frame_r * ( frames_c ) + frame_num_c + 1 ) + ' De: ' + str(frames_r*frames_c))
                if frame_num_c == 0: YN = YNC_prompt('¿Ve una maleza en la imagen? (Primer cuadro de la fila)')
                else: YN = YNC_prompt('¿Ve una maleza en la imagen?')
                
                ver_dw = YN.exec_()
                
                #Si no se ve ninguna maleza en la imagen, se guarda el cuadro como ejemplo negativo. En caso contrario, no se considera guardar
                if ver_dw == 0:
                    
                    notdw_frame = cv2.resize(frame, (new_shape[0], new_shape[1]) )
                    new_frame_notdw = np.concatenate( (new_frame_notdw, np.reshape (notdw_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                    
                elif ver_dw ==-1 and frame_num_c == 0 : break
        
        #Lo mismo que lo anterior pero para los objetos etiquetados como negativos
        if new_frame_notdw.shape[0] > 1:
            
            class_name_window = class_name()
            class_name_window.change_text('Ingrese el nombre del ejemplo negativo que acaba de etiquetar')
            class_name_window.exec_()
            notdw_name = class_name_window.class_name

            #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
            
            for x in range(1, new_frame_notdw.shape[0] ):
                                
                notdw_destiny_folder = destinies_folder + '/' + notdw_name + '/'
                os.makedirs(notdw_destiny_folder, exist_ok = True)
                
                notdw_name_list = os.listdir(notdw_destiny_folder)
                notdw_name_count = len ([notdw_destiny_folder + '/' + s for s in notdw_name_list if  s.endswith('jpg')])
                
                cv2.imwrite(notdw_destiny_folder + str(notdw_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_notdw[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
                
    plt.close('all')
    
#Función que permite etiquetar objetos de múltiples clases dentro de una imagen, partiendo con objetos que se seleccionen de la imagen completa y luego etiquetar dentro de sub-cuadros
def multiclass_frame_tagging(origin_folder, destinies_folder, frame_size = (256, 256, 3), new_shape = (64, 64, 3), ovrwrite = False):
    
    
    #Se lee la carpeta de origen de los datos y se adjuntan las imágenes que contengan un formato permitido
    name_list = os.listdir(origin_folder)
    new_name_list = [origin_folder + '/' + s for s in name_list if ( s.endswith('jpg') or s.endswith('png') or s.endswith('jpeg') )]
    
    #Si se elige sobreescribir (y se verifica que la carpeta de destino existe) se elimina dicho directorio
    if ovrwrite and os.path.isdir(destinies_folder):
        if (input('SEGURO QUE QUIERE BORRAR?!?!??!?! Y/N') == 'Y'): rmtree(destinies_folder)
    
    #Se leen las imágenes y se etiqueta dentro de ellas
    for i, im_path in enumerate(new_name_list): 
        
        img = cv2.cvtColor( cv2.imread(im_path), cv2.COLOR_BGR2RGB ) 
        
        ############################################
        #Se empieza con la maleza tipo wild lettuce#
        ############################################
        
        #Se pregunta si existe alguna maleza que se quiera etiquetar de la imagen completa
        img_display(img, 'Imagen número  ' + str(i + 1) + ' de ' + str(len(new_name_list)))
        YN_wl = YNC_prompt('¿Ve una maleza tipo wild lettuce en la imagen?')
        ver_wl = YN_wl.exec_()
        new_frame_wl = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        new_frame_notwl = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        
        new_img = img
        black_img = img
        
        #Mientras que se decida que sí, se extrae el cuadro donde existe la maleza vista
        while ver_wl == 1:
            
            #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
            cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = black_img)
            img_display(cut_frame, 'Cuadro cortado')
            #cut_frame =  np.reshape(  cv2.resize(cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
            YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
            ver_good_crop = YN_good_crop.exec_()
            
            #Hasta que no se esté conforme con la selección se pide repetir
            while ver_good_crop == 0:
                
                #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
                cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = black_img)
                #cut_frame =  np.reshape(  cv2.resize(cut_frame, (new_shape[0], new_shape[1])), (1, new_shape[0], new_shape[1], new_shape[2]) )
                YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
                ver_good_crop = YN_good_crop.exec_()
            
            #Si el recorte de la imagen es aceptado, se crea la imagen que contiene la maleza wild lettuce
            if ver_good_crop == 1: 
                
                new_img = display_img
                black_img = cut_img
                
                #Se escala la imagen para que sea divisible en el tamaño de los cuadros a revisar
                reshape_cut_frame = cut_frame#cv2.resize( cut_frame, (cut_frame.shape[0] , cut_frame.shape[1] ) )
                frames_r = int ( reshape_cut_frame.shape[0] / frame_size[0] )
                frames_c = int ( reshape_cut_frame.shape[1] / frame_size[1] )
                
                #Se recorren todos los cuadros para etiquetar los espacios donde se encuentre maleza
                for frame_r in range(0, frames_r+1):
                    
                    for frame_c in range(0, frames_c+1):
                        
                        wl_frame = reshape_cut_frame[ np.min( [ frame_r*frame_size[0], reshape_cut_frame.shape[0] - frame_size[0]]) : np.min( [ (frame_r+1)*frame_size[0], reshape_cut_frame.shape[0] ] ),\
                            np.min( [ frame_c*frame_size[1], reshape_cut_frame.shape[1] - frame_size[1]]) : np.min( [ (frame_c+1)*frame_size[1], reshape_cut_frame.shape[1] ] ), : ]
                        img_display(wl_frame, '¿Existe una maleza wild lettuce en el cuadro?')
                        YN_ver_frame = YNC_prompt('¿Existe una maleza wild lettuce en el cuadro?')
                        ver_frame = YN_ver_frame.exec_()
                        
                        #Si se confirma la existencia de maleza en el cuadro correspondiente, se guarda como ejemplo positivo
                        if ver_frame == 1:
                            
                            wl_frame = cv2.resize(wl_frame, (new_shape[0], new_shape[1]) )
                            new_frame_wl = np.concatenate( (new_frame_wl, np.reshape (wl_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                            
                        #En caso contrario, se guarda como no wild lettuce
                        elif ver_frame == 0:
                            
                            notwl_frame = cv2.resize(wl_frame, (new_shape[0], new_shape[1]) )
                            new_frame_notwl = np.concatenate( (new_frame_notwl, np.reshape (notwl_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                                    
            else: break
               
            #Se despliega la nueva imagen cortada
            img_display(new_img, 'Imagen procesada')
            
            #Se vuelve a preguntar si existe una maleza que se deba recortar a mano en la imagen global
            YN_wl = YNC_prompt('¿Ve otra maleza tipo wild lettuce en la imagen?')
            ver_wl = YN_wl.exec_()
            img = new_img
            
            #Si no se etiqueta una nueva imagen se pide el nombre de los objetos etiquetados como maleza y se guardan en una carpeta con el nombre que corresponda
            if new_frame_wl.shape[0] > 1 and ver_wl == 0:
                
                class_name_window = class_name()
                class_name_window.exec_()
                wl_name = class_name_window.class_name
                
                for x in range(1, new_frame_wl.shape[0] ):
                                    
                    wl_destiny_folder = destinies_folder + '/' + wl_name + '/'
                    os.makedirs(wl_destiny_folder, exist_ok = True)
                    
                    wl_name_list = os.listdir(wl_destiny_folder)
                    wl_name_count = len ([wl_destiny_folder + '/' + s for s in wl_name_list if  s.endswith('jpg')])
                    
                    cv2.imwrite(wl_destiny_folder + str(wl_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_wl[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
                    
            #Lo mismo que lo anterior pero para los objetos etiquetados como negativos
            if new_frame_notwl.shape[0] > 1 and ver_wl == 0:
                
                class_name_window = class_name()
                class_name_window.change_text('Ingrese el nombre del ejemplo negativo que acaba de etiquetar')
                class_name_window.exec_()
                notwl_name = class_name_window.class_name

                #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
                
                for x in range(1, new_frame_notwl.shape[0] ):
                                    
                    notwl_destiny_folder = destinies_folder + '/' + notwl_name + '/'
                    os.makedirs(notwl_destiny_folder, exist_ok = True)
                    
                    notwl_name_list = os.listdir(notwl_destiny_folder)
                    notwl_name_count = len ([notwl_destiny_folder + '/' + s for s in notwl_name_list if  s.endswith('jpg')])
                    
                    cv2.imwrite(notwl_destiny_folder + str(notwl_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_notwl[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
            
        plt.close('all')
            
        ##################################################################################################################################################
        #Se pasa ahora a la maleza tipo trébol, método elegir región de tréboles y dividir en cuadros de 256x256 para verificar la existencia de trebóles#
        ##################################################################################################################################################
        
        img_display(img, 'Imagen número  ' + str(i + 1) + ' de ' + str(len(new_name_list)))
        YN_dw= YNC_prompt('¿Ve una maleza tipo trébol en la imagen?')
        ver_dw = YN_dw.exec_()
        new_img = img
        new_frame_dw = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        new_frame_notdw = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        
        #Mientras que se decida que sí, se extrae el cuadro donde existe la maleza vista
        while ver_dw == 1:
            
            #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
            cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = black_img)
            img_display(cut_frame, 'Cuadro cortado')
            YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
            ver_good_cropdw = YN_good_crop.exec_()
            
            #Hasta que no se esté conforme con la elección del cuadro se pide repetir
            while ver_good_cropdw == 0:
                
                #Se corta la imagen en las coordenadas escogidas y se guarda en una matriz para su posterior escritura en la carpeta respectiva
                cut_frame, display_img, cut_img, _, _ = corner_selector(new_img, old_img = black_img)
                YN_good_crop = YNC_prompt('¿Está conforme con la imagen elegida?')
                ver_good_crop = YN_good_crop.exec_()
            
            #Si la selección es aceptada, se pasa a recortar la imagen en sub-cuadros y revisar en cuales se encuentran tréboles
            if ver_good_cropdw == 1: 
                
                new_img = display_img
                black_img = cut_img
                
                
                #Se escala la imagen para que sea divisible en el tamaño de los cuadros a revisar
                reshape_cut_frame = cut_frame#cv2.resize( cut_frame, (cut_frame.shape[0] , cut_frame.shape[1] ) )
                frames_r = int ( reshape_cut_frame.shape[0] / frame_size[0] )
                frames_c = int ( reshape_cut_frame.shape[1] / frame_size[1] )
                
                #Se recorren todos los cuadros para etiquetar los espacios donde se encuentre maleza
                for frame_r in range(0, frames_r):
                    
                    for frame_c in range(0, frames_c):
                        
                        dw_frame = reshape_cut_frame[ frame_r*frame_size[0]:(frame_r+1)*frame_size[0], frame_c*frame_size[1]:(frame_c+1)*frame_size[1], : ]
                        img_display(dw_frame, '¿Existe una maleza trébol en el cuadro?')
                        YN_ver_frame = YNC_prompt('¿Existe una maleza trébol en el cuadro?')
                        ver_frame = YN_ver_frame.exec_()
                        
                        #Si se confirma la existencia de maleza en el cuadro correspondiente, se guarda como ejemplo positivo
                        if ver_frame == 1:
                            
                            dw_frame = cv2.resize(dw_frame, (new_shape[0], new_shape[1]) )
                            new_frame_dw = np.concatenate( (new_frame_dw, np.reshape (dw_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                            
                        #En caso contrario, se guarda como no trébol
                        elif ver_frame == 0:
                            
                            notdw_frame = cv2.resize(dw_frame, (new_shape[0], new_shape[1]) )
                            new_frame_notdw = np.concatenate( (new_frame_notdw, np.reshape (notdw_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
            #Se vuelve a preguntar si existe una maleza que se deba recortar a mano en la imagen global
            img_display(new_img, 'Imagen procesada')
            YN_dw = YNC_prompt('¿Ve otra maleza tipo trébol en la imagen?')
            ver_dw = YN_dw.exec_()
            img = new_img
                                        
            #Si no se etiqueta una nueva imagen se pide el nombre de los objetos etiquetados como maleza y se guardan en una carpeta con el nombre que corresponda
            if new_frame_dw.shape[0] > 1 and ver_dw == 0:
                
                class_name_window = class_name()
                class_name_window.exec_()
                dw_name = class_name_window.class_name
                
                for x in range(1, new_frame_dw.shape[0] ):
                                    
                    dw_destiny_folder = destinies_folder + '/' + dw_name + '/'
                    os.makedirs(dw_destiny_folder, exist_ok = True)
                    
                    dw_name_list = os.listdir(dw_destiny_folder)
                    dw_name_count = len ([dw_destiny_folder + '/' + s for s in dw_name_list if  s.endswith('jpg')])
                    
                    cv2.imwrite(dw_destiny_folder + str(dw_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_dw[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
                    
            #Lo mismo que lo anterior pero para los objetos etiquetados como negativos
            if new_frame_notdw.shape[0] > 1 and ver_dw == 0:
                
                class_name_window = class_name()
                class_name_window.change_text('Ingrese el nombre del ejemplo negativo que acaba de etiquetar')
                class_name_window.exec_()
                notdw_name = class_name_window.class_name

                #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
                
                for x in range(1, new_frame_notdw.shape[0] ):
                                    
                    notdw_destiny_folder = destinies_folder + '/' + notdw_name + '/'
                    os.makedirs(notdw_destiny_folder, exist_ok = True)
                    
                    notdw_name_list = os.listdir(notdw_destiny_folder)
                    notdw_name_count = len ([notdw_destiny_folder + '/' + s for s in notdw_name_list if  s.endswith('jpg')])
                    
                    cv2.imwrite(notdw_destiny_folder + str(notdw_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_notdw[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
            
        plt.close('all')
                      
        #################################################################################################################
        #Se pasa ahora a barrer la imagen por sliding windows, para determinar los espacios que correspondan a no maleza#
        #################################################################################################################
        new_frame_notdw = np.zeros( ( 1, new_shape[0], new_shape[1], new_shape[2] ) , dtype = np.uint8)
        #Se lee la imagen, se despliega un cuadro y se pregunta si existe un objeto de interés dentro del mismo
        frames_r = int ( black_img.shape[0] / frame_size[0] )
        frames_c = int ( black_img.shape[1] / frame_size[1] )
        
        for frame_num_r, frame_r in enumerate(range(0, frames_r)):
            
            for frame_num_c, frame_c in  enumerate(range(0, frames_c)):
                
                #Se despliega un cuadro dentro de la imagen
                frame_display = black_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                frame = black_img[frame_r * frame_size[0] : (frame_r+1) * frame_size[0], frame_c * frame_size[1] : (frame_c+1) * frame_size[1], :]
                frame_orig = np.zeros( (frame.shape[0], frame.shape[1], frame.shape[2]), dtype = np.uint8 )
                frame_orig[:,:,0]= frame[:,:,0]
                frame_orig[:,:,1]= frame[:,:,1]
                frame_orig[:,:,2]= frame[:,:,2]
                
                img_display(frame_display, 'Cuadro: ' + str( frame_r * ( frames_c ) + frame_num_c + 1 )  + ' De: ' + str(frames_r*frames_c))
                if frame_num_c == 0: YN = YNC_prompt('¿Ve una maleza en la imagen? (Primer cuadro de la fila)')
                else: YN = YNC_prompt('¿Ve una maleza en la imagen?')
                
                ver_dw = YN.exec_()
                
                #Si no se ve ninguna maleza en la imagen, se guarda el cuadro como ejemplo negativo. En caso contrario, no se considera guardar
                if ver_dw == 0:
                    
                    notdw_frame = cv2.resize(frame, (new_shape[0], new_shape[1]) )
                    new_frame_notdw = np.concatenate( (new_frame_notdw, np.reshape (notdw_frame, (1, new_shape[0], new_shape[1], new_shape[2]))), axis = 0 )
                    
                elif ver_dw ==-1 and frame_num_c == 0 : break
        
        #Lo mismo que lo anterior pero para los objetos etiquetados como negativos
        if new_frame_notdw.shape[0] > 1:
            
            class_name_window = class_name()
            class_name_window.change_text('Ingrese el nombre del ejemplo negativo que acaba de etiquetar')
            class_name_window.exec_()
            notdw_name = class_name_window.class_name

            #wl_name = input('Ingrese el nombre de carpeta con el que desea guardar las imágenes anteriormente etiquetadas: ')
            
            for x in range(1, new_frame_notdw.shape[0] ):
                                
                notdw_destiny_folder = destinies_folder + '/' + notdw_name + '/'
                os.makedirs(notdw_destiny_folder, exist_ok = True)
                
                notdw_name_list = os.listdir(notdw_destiny_folder)
                notdw_name_count = len ([notdw_destiny_folder + '/' + s for s in notdw_name_list if  s.endswith('jpg')])
                
                cv2.imwrite(notdw_destiny_folder + str(notdw_name_count + 1) + '.jpg', cv2.cvtColor(np.reshape( new_frame_notdw[x,:,:,:], new_shape ), cv2.COLOR_RGB2BGR) )
    plt.close('all')
#Función que realiza aumento de datos basado (de momento) en métodos geométricos
def data_augmentation(origin_folder_list, aug_method = 'rot_flip', angle_num = 3):
    
    #Se levanta una advertencia por el número de ángulo
    if angle_num < 1 or angle_num > 5: warnings.warn('El rango de ángulos es entre 1 y 5')
    angle_num = int(np.max([np.min([angle_num, 5]), 1]))
    print(angle_num)
    
    #Si la entrada de origen corresponde a una lista se recorren todas las carpetas una por una para aumentar las imágenes
    if isinstance(origin_folder_list, list):
        for folder in origin_folder_list:
            if os.path.isdir(folder):
                #Si el nombre de la carpeta dirige a un directorio se proceden a adjuntar todos los formatos de imagen válidos
                img_list = [img_name for img_name in os.listdir(folder) if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')]
                
    #En caso contrario, se recorre la única carpeta
    elif isinstance(origin_folder_list, str) and os.path.isdir(origin_folder_list):
        #Si el nombre de la carpeta dirige a un directorio se proceden a adjuntar todos los formatos de imagen válidos
        img_list = [origin_folder_list + '/' + img_name for img_name in os.listdir(origin_folder_list) if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')]
        angle_list = np.linspace(0, 360, 2 + angle_num).astype(int)
        
        #Se lee cada imagen y se aumenta conforme a la técnica especificada
        for img_name in img_list:
            img = cv2.cvtColor(cv2.imread( img_name ),cv2.COLOR_BGR2HSV)
            #Se aumenta el tamaño de la imagen para que al girar las imágenes nunca se pierda información
            [rows, cols] = [img.shape[0], img.shape[1]]
            [hip, top, bottom, left, right] = [int( np.sqrt( rows ** 2 + cols ** 2) ), int ( 0.5 * ( int( np.sqrt( rows ** 2 + cols ** 2) ) - rows ) ),\
                int ( 0.5 * ( int( np.sqrt( rows ** 2 + cols ** 2) ) - rows ) ), int(0.5*( int( np.sqrt( rows ** 2 + cols ** 2) ) - cols ) ),\
                    int(0.5*( int( np.sqrt( rows ** 2 + cols ** 2) ) - cols ) )]
            [new_rows, new_cols] = [rows + top + bottom, cols + left + right]
            
            #Se construye la imagen padeada y se extraen las matrices de Hue, Saturation y Value
            img_hsv_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,0) 
            H = img_hsv_pad[:,:,0]
            S = img_hsv_pad[:,:,1]
            V = img_hsv_pad[:,:,2]
            img_shape = img_hsv_pad.shape
            img_hsv_stack = np.zeros((3 + len(angle_list), img.shape[0] , img.shape[1], 3), dtype = np.uint8)
            
            #Se guardan las imágenes original y las dos flipeadas en los primeros 3 espacios del stack
            flip_Vert = cv2.flip(img,0)
            flip_Hor = cv2.flip(img,1)
            img_hsv_stack[0,:,:,:] = img
            img_hsv_stack[1,:,:,:] = flip_Vert
            img_hsv_stack[2,:,:,:] = flip_Hor
            i = 3

            #Se calculan y guardan las imágenes giradas en el stack de HSV
            for ang in angle_list:
                M = cv2.getRotationMatrix2D(((new_cols-1)/2.0,(new_rows-1)/2.0),ang,1)
                H_flip = cv2.warpAffine(H,M,(new_cols,new_rows))
                S_flip = cv2.warpAffine(S,M,(new_cols,new_rows))
                V_flip = cv2.warpAffine(V,M,(new_cols,new_rows))
                img_hsv_new = np.zeros(img_shape)
                img_hsv_new[:,:,0] = H_flip
                img_hsv_new[:,:,1] = S_flip
                img_hsv_new[:,:,2] = V_flip
                
                img_hsv_stack[i,:,:,:] = cv2.resize(img_hsv_new, ( img.shape[0], img.shape[1] ) )
                i = i + 1

            for img_hsv in img_hsv_stack:
                img_idx = len([origin_folder_list + '/' + img_name for img_name in os.listdir(origin_folder_list) if img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')])
                print(img_idx)
                cv2.imwrite( origin_folder_list + '/' + str(img_idx + 1) + '.jpg', cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) )
                cv2.imshow('', cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR))
                cv2.waitKey(0)
        
#Función que crea ventanas móviles para ser analizadas por algún tipo de segmentador
def slide_window_creator(img_rgb, win_size = (256, 256), new_win_shape = (64,64), overlap_factor = 0.5, data_type = 'CSN', verbose = False):
    sld_step = np.round( [ win_size[0]*(1-overlap_factor),  win_size[1]*(1-overlap_factor)] ).astype(int)
    #Si se escoge hacer pares de cuadros para CSN
    if data_type == 'CSN':
        
        #Se rellena la imagen con ceros para que no sobre espacio de la foto al deslizar la ventana
        img_rgb_shape = img_rgb.shape
        top = int(win_size[0]*.5)#win_size[0] - img_rgb_shape[0] % sld_step[0] if img_rgb_shape[0] % sld_step[0] != 0 else 0
        bottom = top
        left = int(win_size[1]*.5)#win_size[1] - img_rgb_shape[1] % sld_step[1] if img_rgb_shape[1] % sld_step[1] != 0 else 0
        right = left
        img_rgb_new = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT)
            #cv2.resize( img_rgb, ( int( ( img_rgb_shape[1] - win_size[1]) / sld_step[1] )  * sld_step[1] + win_size[1] ,\
            #int( ( img_rgb_shape[0] - win_size[0]) / sld_step[0] )  * sld_step[0] + win_size[0] ) )
        frames_r = np.floor( (img_rgb.shape[0]-win_size[0]) / ( win_size[0] * (1-overlap_factor) ) + 1  ).astype(int)#int( ( img_rgb_new.shape[0] - win_size[0]  ) /  sld_step[0] ) + 1
        frames_c = np.floor( (img_rgb.shape[1]-win_size[1]) / ( win_size[1] * (1-overlap_factor) ) + 1  ).astype(int)#int( ( img_rgb_new.shape[1] - win_size[1]  ) /  sld_step[1] ) + 1
        frame_array_in = np.zeros( (2, (frames_r-1)*(frames_c-1)*2, new_win_shape[0], new_win_shape[1], 3), dtype = np.uint8 )
        frame_coordinates_in = np.zeros( (2, (frames_r-1)*(frames_c-1)*2, 4), dtype = np.int32 )
        new_img = np.zeros(img_rgb_new.shape, dtype = np.uint8)
        frame_array = np.zeros( (1 , 4), dtype = np.int32 )
        frame_array_tot = np.zeros( ( 1, new_win_shape[0], new_win_shape[1], 3), dtype = np.uint8 )
        frame_coord_single = np.zeros( ( 1, 4), dtype = np.int32 )

        #Para cada subdivisión se crean los cuadros interiores
        for frame_r in range(frames_r-1):
            for frame_c in range(frames_c-1):
                #Se calculan las coordenadas para cada par de imágenes central-derecha y central-abajo
                frame_central_coord = [ frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                frame_right_coord = [ frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0], ( frame_c + 1 )*sld_step[1], ( frame_c + 1 )*sld_step[1] + win_size[1] ]
                frame_down_coord = [ ( frame_r + 1) * sld_step[0], ( frame_r + 1 )*sld_step[0] + win_size[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                
                #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen, además del arreglo total de cuadros centrales
                frame_array = np.concatenate( (frame_array, np.reshape(frame_central_coord, ( 1, 4) ) ), axis = 0)
                frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( cv2.resize(img_rgb[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) ), ( 1, new_win_shape[0], new_win_shape[1], 3 ) ) ), axis = 0 )
                
                #frame_array_tot[( frames_c - 1 )*frame_r + frame_c , :, :, :] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                #frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
                
                #OJO con el orden, es primero cuadro-cuadro de la derecha y luego cuadro-cuadro de abajo
                frame_array_in[0,  2*( ( frames_c - 1 )*frame_r + frame_c ), : ,: ,: ] = cv2.resize(img_rgb[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
                frame_array_in[1,  2*(  ( frames_c - 1 )*frame_r + frame_c ) , : ,: ,: ] = cv2.resize(img_rgb[ frame_right_coord[0] : frame_right_coord[1],\
                    frame_right_coord[2] : frame_right_coord[3] , :], (new_win_shape[0], new_win_shape[1]) )
                
                frame_array_in[0, 2*( ( frames_c - 1 )*frame_r + frame_c ) + 1, : ,: ,: ] = cv2.resize(img_rgb[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :] , (new_win_shape[0], new_win_shape[1]) ) 
                frame_array_in[1, 2*( ( frames_c - 1 )*frame_r + frame_c ) + 1, : ,: ,: ] = cv2.resize( img_rgb[ frame_down_coord[0] : frame_down_coord[1],\
                    frame_down_coord[2] : frame_down_coord[3] , :], ( new_win_shape[0], new_win_shape[1] ) )
                
                #Se adjuntan las coordenadas para asociar a predicciones de CSN más adelante
                frame_coordinates_in[0, 2*( ( frames_c - 1)*frame_r + frame_c ), :  ] = frame_central_coord
                frame_coordinates_in[1, 2*( ( frames_c - 1)*frame_r + frame_c ), :  ] = frame_right_coord
                
                frame_coordinates_in[0, 2*( ( frames_c - 1)*frame_r + frame_c ) + 1, : ] = frame_central_coord
                frame_coordinates_in[1, 2*( ( frames_c - 1)*frame_r + frame_c ) + 1, : ] = frame_down_coord   
                
                frame_coord_single = np.concatenate( ( frame_coord_single, np.reshape( frame_central_coord, (1,4) ) ), axis = 0 )#[frames_c*frame_r + frame_c,:] = frame_central_coord
                #print(frame_coord_single)
        
        #Se rellenan los valores restantes de las coordenadas de los cuadros
        for frame_r in range(0, frames_r):
            for frame_c in range(frames_c-1, frames_c):
                
                frame_central_coord = [ frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                #frame_array[ ( frames_c - 1)*frame_r + frame_c,:] = frame_central_coord
                frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( cv2.resize(img_rgb[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) ), ( 1, new_win_shape[0], new_win_shape[1], 3 ) ) ), axis = 0 )
                frame_array = np.concatenate( (frame_array, np.reshape(frame_central_coord, ( 1, 4) ) ), axis = 0)
                #frame_array_tot[( frames_c - 1 )*frame_r + frame_c , :, :, :] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                #frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
                frame_coord_single = np.concatenate( ( frame_coord_single, np.reshape( frame_central_coord, (1,4) ) ), axis = 0 )
                #print(frame_coord_single)
        
        for frame_r in range(frames_r-1, frames_r):
            
            for frame_c in range(0, frames_c):
                
                frame_central_coord = [ frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                frame_array = np.concatenate( (frame_array, np.reshape(frame_central_coord, ( 1, 4) ) ), axis = 0)
                
                frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( cv2.resize(img_rgb[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) ), ( 1, new_win_shape[0], new_win_shape[1], 3 ) ) ), axis = 0 )
                #frame_array_tot[( frames_c - 1 )*frame_r + frame_c , :, :, :] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                #frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
                frame_coord_single = np.concatenate( ( frame_coord_single, np.reshape( frame_central_coord, (1,4) ) ), axis = 0 )
                #print(frame_coord_single)
        
        #Se corta el primer cuadro del arreglo de cuadros centrales para eliminar el 0
        frame_array = frame_array[1:-1, :]
        frame_array = np.concatenate( ( frame_array, np.reshape( frame_array[-1, :] , ( 1, 4 ) ) ), axis = 0 )
        
        frame_array_tot = frame_array_tot[1:-1, :, :, :]
        #frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( frame_array_tot[-1, :, :, :] , ( 1, frame_array_tot[-1, :, :, :].shape[0], frame_array_tot[-1, :, :, :].shape[1], frame_array_tot[-1, :, :, :].shape[2] ) ) ), axis = 0 )
        
        frame_coord_single = frame_coord_single[1:-1, :] 
        #frame_coord_single = np.concatenate( ( frame_coord_single, np.reshape( frame_coord_single[-1, :,] , ( 1, frame_coord_single[-1, :].shape[0] ) ) ), axis = 0 )

        return frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array, frame_array_tot, frame_coord_single
    
    #Si se elige simplemente generar cuadros dentro de la imagen 
    elif data_type == 'CNN': 
        #Se rellena la imagen con ceros para que no sobre espacio de la foto al deslizar la ventana
        img_rgb_shape = img_rgb.shape
        img_rgb_new = 0
        frames_r = np.floor( (img_rgb.shape[0]-win_size[0]) / ( win_size[0] * (1-overlap_factor) ) + 1  ).astype(int)
        frames_c = np.floor( (img_rgb.shape[1]-win_size[1]) / ( win_size[1] * (1-overlap_factor) ) + 1  ).astype(int)
        
        if frames_r<0: frames_r = 1
        if frames_c<0: frames_c = 1
        if len(img_rgb_shape) > 2: frame_array_in = np.zeros( ( (frames_r+1)*(frames_c+1), new_win_shape[0], new_win_shape[1], 3), dtype = np.uint8 )
        else: frame_array_in = np.zeros( ( (frames_r+1)*(frames_c+1), new_win_shape[0], new_win_shape[1]), dtype = np.uint8 )
        frame_coordinates_in = np.zeros( ( (frames_r+1)*(frames_c+1), 4), dtype = np.int32 )
        frame_array = np.zeros( ( (frames_r+1), (frames_c+1), 4), dtype = np.int32 )
        new_size = [new_win_shape[0]/win_size[0], new_win_shape[1]/win_size[1]]
        img_resized = cv2.resize(img_rgb, None, fx = new_size[1], fy = new_size[0])
        sld_step_resized = np.round( [(new_win_shape[0])*(1-overlap_factor),  (new_win_shape[1])*(1-overlap_factor)] ).astype(int)
        tic = time.time()
        #Para cada subdivisión se crean los cuadros interiores
        for frame_r in range(frames_r+1):
            for frame_c in range(frames_c+1):
                if frame_c < frames_c and frame_r < frames_r:
                    #Se calculan las coordenadas para el cuadro correspondiente
                    frame_central_coord = [ frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                    frame_central_coord_resized = [ frame_r * sld_step_resized[0], frame_r*sld_step_resized[0] + new_win_shape[0], frame_c*sld_step_resized[1],\
                        frame_c*sld_step_resized[1] + new_win_shape[1] ]
                    #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen
                    frame_array[frame_r, frame_c,:] = frame_central_coord
                    if len(img_resized.shape) > 2:
                        frame_array_in[ ( ( frames_c+1)*frame_r + frame_c ), : ,: ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3], :]
                    else:
                        frame_array_in[ ( ( frames_c+1)*frame_r + frame_c ), : ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3]]
                    #Se adjuntan las coordenadas para asociar a predicciones de CNN más adelante
                    frame_coordinates_in[ ( ( frames_c +1)*frame_r + frame_c ), :  ] = frame_central_coord
                #Última fila
                elif frame_c < frames_c and frame_r == frames_r:
                    frame_central_coord = [ img_rgb.shape[0]-win_size[0], img_rgb.shape[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                    frame_central_coord_resized = [ img_resized.shape[0]-new_win_shape[0], img_resized.shape[0], frame_c*sld_step_resized[1],\
                        frame_c*sld_step_resized[1] + new_win_shape[1] ]
                    #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen
                    frame_array[frames_r, frame_c,:] = frame_central_coord
                    if len(img_resized.shape) > 2:
                        frame_array_in[ ( ( frames_c+1)*(frames_r) + frame_c ), : ,: ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3], :]
                    else:
                        frame_array_in[ ( ( frames_c+1)*(frames_r) + frame_c ), : ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3]]
                    #Se adjuntan las coordenadas para asociar a predicciones de CNN más adelante
                    frame_coordinates_in[ ( ( frames_c +1)*frames_r + frame_c ), :  ] = frame_central_coord
                    
                #Última columna
                elif frame_c == frames_c and frame_r < frames_r:
                    frame_central_coord = [frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0],  img_rgb.shape[1]-win_size[1], img_rgb.shape[1] ]
                    frame_central_coord_resized = [ frame_r*sld_step_resized[0], frame_r*sld_step_resized[0] + new_win_shape[0],\
                        img_resized.shape[1]-new_win_shape[1], img_resized.shape[1] ]
                    #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen
                    frame_array[frame_r, frames_c,:] = frame_central_coord
                    if len(img_resized.shape) > 2:
                        frame_array_in[ ( ( frames_c+1)*(frame_r) + frames_c ), : ,: ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3], :]
                    else:
                        frame_array_in[ ( ( frames_c+1)*(frame_r) + frames_c ), : ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3]]
                    
                    #Se adjuntan las coordenadas para asociar a predicciones de CNN más adelante
                    frame_coordinates_in[ ( ( frames_c +1)*frame_r + frames_c ), :  ] = frame_central_coord
                    
                #Último cuadro
                else:
                    frame_central_coord = [img_rgb.shape[0]-win_size[0], img_rgb.shape[0], img_rgb.shape[1]-win_size[1], img_rgb.shape[1] ]
                    #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen
                    frame_array[frame_r, frames_c,:] = frame_central_coord
                    frame_array_in[ ( ( frames_c+1)*(frame_r) + frames_c ), : ,: ] = img_resized[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                        frame_central_coord_resized[2] : frame_central_coord_resized[3]]
                    
                    #Se adjuntan las coordenadas para asociar a predicciones de CNN más adelante
                    frame_coordinates_in[ ( ( frames_c +1)*frame_r + frames_c ), :  ] = frame_central_coord
        #frame_coordinates_in, frame_array_in = frame_coordinates_in[0:frame_ct], frame_array_in[0:frame_ct]
        if verbose: print('Tiempo frames ' + str(time.time()-tic))          
        return frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r+1, frames_c+1], frame_array

    #Para el caso de las ventanas deslizantes
    elif data_type == 'CNN_sld':
        
        #Se rellena la imagen con ceros para que no sobre espacio de la foto al deslizar la ventana
        img_rgb_shape = img_rgb.shape
        top = int(win_size[0]/2)#win_size[0] - img_rgb_shape[0] % sld_step[0] if img_rgb_shape[0] % sld_step[0] != 0 else 0
        bottom = top
        left = int(win_size[1]/2)#win_size[1] - img_rgb_shape[1] % sld_step[1] if img_rgb_shape[1] % sld_step[1] != 0 else 0
        right = left
        img_rgb_new = 0#np.copy(img_rgb)#cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT)
        
        #img_rgb_new = cv2.resize( img_rgb, ( int( ( img_rgb_shape[1] - win_size[1]) / sld_step[1] )  * sld_step[1] + win_size[1] ,\
        #    int( ( img_rgb_shape[0] - win_size[0]) / sld_step[0] )  * sld_step[0] + win_size[0] ) )
        #frames_r = int( ( img_rgb_new.shape[0] - win_size[0]  ) /  sld_step[0] ) + 1#np.floor( (img_rgb.shape[0]-win_size[0]) / ( win_size[0] * (1-overlap_factor) ) + 1  ).astype(int)#
        #frames_c = int( ( img_rgb_new.shape[1] - win_size[1]  ) /  sld_step[1] ) + 1#np.floor( (img_rgb.shape[1]-win_size[1]) / ( win_size[1] * (1-overlap_factor) ) + 1  ).astype(int)#
        frames_r = np.floor( (img_rgb.shape[0]-win_size[0]) / ( win_size[0] * (1-overlap_factor) ) + 1  ).astype(int)
        frames_c = np.floor( (img_rgb.shape[1]-win_size[1]) / ( win_size[1] * (1-overlap_factor) ) + 1  ).astype(int)
        
        if frames_r<0: frames_r = 1
        if frames_c<0: frames_c = 1
        #print('frames_r'), print(frames_r), print('frames_c'), print(frames_c)
        if len(img_rgb_shape) > 2: frame_array_in = np.zeros( ( (frames_r+1)*(frames_c+1), new_win_shape[0], new_win_shape[1], 3), dtype = np.uint8 )
        else: frame_array_in = np.zeros( ( (frames_r+1)*(frames_c+1), new_win_shape[0], new_win_shape[1]), dtype = np.uint8 )
        frame_coordinates_in = np.zeros( ( (frames_r+1)*(frames_c+1), 4), dtype = np.int32 )
        #new_img = np.zeros(img_rgb_new.shape, dtype = np.uint8)
        frame_array = np.zeros( ( (frames_r+1), (frames_c+1), 4), dtype = np.int32 )
        new_size = [new_win_shape[0]/win_size[0], new_win_shape[1]/win_size[1]]
        img_resized = cv2.resize(img_rgb, None, fx = new_size[1], fy = new_size[0])
        remainder_r = new_win_shape[0] - ( img_resized.shape[0] -  frames_r*new_win_shape[0]*(1-overlap_factor) )
        remainder_c = new_win_shape[1] - ( img_resized.shape[1] - frames_c*new_win_shape[1]*(1-overlap_factor) )
        img_resized_pad = cv2.copyMakeBorder(img_resized, np.ceil(remainder_r/2).astype(int), np.ceil(remainder_r/2).astype(int),\
            np.ceil(remainder_c/2).astype(int), np.ceil(remainder_c/2).astype(int), cv2.BORDER_CONSTANT)
        sld_step_resized = np.round( [(new_win_shape[0])*(1-overlap_factor),  (new_win_shape[1])*(1-overlap_factor)] ).astype(int)
        #frames_r = np.floor( (img_rgb.shape[0]-win_size[0]) / ( win_size[0] * (1-overlap_factor) ) + 1  ).astype(int)
        #frames_c = np.floor( (img_rgb.shape[1]-win_size[1]) / ( win_size[1] * (1-overlap_factor) ) + 1  ).astype(int)
        #raise ValueError('hasta acá optimización de cuadros')
        tic = time.time()
        ct = 0
        #Para cada subdivisión se crean los cuadros interiores
        for frame_r in range(frames_r+1):
            for frame_c in range(frames_c+1):
                ct += 1
                #Se calculan las coordenadas para el cuadro correspondiente
                frame_central_coord = [ frame_r * sld_step[0], frame_r*sld_step[0] + win_size[0], frame_c*sld_step[1], frame_c*sld_step[1] + win_size[1] ]
                frame_central_coord_resized = [ frame_r * sld_step_resized[0], frame_r*sld_step_resized[0] + new_win_shape[0], frame_c*sld_step_resized[1],\
                    frame_c*sld_step_resized[1] + new_win_shape[1] ]
                #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen
                frame_array[frame_r, frame_c,:] = frame_central_coord
                if len(img_resized.shape) > 2:
                    frame_array_in[ ( ( frames_c+1)*frame_r + frame_c ), : ,: ,: ] = img_resized_pad[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                frame_central_coord_resized[2] : frame_central_coord_resized[3], :]
                else:
                    frame_array_in[ ( ( frames_c+1)*frame_r + frame_c ), : ,: ] = img_resized_pad[ frame_central_coord_resized[0] : frame_central_coord_resized[1],\
                frame_central_coord_resized[2] : frame_central_coord_resized[3]]
                #Se adjuntan las coordenadas para asociar a predicciones de CNN más adelante
                frame_coordinates_in[ ( ( frames_c+1 )*frame_r + frame_c ), :  ] = frame_central_coord
            
        if verbose: print('Tiempo frames ' + str(time.time()-tic))          
        return frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r+1, frames_c+1], frame_array
        
#Función que separa imágenes dentro de una carpeta de dada en sub-carpetas de conjuntos de entrenamiento, validación y prueba
def folder_stratifier(img_folder, destiny_folder, distribution = [.7, .2, .1], randomized = False, nfold_pos = 0):
    
    #Todos los archivos que correspondan a imágenes se consideran dentro de la lista objetivo y se mezclan con una semilla
    np.random.seed(1)
    if not os.listdir(img_folder): raise ValueError('La carpeta ingresada no contiene archivos')
    img_name_list = [img_folder + '/' + s for s in os.listdir(img_folder) if s.endswith('.jpg') or s.endswith('.jpeg') or s.endswith('.png')]
    if randomized: np.random.shuffle(img_name_list) 
        
    #Se rellenan los conjuntos dada la distribución ingresada
    train_imgs = img_name_list[0:int(distribution[0] *len(img_name_list))]   
    val_imgs = img_name_list[int(distribution[0] *len(img_name_list)):int( (distribution[0] + distribution[1]) *len(img_name_list))]   
    test_imgs = img_name_list[int( (distribution[0] + distribution[1]) *len(img_name_list)):-1]   
    
    if distribution[2] != 0: test_imgs.append(img_name_list[-1] )
    
    #Se copian las imágenes desde la carpeta de origen hacia las de de destino
    for train_idx, train_img in enumerate(train_imgs):
        os.makedirs(destiny_folder + '/train', exist_ok=True)
        copyfile(train_img, destiny_folder + '/train/' + str(train_idx) + '.jpg')
    for val_idx, val_img in enumerate(val_imgs):
        os.makedirs(destiny_folder + '/val', exist_ok=True)
        copyfile(val_img, destiny_folder + '/val/' + str(val_idx) + '.jpg')
    for test_idx, test_img in enumerate(test_imgs):
        os.makedirs(destiny_folder + '/test', exist_ok=True)
        copyfile(test_img, destiny_folder + '/test/' + str(test_idx) + '.jpg')
#Función que define los métodos de traslación para aumentar la base de datos
def translate_pad(img_rgb, pad_factor, pad_img, pad_method = 1):
    
    #Se extrae la forma de la imagen y se crea la nueva imagen que será rellenada
    img_rgb_shape = img_rgb.shape
    new_img_pad = np.zeros(pad_img.shape, dtype = np.uint8)
    new_img_pad[:,:,:] = pad_img[:,:,:]
    
    #MÉTODO 1: Relleno de pad_factor filas y pad_factor columnas con el valor aleatorio arriba y a la izquierda de la imagen
    if pad_method == 0:
        img_shrink = cv2.resize( img_rgb, None, fx = 1-pad_factor, fy = 1-pad_factor )
        new_img_pad[int( pad_factor*img_rgb_shape[0] )-1:-1, int( pad_factor*img_rgb_shape[1] )-1:-1, :] = img_shrink
    #MÉTODO 2: Relleno de pad_factor filas con el valor aleatorio arriba de la imagen
    elif pad_method == 1:
        img_shrink = cv2.resize( img_rgb, None, fx = 1, fy = 1-pad_factor )
        new_img_pad[int( pad_factor*img_rgb_shape[0] )-1:-1, :, :] = img_shrink
    #MÉTODO 3: Relleno de pad_factor columnas con el valor aleatorio a la izquierda de la imagen
    elif pad_method == 2:
        img_shrink = cv2.resize( img_rgb, None, fx = 1-pad_factor, fy = 1)
        new_img_pad[:, int( pad_factor*img_rgb_shape[1] )-1:-1, :] = img_shrink
    #MÉTODO 4: Relleno de pad_factor filas y pad_factor columnas con el valor aleatorio arriba y a la derecha de la imagen
    elif pad_method == 3:
        img_shrink = cv2.resize( img_rgb, None, fx = 1-pad_factor, fy = 1-pad_factor)
        new_img_pad[int( pad_factor*img_rgb_shape[0] )-1:-1, 0:int( (1-pad_factor)*img_rgb_shape[1] ), :] = img_shrink
    #MÉTODO 5: Relleno de pad_factor columnas con el valor aleatorio a la derecha de la imagen
    elif pad_method == 4:
        img_shrink = cv2.resize( img_rgb, None, fx = 1-pad_factor, fy = 1)
        new_img_pad[:, 0:int( (1-pad_factor)*img_rgb_shape[1] ), :] = img_shrink
    #MÉTODO 6: Relleno de pad_factor filas y pad_factor columnas con el valor aleatorio abjo y a la izquierda de la imagen
    elif pad_method == 5:
        img_shrink = cv2.resize( img_rgb, None, fx = 1-pad_factor, fy = 1-pad_factor)
        new_img_pad[0:int( (1-pad_factor)*img_rgb_shape[0] ), int( pad_factor*img_rgb_shape[1] )-1:-1, :] = img_shrink
    #MÉTODO 7: Relleno de pad_factor filas con el valor aleatorio abajo de la imagen
    elif pad_method == 6:
        img_shrink = cv2.resize( img_rgb, None, fx = 1, fy = 1-pad_factor)
        new_img_pad[0:int( (1-pad_factor)*img_rgb_shape[0] ), :, :] = img_shrink
    #MÉTODO 8: Relleno de pad_factor columnas con el valor aleatorio a la derecha de la imagen
    elif pad_method == 7:
        img_shrink = cv2.resize( img_rgb, None, fx = 1-pad_factor, fy = 1-pad_factor)
        new_img_pad[0:int( (1-pad_factor)*img_rgb_shape[0] ), 0:int( (1-pad_factor)*img_rgb_shape[1] ), :] = img_shrink
    
    return new_img_pad
        
#Función que aumenta la base de datos a través de traslación con rellenos sólidos de colores aleatorios
def translate_augment(img_rgb_source, img_aug_dest, pad_factor, pad_methods = 4, color_rand = False):
    
    #Se carga la semilla para repetibilidad de resultados y los nombres válidos de imágenes
    np.random.seed(1)
    img_rgb_list = os.listdir(img_rgb_source)
    img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
    np.random.shuffle(img_name_list)
    new_img_pad_array = np.zeros((0))
    
    #Para cada imagen se crean las máscaras aleatorias de relleno y traslación
    for img_name in img_name_list:
        
        pad_list = np.random.choice(np.linspace(0, 7, 8), pad_methods, False).astype(int)
        img = cv2.imread(img_name)
        
        #Se crean las máscaras aleatorias con los métodos escogidos al azar
        for pad_method in pad_list:
            #Se crea la imagen con valores RGB aleatorios
            random_rgb = (np.random.rand(3, 1)*255).astype(np.uint8)*color_rand
            img_ones = np.ones( ( img.shape[0], img.shape[1],1 ), dtype = np.uint8)
            new_img_pad = np.concatenate(  (img_ones*random_rgb[0], img_ones*random_rgb[1], img_ones*random_rgb[2] ), axis = 2)
            #Se crea la imagen con todos los métodos encontrados al azar
            if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( translate_pad(img, pad_factor, new_img_pad, pad_method=pad_method), (1, img.shape[0], img.shape[1], 3) )
            else: new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( translate_pad(img, pad_factor, new_img_pad, pad_method=pad_method), (1, img.shape[0], img.shape[1], 3) ) ), axis = 0)

        #Se agrega la imagen original
        new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( img, (1, img.shape[0], img.shape[1], 3) ) ), axis = 0)
    
    #Se aleatoriza la lista de imágenes creada y se guardan en la carpeta de destino
    np.random.shuffle(new_img_pad_array)
    os.makedirs(img_aug_dest, exist_ok=True)
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite(img_aug_dest + '/' + str(idx) + '.jpg' ,new_img_pad_array[idx,:,:,:])

#Función que aumenta la base de datos en base a un 'zoom-out' con traslación
def zoomout_augment(img_rgb_source, img_aug_dest, keep_orig = True, overwrite = False, zoom_factor = 0.75, rl_pad = 0.5, tb_pad = 0.5, repeats = 1, color_rand = False, pad_rand = False):
    
    #Se carga la semilla para repetibilidad de resultados y los nombres válidos de imágenes
    np.random.seed(1)
    img_rgb_list = os.listdir(img_rgb_source)
    img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
    np.random.shuffle(img_name_list)
    new_img_pad_array = np.zeros((0))
    
    for _ in range(repeats):
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name in img_name_list:
            
            #Se lee la imagen y se disminuye en tamaño con respecto al "zoom_factor"
            img = cv2.imread(img_name)
            zoom_img = cv2.resize(img, None, fx = zoom_factor, fy = zoom_factor)
            
            #Se construye la imagen padeada de la forma aleatoria o fija (según se elija)
            value_bgr = list(map(int, list(np.int32(np.ravel( (np.random.rand(3, 1)*255).astype(np.uint8)*color_rand )))))
            if pad_rand:
                tb_pad = np.random.uniform(0, 1)
                top = np.round(tb_pad * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                bottom = np.round( (1-tb_pad) * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                rl_pad = np.random.uniform(0, 1)
                left = np.round(rl_pad * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                right = np.round( (1-rl_pad) * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
            else:
                top = np.round(tb_pad * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                bottom = np.round( (1-tb_pad) * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                left = np.round(rl_pad * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                right = np.round( (1-rl_pad) * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                
            new_img = cv2.copyMakeBorder(zoom_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = value_bgr)
            
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )

    #Se escriben las imágenes según se quiera reescribir o no
    len_orig = len(os.listdir(img_aug_dest)) if os.path.isdir(img_aug_dest) else 0
    os.makedirs(img_aug_dest, exist_ok=True)
    
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite( img_aug_dest + '/' + str(idx + (not overwrite) * len_orig ) + '.jpg', new_img_pad_array[idx,:,:,:] )
    
#Función que aumenta con método de rotación de imágenes
def rotation_augment(img_rgb_source, img_aug_dest, keep_orig = True, overwrite = False, repeats = 1, color_rand = False):
    np.random.seed(1)
    img_rgb_list = os.listdir(img_rgb_source)
    img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
    np.random.shuffle(img_name_list)
    new_img_pad_array = np.zeros((0))
    
    for _ in range(repeats):
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name in img_name_list:
            
            #Se lee la imagen y se realiza la aumentación
            img = cv2.imread(img_name)
            
            #Se construye la imagen padeada y se rota la imagen con el valor aleatorio
            value_bgr = list(map(int, list(np.int32(np.ravel( (np.random.rand(3, 1)*255).astype(np.uint8)*color_rand )))))
            rot = np.random.randint(1, 360)
            img_pad = img[:,:,:]
            img_pad = cv2.copyMakeBorder(img_pad, abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[0]))), abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[0])))\
                , abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[1]))), abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[1]))), cv2.BORDER_CONSTANT, value = value_bgr)
            img_center = tuple(np.array(img_pad.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(img_center, rot, 1.0)
            new_img = cv2.resize(cv2.warpAffine(img_pad, rot_mat, img_pad.shape[1::-1], flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = value_bgr), (img.shape[0], img.shape[1]) )
            
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
        
    #Se escriben las imágenes según se quiera reescribir o no
    len_orig = len(os.listdir(img_aug_dest)) if os.path.isdir(img_aug_dest) else 0
    os.makedirs(img_aug_dest, exist_ok=True)
    
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite( img_aug_dest + '/' + str(idx + (not overwrite) * len_orig ) + '.jpg', new_img_pad_array[idx,:,:,:] )
 
#Función que aumenta con método de alteración de brillo
def bright_augment(img_rgb_source, img_aug_dest, bright_change = 20, keep_orig = True, overwrite = False, repeats = 1):
    np.random.seed(1)
    img_rgb_list = os.listdir(img_rgb_source)
    img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
    np.random.shuffle(img_name_list)
    new_img_pad_array = np.zeros((0))
    
    for _ in range(repeats):
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name in img_name_list:
            
            #Se lee la imagen y se realiza la aumentación
            img = cv2.imread(img_name)
            
            #Se construye la imagen con valor aleatorio alterado
            h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            new_bright_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*bright_change
            if new_bright_change>0:
                v[v > 255-new_bright_change] = 255
                v[ (v <= 255-new_bright_change) ] += new_bright_change
            else:
                v[v < bright_change] = 0
                v[ (v >= bright_change) ] -= bright_change
            new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
            
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
        
    #Se escriben las imágenes según se quiera reescribir o no
    len_orig = len(os.listdir(img_aug_dest)) if os.path.isdir(img_aug_dest) else 0
    os.makedirs(img_aug_dest, exist_ok=True)
    
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite( img_aug_dest + '/' + str(idx + (not overwrite) * len_orig ) + '.jpg', new_img_pad_array[idx,:,:,:] ) 
    
#Función que aumenta alterando los valores H y S sobre la imagen aleatoriamente
def color_augment(img_rgb_source, img_aug_dest, color_variation = 20, keep_orig = True, overwrite = False, repeats = 1):
    np.random.seed(1)
    img_rgb_list = os.listdir(img_rgb_source)
    img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
    np.random.shuffle(img_name_list)
    new_img_pad_array = np.zeros((0))
    
    for _ in range(repeats):
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name in img_name_list:
            
            #Se lee la imagen y se realiza la aumentación
            img = cv2.imread(img_name)
            
            #Se construye la imagen con color aleatorio alterado
            h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            new_hue_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
            new_sat_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
            if new_hue_change>0:
                h[h > 255-new_hue_change] = 255
                h[ (h <= 255-new_hue_change) ] += new_hue_change
            else:
                h[h < -1*new_hue_change] = 0
                h[ (h >= -1*new_hue_change) ] -= -1*new_hue_change
            if new_sat_change>0:
                s[s > 255-new_sat_change] = 255
                s[ (s <= 255-new_sat_change) ] += new_sat_change
            else:
                s[s < -1*new_sat_change] = 0
                s[ (s >= -1*new_sat_change) ] -= -1*new_sat_change
            new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
            
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
        
    #Se escriben las imágenes según se quiera reescribir o no
    len_orig = len(os.listdir(img_aug_dest)) if os.path.isdir(img_aug_dest) else 0
    os.makedirs(img_aug_dest, exist_ok=True)
    
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite( img_aug_dest + '/' + str(idx + (not overwrite) * len_orig ) + '.jpg', new_img_pad_array[idx,:,:,:] )

#Función que aumenta sumando una imagen de ruido gaussiano RGN sobre una carpeta de imágenes
def noise_augment(img_rgb_source, img_aug_dest, noise_var = 20, keep_orig = True, overwrite = False, repeats = 1):
    
    np.random.seed(1)
    img_rgb_list = os.listdir(img_rgb_source)
    img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
    np.random.shuffle(img_name_list)
    new_img_pad_array = np.zeros((0))
    
    for _ in range(repeats):
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name in img_name_list:
            
            #Se lee la imagen y se realiza la aumentación
            img = cv2.imread(img_name)
            
            #Se genera la imagen de ruido y se suma teniendo cuidado en el overflow por tipo de dato
            new_img = np.random.normal(0, noise_var, img.shape) + img.astype(np.int32)
            new_img[new_img<0] = 0
            new_img[new_img>255] = 255
            new_img = new_img.astype(np.uint8)
            '''
            #Se construye la imagen con color aleatorio alterado
            h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            new_hue_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
            new_sat_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
            if new_hue_change>0:
                h[h > 255-new_hue_change] = 255
                h[ (h <= 255-new_hue_change) ] += new_hue_change
            else:
                h[h < -1*new_hue_change] = 0
                h[ (h >= -1*new_hue_change) ] -= -1*new_hue_change
            if new_sat_change>0:
                s[s > 255-new_sat_change] = 255
                s[ (s <= 255-new_sat_change) ] += new_sat_change
            else:
                s[s < -1*new_sat_change] = 0
                s[ (s >= -1*new_sat_change) ] -= -1*new_sat_change
            new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
            '''
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
        
    #Se escriben las imágenes según se quiera reescribir o no
    len_orig = len(os.listdir(img_aug_dest)) if os.path.isdir(img_aug_dest) else 0
    os.makedirs(img_aug_dest, exist_ok=True)
    
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite( img_aug_dest + '/' + str(idx + (not overwrite) * len_orig ) + '.jpg', new_img_pad_array[idx,:,:,:] )
    
#Función que realiza aumentación aleatoria sobre una carpeta de imágenes con los métodos ingresados
def random_augment(img_rgb_source, img_aug_dest, augment_list = ['flip', 'rot', 'bright', 'pad', 'color', 'zoom', 'noise' ], bright_change = 20, color_variation = 20, keep_orig = True, overwrite = False,\
    double_aug = 'rand', color_rand = False, pad_rand = True, zoom_factor = 0.75, noise_var = 30, original_folder_name = '', input_array = False):
    
    #Si se quieren sobreescribir los datos
    if overwrite and os.path.isdir(img_aug_dest): rmtree(img_aug_dest)
    os.makedirs(img_aug_dest, exist_ok=True)
    
    #Se carga la semilla para repetibilidad de resultados y los nombres válidos de imágenes
    #np.random.seed(1)
    #Si se eligió que entraran imágenes (tipo X_train) se realiza la aumentación por separado
    if input_array:
        
        new_img_pad_array = np.zeros((0))
        new_augment_list = augment_list[:]
        log_name = img_aug_dest + '/stats_log.txt'
        img_idx = 0
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name_idx, img in enumerate(img_rgb_source):
            if len(new_augment_list) != 0: np.random.shuffle(new_augment_list)
            else: new_augment_list = augment_list[:]

            #Se determina si se realizará o no doble aumentación
            if double_aug == 'Yes': keep_aumeg = 1 
            elif double_aug == 'No': keep_aumeg = 0
            else: keep_aumeg = np.random.randint(0,2)
            
            #Se obtiene el método de aumentación siguiente y se aplica a la imagen
            augment_method = new_augment_list.pop(0)
            orig_img = np.copy(img)
            value_bgr = list(map(int, list(np.int32(np.ravel( (np.random.rand(3, 1)*255*color_rand).astype(np.uint8) )))))
            
            #Se empieza a escribir el log de aumentación por imagen
            #Si ya existe el archivo simplemente se apendizan los datos
            if os.path.isfile(log_name):
                log_file = open(log_name,'a')
                txt_2_save = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image number '+ str(img_idx + 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' 
                img_idx += 1
                    #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
            #Si no, se escribe con encabezado
            else:
                log_file = open(log_name,'w+')
                txt_2_save = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image augmentation log~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image number: '+ str(img_idx+ 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
                img_idx += 1
                    #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
            if keep_orig:
                txt_2_save = txt_2_save + '- Augmentation method: ' + 'None' '\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image number '+ str(img_idx + 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' 
                img_idx += 1
                
            #Se itera hasta que se termine la o las aumentaciones
            aug_idx = 0
            while True:
                if augment_method == 'flip':
                    aug_idx += 1
                    flip = np.random.randint(0,2)
                    new_img = cv2.flip(img, flip)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value: ' + str(flip) + '\r\n'
                elif augment_method == 'bright':
                    aug_idx += 1
                    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    new_bright_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*bright_change
                    if new_bright_change>0:
                        v[v > 255-new_bright_change] = 255
                        v[ (v <= 255-new_bright_change) ] += new_bright_change
                    else:
                        v[v < bright_change] = 0
                        v[ (v >= bright_change) ] -= bright_change
                    new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value: ' + str(new_bright_change) + '\r\n'
                elif augment_method == 'color':
                    aug_idx += 1
                    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    new_hue_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
                    new_sat_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
                    if new_hue_change>0:
                        h[h > 255-new_hue_change] = 255
                        h[ (h <= 255-new_hue_change) ] += new_hue_change
                    else:
                        h[h < -1*new_hue_change] = 0
                        h[ (h >= -1*new_hue_change) ] -= -1*new_hue_change
                    if new_sat_change>0:
                        s[s > 255-new_sat_change] = 255
                        s[ (s <= 255-new_sat_change) ] += new_sat_change
                    else:
                        s[s < -1*new_sat_change] = 0
                        s[ (s >= -1*new_sat_change) ] -= -1*new_sat_change
                    new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with values (hue, sat): ' + str((new_hue_change, new_sat_change)) + '\r\n'
                elif augment_method == 'rot':
                    aug_idx += 1
                    #Se construye la imagen padeada y se rota la imagen con el valor aleatorio
                    rot = np.random.randint(1, 360)
                    img_pad = img[:,:,:]
                    img_pad = cv2.copyMakeBorder(img_pad, abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[0]))), abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[0])))\
                        , abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[1]))), abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[1]))), cv2.BORDER_CONSTANT, value = value_bgr)
                    img_center = tuple(np.array(img_pad.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(img_center, rot, 1.0)
                    new_img = cv2.resize(cv2.warpAffine(img_pad, rot_mat, img_pad.shape[1::-1], flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = value_bgr), (img.shape[0], img.shape[1]) )
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value : ' + str(rot) + '\r\n'
                elif augment_method == 'pad':
                    aug_idx += 1
                    pad_method = np.random.randint(0,8)
                    img_ones = np.ones( ( img.shape[0], img.shape[1],1 ), dtype = np.uint8)
                    new_img_pad = np.concatenate(  (img_ones*value_bgr[0], img_ones*value_bgr[1], img_ones*value_bgr[2] ), axis = 2)
                    new_img = translate_pad(img, 0.25, new_img_pad, pad_method = pad_method)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value : ' + str(pad_method) + '\r\n'
                elif augment_method == 'zoom':
                    aug_idx += 1
                    zoom_img = cv2.resize(img, None, fx = zoom_factor, fy = zoom_factor)
                    #Se construye la imagen padeada de la forma aleatoria o fija (según se elija)
                    if pad_rand:
                        tb_pad = np.random.uniform(0, 1)
                        top = np.round(tb_pad * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        bottom = np.round( (1-tb_pad) * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        rl_pad = np.random.uniform(0, 1)
                        left = np.round(rl_pad * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                        right = np.round( (1-rl_pad) * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                    else:
                        top = np.round(tb_pad * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        bottom = np.round( (1-tb_pad) * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        left = np.round(rl_pad * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                        right = np.round( (1-rl_pad) * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                    new_img = cv2.copyMakeBorder(zoom_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = value_bgr)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with values (top, bottom, left, right) : ' + str((top, bottom, left, right)) + '\r\n'
                elif augment_method =='noise':
                    aug_idx += 1
                    noise_mat = np.random.normal(0, noise_var, img.shape)
                    new_img = noise_mat + img.astype(np.int32)
                    new_img[new_img<0] = 0
                    new_img[new_img>255] = 255
                    new_img = new_img.astype(np.uint8)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value : ' + str(noise_var) + '\r\n'
                if keep_aumeg == 1: 
                    if len(new_augment_list) == 0: break
                    img = new_img[:,:,:]
                    keep_aumeg = 0
                    augment_method = new_augment_list[0]
                else:
                    break
            
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( orig_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( orig_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
            
            #Se escriben los datos sobre el log de aumentación    
            log_file.write(txt_2_save)
            log_file.close()        
    else:
        
        img_rgb_list = [s for s in os.listdir(img_rgb_source) if s.endswith('.jpg') or s.endswith('.jpeg')]
        #np.random.shuffle(img_rgb_list)
        img_name_list = [img_rgb_source + '/' + img_rgb_name for img_rgb_name in img_rgb_list if img_rgb_name.endswith('.jpg') or img_rgb_name.endswith('.jpeg') or img_rgb_name.endswith('.png')]
        new_img_pad_array = np.zeros((0))
        new_augment_list = augment_list[:]
        log_name = img_aug_dest + '/stats_log.txt'
        img_idx = 0
        #Se lee cada imagen y se generan las imágenes aumentadas correspondientes
        for img_name_idx, img_name in enumerate(img_name_list):
            if len(new_augment_list) != 0: np.random.shuffle(new_augment_list)
            else: new_augment_list = augment_list[:]

            #Se determina si se realizará o no doble aumentación
            if double_aug == 'Yes': keep_aumeg = 1 
            elif double_aug == 'No': keep_aumeg = 0
            else: keep_aumeg = np.random.randint(0,2)
            
            #Se obtiene el método de aumentación siguiente y se aplica a la imagen
            augment_method = new_augment_list.pop(0)
            img = cv2.imread(img_name)
            orig_img = np.copy(img)
            value_bgr = list(map(int, list(np.int32(np.ravel( (np.random.rand(3, 1)*255*color_rand).astype(np.uint8) )))))
            
            #Se empieza a escribir el log de aumentación por imagen
            #Si ya existe el archivo simplemente se apendizan los datos
            if os.path.isfile(log_name):
                log_file = open(log_name,'a')
                txt_2_save = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image number '+ str(img_idx + 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '- Original image name: ' + img_rgb_list[img_name_idx] + '\r\n'
                img_idx += 1
                    #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
            #Si no, se escribe con encabezado
            else:
                log_file = open(log_name,'w+')
                txt_2_save = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image augmentation log~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image number: '+ str(img_idx+ 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '- Original image name: ' + img_rgb_list[img_name_idx] + '\r\n'
                img_idx += 1
                    #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n'
            if keep_orig:
                txt_2_save = txt_2_save + '- Augmentation method: ' + 'None' '\r\n' + \
                    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Image number '+ str(img_idx + 1) +':~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n' + \
                    '- Original image name: ' + img_name + '\r\n'
                img_idx += 1
                
            #Se itera hasta que se termine la o las aumentaciones
            aug_idx = 0
            while True:
                if augment_method == 'flip':
                    aug_idx += 1
                    flip = np.random.randint(0,2)
                    new_img = cv2.flip(img, flip)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value: ' + str(flip) + '\r\n'
                elif augment_method == 'bright':
                    aug_idx += 1
                    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    new_bright_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*bright_change
                    if new_bright_change>0:
                        v[v > 255-new_bright_change] = 255
                        v[ (v <= 255-new_bright_change) ] += new_bright_change
                    else:
                        v[v < bright_change] = 0
                        v[ (v >= bright_change) ] -= bright_change
                    new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value: ' + str(new_bright_change) + '\r\n'
                elif augment_method == 'color':
                    aug_idx += 1
                    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
                    new_hue_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
                    new_sat_change = (-1 + 2*np.random.randint(0,2,dtype=np.int16))*color_variation
                    if new_hue_change>0:
                        h[h > 255-new_hue_change] = 255
                        h[ (h <= 255-new_hue_change) ] += new_hue_change
                    else:
                        h[h < -1*new_hue_change] = 0
                        h[ (h >= -1*new_hue_change) ] -= -1*new_hue_change
                    if new_sat_change>0:
                        s[s > 255-new_sat_change] = 255
                        s[ (s <= 255-new_sat_change) ] += new_sat_change
                    else:
                        s[s < -1*new_sat_change] = 0
                        s[ (s >= -1*new_sat_change) ] -= -1*new_sat_change
                    new_img = cv2.cvtColor( cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with values (hue, sat): ' + str((new_hue_change, new_sat_change)) + '\r\n'
                elif augment_method == 'rot':
                    aug_idx += 1
                    #Se construye la imagen padeada y se rota la imagen con el valor aleatorio
                    rot = np.random.randint(1, 360)
                    img_pad = img[:,:,:]
                    img_pad = cv2.copyMakeBorder(img_pad, abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[0]))), abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[0])))\
                        , abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[1]))), abs(int(np.round(np.sin(rot*np.pi/90)* ((np.sqrt(2)-1) / 2)*img.shape[1]))), cv2.BORDER_CONSTANT, value = value_bgr)
                    img_center = tuple(np.array(img_pad.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(img_center, rot, 1.0)
                    new_img = cv2.resize(cv2.warpAffine(img_pad, rot_mat, img_pad.shape[1::-1], flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = value_bgr), (img.shape[0], img.shape[1]) )
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value : ' + str(rot) + '\r\n'
                elif augment_method == 'pad':
                    aug_idx += 1
                    pad_method = np.random.randint(0,8)
                    img_ones = np.ones( ( img.shape[0], img.shape[1],1 ), dtype = np.uint8)
                    new_img_pad = np.concatenate(  (img_ones*value_bgr[0], img_ones*value_bgr[1], img_ones*value_bgr[2] ), axis = 2)
                    new_img = translate_pad(img, 0.25, new_img_pad, pad_method = pad_method)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value : ' + str(pad_method) + '\r\n'
                elif augment_method == 'zoom':
                    aug_idx += 1
                    zoom_img = cv2.resize(img, None, fx = zoom_factor, fy = zoom_factor)
                    #Se construye la imagen padeada de la forma aleatoria o fija (según se elija)
                    if pad_rand:
                        tb_pad = np.random.uniform(0, 1)
                        top = np.round(tb_pad * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        bottom = np.round( (1-tb_pad) * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        rl_pad = np.random.uniform(0, 1)
                        left = np.round(rl_pad * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                        right = np.round( (1-rl_pad) * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                    else:
                        top = np.round(tb_pad * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        bottom = np.round( (1-tb_pad) * ( img.shape[0] - zoom_img.shape[0] )).astype(int)
                        left = np.round(rl_pad * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                        right = np.round( (1-rl_pad) * ( img.shape[1] - zoom_img.shape[1] )).astype(int)
                    new_img = cv2.copyMakeBorder(zoom_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = value_bgr)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with values (top, bottom, left, right) : ' + str((top, bottom, left, right)) + '\r\n'
                elif augment_method =='noise':
                    aug_idx += 1
                    noise_mat = np.random.normal(0, noise_var, img.shape)
                    new_img = noise_mat + img.astype(np.int32)
                    new_img[new_img<0] = 0
                    new_img[new_img>255] = 255
                    new_img = new_img.astype(np.uint8)
                    txt_2_save = txt_2_save + '- Augmentation method: ' + str(aug_idx) + ' ' + augment_method + ' with value : ' + str(noise_var) + '\r\n'
                if keep_aumeg == 1: 
                    if len(new_augment_list) == 0: break
                    img = new_img[:,:,:]
                    keep_aumeg = 0
                    augment_method = new_augment_list[0]
                else:
                    break
            
            #Si se define que se quiere mantener la imagen original además de la aumentada
            if keep_orig:
                if new_img_pad_array.shape[0] == 0:
                    new_img_pad_array = np.reshape( orig_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                else: 
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( orig_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    new_img_pad_array = np.concatenate( (new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
                    
            else:
                if new_img_pad_array.shape[0] == 0: new_img_pad_array = np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )
                else: new_img_pad_array = np.concatenate( ( new_img_pad_array, np.reshape( new_img, (1, img.shape[0], img.shape[1], img.shape[2]) )), axis = 0 )
            
            #Se escriben los datos sobre el log de aumentación    
            log_file.write(txt_2_save)
            log_file.close()
            
    #Se escriben las imágenes según se quiera reescribir o no y se devuelve el arreglo total de datos aumentados
    len_orig = len(os.listdir(img_aug_dest)) if os.path.isdir(img_aug_dest) else 0
    
    for idx in range(new_img_pad_array.shape[0]): cv2.imwrite( img_aug_dest + '/' + str(idx + (not overwrite) * len_orig ) + '.jpg', new_img_pad_array[idx,:,:,:] )
    return new_img_pad_array

    
#Función que calcula el auto-umbral Otsu (NO USAR POR EL MOMENTO)
def otsu_thresh(img_hsv):

    V = img_hsv[:,:,2]
    
    if np.where(V == 0)[0].shape[0] != V.shape[0]*V.shape[1]:
        from skimage.filters import threshold_otsu
        thresholds = threshold_otsu(V)
        print(thresholds)
        regions = np.digitize(V, bins=thresholds)
        shape_V = V.shape
        unos = np.ones(shape_V)
        ret, th = cv2.threshold(V, 0, 1 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        V[V<ret] = 0
        new_V = V #np.multiply(V, unos-th)
        print(th)
        print(ret)
        img_hsv[:,:,2] = new_V
        
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        cv2.imshow('', cv2.resize( regions,(480,480) ))
        cv2.waitKey(0)
        
    return(img_hsv)

#Función para el etiquetado detallado de regiones dentro de imágenes para evaluación de los métodos de identificación
def in_img_tag(folder_name, multires_name = 'wild_lettuce', hm_name = 'trebol', bg_name = 'pasto', region_size = (1800, 1800), frame_size = (256, 256), frame_overlap = .75, multires_method = 'corners',\
    savedir = '', overwrite = False):
    
    #Se cargan las imágenes con el formato admitido
    img_name_list = [folder_name + '/' +  s for s in os.listdir(folder_name) if s.endswith('.jpg')]
    param_dict = { 'region_size' : region_size, 'frame_size' : frame_size  }
    
    if not overwrite:
        from pickle import load as pickle_load, HIGHEST_PROTOCOL
        seg_dict_name = savedir + '/seg_dict.pickle'
        if os.path.isfile(seg_dict_name):
            with open(seg_dict_name, 'rb') as handle: seg_dict = pickle_load(handle)
            new_img_name_list = img_name_list[len(seg_dict['img_size']):-1]
            new_img_name_list.append(img_name_list[-1])
            img_name_list = new_img_name_list
            print('Datos cargados')
        else:
            seg_dict = {'img_size' : [], 'multires_coords' : [], 'multires_wh' : [], 'bin_hm' : [], 'est_dens': [], 'param_dict': [param_dict]}
            print('No existen datos para cargar')
    else: seg_dict = {'img_size' : [], 'multires_coords' : [], 'multires_wh' : [], 'bin_hm' : [], 'est_dens': [], 'param_dict': [param_dict]}
    
    #Para cada imagen se etiquetan los objetos
    for img_idx, img_name in enumerate(img_name_list):
        #Se despliega cada imagen válida
        string = 'Imagen ' + str(img_idx+1) + ' de ' + str(len(img_name_list))
        img = cv2.imread(img_name)
        multires_coord, multires_wh = [], []
        print(string)
        img_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), string)
        
        ##ETIQUETADO DE WILD LETTUCE##
        img_seg = np.copy(img)
        YN_wl = YNC_prompt('¿Ve una maleza tipo wild lettuce en la imagen?')
        ver_wl = YN_wl.exec_()
        #Si existe una maliza de este tipo, se procede a seleccionar
        while ver_wl == 1:
            #Se eligen las esquinas y se calcula el centro de la región wild lettuce
            if multires_method == 'corners':
                img_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Elija las esquinas')
                corners = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)
                [x, y] = [[ corners[0,0], corners[1,0] ], [ corners[0,1], corners[1,1] ]]
                [x.sort(), y.sort()]
                [x_center, y_center] = [ int( ( x[0] + x[1] ) / 2 ), int( ( y[0] + y[1] ) / 2 ) ]
                [w, h] = [ int( x[1]-x[0] ), int( y[1]-y[0] ) ]
                #Después de elegir las esquinas se pregunta si el objeto está bien segmentado
                x_multires = [x[0],  x[1] ]
                y_multires = [ y[0], y[1] ]
            elif multires_method == 'center':
                img_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Elija el centro DOS VECES')
                center_coords = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)
                [x, y] = [[ center_coords[0,0], center_coords[1,0] ], [ center_coords[0,1], center_coords[1,1] ]]
                [x.sort(), y.sort()]
                [x_center, y_center] = [ int( ( x[0] + x[1] ) / 2 ), int( ( y[0] + y[1] ) / 2 ) ]
                #Después de elegir las esquinas se pregunta si el objeto está bien segmentado
                x_multires = [ np.max( [ 0, x_center- int(region_size[1]/2) ] ),  np.min( [ img.shape[1]-1, x_center + int(region_size[1]/2) ] ) ]
                y_multires = [ np.max( [ 0, y_center- int(region_size[0]/2) ] ),  np.min( [ img.shape[0]-1, y_center + int(region_size[0]/2) ] ) ]
                [w, h] = [ int( x[1]-x[0] ), int( y[1]-y[0] ) ]
        
            img_display(cv2.cvtColor(img[y_multires[0]:y_multires[1], x_multires[0] : x_multires[1],:], cv2.COLOR_BGR2RGB), string)
            YN_good_crop = YNC_prompt('¿Está bien etiquetado el objeto?')
            ver_good_crop = YN_good_crop.exec_()
            
            #Si no, se repite hasta un sí o cancel
            while ver_good_crop == 0:
                #Se eligen las esquinas y se rellena con 0's según se necesiten
                img_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Elija las esquinas')
                corners = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)
                [x, y] = [[ corners[0,0], corners[1,0] ], [ corners[0,1], corners[1,1] ]]
                [x.sort(), y.sort()]
                
                [x_center, y_center] = [ int( ( x[0] + x[1] ) / 2 ), int( ( y[0] + y[1] ) / 2 ) ]
                
                #Después de elegir las esquinas se pregunta si el objeto está bien segmentado
                x_multires = [ np.max( [ 0, x_center- int(region_size[1]/2) ] ),  np.min( [ img.shape[1]-1, x_center + int(region_size[1]/2) ] ) ]
                y_multires = [ np.max( [ 0, y_center- int(region_size[0]/2) ] ),  np.min( [ img.shape[0]-1, y_center + int(region_size[0]/2) ] ) ]
            
                img_display(cv2.cvtColor(img[y_multires[0]:y_multires[1], x_multires[0] : x_multires[1],:], cv2.COLOR_BGR2RGB), string)
                YN_good_crop = YNC_prompt('¿Está bien etiquetado el objeto?')
                ver_good_crop = YN_good_crop.exec_()
            #Si se decide que está bien etiquetado el objeto se guardan las coordenadas centrales y se dibuja un cuadrado sobre las coordenadas escogidas
            if ver_good_crop == 1: multires_coord.append([y_center, x_center]), multires_wh.append([w, h])
            img_seg[y_multires[0]-10:y_multires[0]+10, x_multires[0] : x_multires[1],0:2] = 0
            img_seg[y_multires[0]-10:y_multires[0]+10, x_multires[0] : x_multires[1],2] = 255
            img_seg[y_multires[1]-10:y_multires[1]+10, x_multires[0] : x_multires[1],0:2] = 0
            img_seg[y_multires[1]-10:y_multires[1]+10, x_multires[0] : x_multires[1],2] = 255
            img_seg[y_multires[0]:y_multires[1], x_multires[0]-10 : x_multires[0]+10,0:2] = 0
            img_seg[y_multires[0]:y_multires[1], x_multires[0]-10 : x_multires[0]+10,2] = 255
            img_seg[y_multires[0]:y_multires[1], x_multires[1]-10 : x_multires[1]+10,0:2] = 0
            img_seg[y_multires[0]:y_multires[1], x_multires[1]-10 : x_multires[1]+10,2] = 255
            
            #Se vuelve a preguntar si existe (otra) maleza tipo WL
            img_display(cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB), 'Imagen segmentada actualizada')
            YN_wl = YNC_prompt('¿Ve otra maleza tipo wild lettuce en la imagen?')
            ver_wl = YN_wl.exec_()
            
        ##ETIQUETADO DE TREBOL##
        img_display(cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB), '¿Ve una maleza tipo trébol en la imagen?')
        heat_map_tot, hmt_max = np.zeros( ( img_seg.shape[0], img_seg.shape[1] ) ), 1
        YN_dw = YNC_prompt('¿Ve una maleza tipo trébol en la imagen?')
        ver_dw = YN_dw.exec_()
        ver_img = np.zeros(img.shape)
        
        #Mientras siga existiendo maleza tipo trébol que se quiera etiquetar se continúa el algoritmo
        while ver_dw == 1:
            #Se cortan las esquinas de región tipo trébol
            img_display(cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB), 'Elija las esquinas')
            corners = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)
            [x_dw_reg, y_dw_reg] = [[ corners[0,0], corners[1,0] ], [ corners[0,1], corners[1,1] ]]
            [x_dw_reg.sort(), y_dw_reg.sort()]
            cropped_img = img[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[0]:x_dw_reg[1],:]
            #Se corta el cuadro de la imagen que se segmentará como heat map manualmente
            img_display(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 'Imagen recortada')
            YN_good_crop = YNC_prompt('¿Está bien el recorte?')
            ver_good_crop = YN_good_crop.exec_()
            
            #Si no, se repite hasta un sí o cancel
            while ver_good_crop == 0:
                img_display(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'Elija las esquinas')
                corners = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)
                [x_dw_reg, y_dw_reg] = [[ corners[0,0], corners[1,0] ], [ corners[0,1], corners[1,1] ]]
                [x_dw_reg.sort(), y_dw_reg.sort()]
                cropped_img = img[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[0]:x_dw_reg[1],:]
                #Se corta el cuadro de la imagen que se segmentará como heat map manualmente
                img_display(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 'Imagen recortada')
                YN_good_crop = YNC_prompt('¿Está bien el recorte?')
                ver_good_crop = YN_good_crop.exec_()
                
            #Si el recorte está bien, se prosigue a etiquetar en forma de densidad
            if ver_good_crop == 1:
                
                #Se crean los cuadros para etiquetar a mano
                frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], _ =\
                    slide_window_creator(cropped_img, win_size = frame_size, new_win_shape = frame_size, overlap_factor = frame_overlap, data_type = 'CNN_sld')
                heat_map_temp = np.zeros((cropped_img.shape[0], cropped_img.shape[1]))
                r_exc, c_exc = np.ceil([(frame_coordinates_in[-1,1]-cropped_img.shape[0])/2 , (frame_coordinates_in[-1,3]-cropped_img.shape[1])/2] ).astype(int)
                
                
                for frame_idx, frame_coordinate in enumerate(frame_coordinates_in):
                    
                    #Se pregunta si el cuadro contiene trébol
                    img_display(cv2.cvtColor(frame_array_in[frame_idx], cv2.COLOR_BGR2RGB), 'Cuadro ' + str(frame_idx+1) + ' de ' + str(frame_coordinates_in.shape[0]))
                    YN_heat_map =  YNC_prompt('¿Hay un fragmento de trébol en el cuadro?')
                    ver_heat_map = YN_heat_map.exec_()
                    if ver_heat_map == 1: 
                        #Si es así, se pregunta si es todo el cuadro o se debe re-etiquetar a mano
                        YN_all_frame =  YNC_prompt('¿El trébol ocupa todo el cuadro?')
                        ver_all_frame = YN_all_frame.exec_()
                        if ver_all_frame == 1: 
                            heat_map_temp[np.max([frame_coordinate[0]-r_exc,0]):frame_coordinate[1]-r_exc, np.max([0, frame_coordinate[2]-c_exc]):frame_coordinate[3]-c_exc] += ver_heat_map > 0 
                            
                        elif ver_all_frame == 0:
                            img_display(cv2.cvtColor(frame_array_in[frame_idx], cv2.COLOR_BGR2RGB), 'Elija las esquinas')
                            dw_coordinates = np.asarray(plt.ginput(2, timeout=-1), dtype = np.uint16)
                            [dw_x, dw_y] = [[ dw_coordinates[0,0], dw_coordinates[1,0] ], [ dw_coordinates[0,1], dw_coordinates[1,1] ]]
                            [dw_x.sort(), dw_y.sort()]
                            new_frame_coordinate = [np.max([0, frame_coordinate[0] + dw_y[0]-r_exc]), frame_coordinate[0] + dw_y[1]-r_exc,\
                                np.max([0, frame_coordinate[2] + dw_x[0]-c_exc]), frame_coordinate[2] + dw_x[1]-c_exc]
                            print(new_frame_coordinate[0] )
                            #Si la imagen creada existe entre los márgenes de la imagen original se suma al heat map
                            heat_map_temp[new_frame_coordinate[0]:new_frame_coordinate[1],new_frame_coordinate[2]:new_frame_coordinate[3]] += 1
                
                hmt_max = np.max( heat_map_temp + np.spacing(1) )
                heat_map_temp = ( heat_map_temp / hmt_max )
                heat_map_temp[heat_map_temp < 1/ hmt_max] = 0
                                                
                #Se dibuja un cuadro para mostrar la región de trébol segmentada
                img_seg[y_dw_reg[0]-5:y_dw_reg[0]+5, x_dw_reg[0] : x_dw_reg[1],0:2] = 0
                img_seg[y_dw_reg[0]-5:y_dw_reg[0]+5, x_dw_reg[0] : x_dw_reg[1],2] = 255
                img_seg[y_dw_reg[1]-5:y_dw_reg[1]+5, x_dw_reg[0] : x_dw_reg[1],0:2] = 0
                img_seg[y_dw_reg[1]-5:y_dw_reg[1]+5, x_dw_reg[0] : x_dw_reg[1],2] = 255
                img_seg[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[0]-5 : x_dw_reg[0]+5,0:2] = 0
                img_seg[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[0]-5 : x_dw_reg[0]+5,2] = 255
                img_seg[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[1]-5 : x_dw_reg[1]+5,0:2] = 0
                img_seg[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[1]-5 : x_dw_reg[1]+5,2] = 255
                
            heat_map_tot[y_dw_reg[0]:y_dw_reg[1], x_dw_reg[0] : x_dw_reg[1]] = heat_map_temp
            img_seg[heat_map_tot>= 1/hmt_max ,1] = 0
            cv2.imshow('heat_map_tot', cv2.resize( (np.multiply( np.repeat(heat_map_tot[:, :, np.newaxis], 3, axis=2), img_seg )).astype(np.uint8) , None, fx = .15, fy = .15))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #Se vuelve a preguntar si se ve una región tipo trébol
            img_display(cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB), '¿Ve otra maleza tipo trébol en la imagen?')
        
            YN_dw = YNC_prompt('¿Ve otra maleza tipo trébol en la imagen?')
            ver_dw = YN_dw.exec_()
            
        #Se actualiza el diccionario con los datos de cada imagen
        seg_dict['img_size'].append(img.shape)
        seg_dict['multires_coords'].append(multires_coord)
        seg_dict['multires_wh'].append(multires_wh)
        seg_dict['bin_hm'].append(heat_map_tot[:,:] >= 1/hmt_max)
        seg_dict['est_dens'].append(np.sum(heat_map_tot)/(img.shape[0]*img.shape[1]))
        seg_dict['param_dict'].append(param_dict)
        plt.close('all')
    
        #Si se ingresa la dirección de guardado, se guarda el diccionario de segmentación
        if savedir: 
            from pickle import dump as pickle_dump, HIGHEST_PROTOCOL
            os.makedirs(savedir, exist_ok=True)
            seg_dict_name = savedir + '/seg_dict.pickle'
            with open(seg_dict_name, 'wb') as handle: pickle_dump(seg_dict, handle, protocol=HIGHEST_PROTOCOL)
            print(seg_dict)
    
    return seg_dict
    
#Función para eliminar las exif data de las imágenes originales
def exif_deleter(origin_folder, destiny_folder):
    
    #Se leen los nombres de imágenes válidos
    if os.listdir(origin_folder): 
        img_name_list = [origin_folder + '/' + s for s in os.listdir(origin_folder) if s.endswith('.jpg') or s.endswith('.jpeg') or s.endswith('.png')]
        os.makedirs(destiny_folder, exist_ok = True)
        img_destiny_list = [destiny_folder + '/' + s for s in os.listdir(origin_folder) if s.endswith('.jpg') or s.endswith('.jpeg') or s.endswith('.png')]
    else: raise ValueError('Ingrese una carpeta válida')
    if not img_name_list: raise ValueError('La carpeta no contiene nombres de archivo admitidos')
    #Se reescriben las imágenes para eliminar la información contenida en los archivos exif
    for img_idx, img_name in enumerate(img_name_list):
        img = cv2.imread(img_name)
        cv2.imwrite(img_destiny_list[img_idx], img)

#Función para convertir un video en cuadros individuales
def video2frame(origin_folder, destiny_folder):
    #Se carga la lista de nombres de archivo admitidos
    if os.listdir(origin_folder): 
        video_name_list = [origin_folder + '/' + s for s in os.listdir(origin_folder) if s.endswith('.mpg') or s.endswith('.mp4') or s.endswith('.avi')]
        os.makedirs(destiny_folder, exist_ok = True)
    else: raise ValueError('Ingrese una carpeta válida')
    if not video_name_list: raise ValueError('La carpeta no contiene nombres de archivo admitidos')
    
    #Se lee cada video y se construyen los cuadros a partir de él
    for video_idx, video_name in enumerate(video_name_list):
        capture = cv2.VideoCapture(video_name)
        os.makedirs(destiny_folder + '/' + os.listdir(origin_folder)[video_idx], exist_ok = True )
        success,image = capture.read()
        count = 0
        while success:
            cv2.imwrite(destiny_folder + '/' + os.listdir(origin_folder)[video_idx]  + '/' + str(count) + '.jpg', image)     # save frame as JPEG file      
            success,image = capture.read()
            #print('Read a new frame: ', success)
            count += 1