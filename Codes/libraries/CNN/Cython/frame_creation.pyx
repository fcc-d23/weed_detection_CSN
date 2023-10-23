#Se importan las librerías en cython y opencv en python
cimport cython
cimport numpy as cnp
import numpy as np
import cv2
@cython.wraparound(False)
@cython.boundscheck(False)

#Función que genera los cuadros para la segmentación basada en CNN

cpdef slide_window_creator(cnp.ndarray[cnp.uint8_t, ndim = 3] img_rgb, cnp.ndarray[cnp.int16_t, ndim = 1] win_size , cnp.ndarray[cnp.int16_t, ndim = 1] new_win_shape, int sld_step, str data_type):

    #Si se escoge hacer pares de cuadros para CSN
    if data_type == 'CSN':
        
        #Se escala la imagen para que no sobre espacio de la foto al deslizar la ventana
        img_rgb_shape = img_rgb.shape
        img_rgb_new = cv2.resize( img_rgb, ( int( ( img_rgb_shape[1] - win_size[1]) / sld_step )  * sld_step + win_size[1] , int( ( img_rgb_shape[0] - win_size[0]) / sld_step )  * sld_step + win_size[0] ) )
        frames_r = int( ( img_rgb_new.shape[0] - win_size[0]  ) /  sld_step ) + 1
        frames_c = int( ( img_rgb_new.shape[1] - win_size[1]  ) /  sld_step ) + 1
        frame_array_in = np.zeros( (2, (frames_r-1)*(frames_c-1)*2, new_win_shape[0], new_win_shape[1], 3), dtype = np.uint8 )
        frame_coordinates_in = np.zeros( (2, (frames_r-1)*(frames_c-1)*2, 4), dtype = np.int32 )
        new_img = np.zeros(img_rgb_new.shape, dtype = np.uint8)
        frame_array = np.zeros( (1 , 4), dtype = np.int32 )
        frame_array_tot = np.zeros( ( 1, new_win_shape[0], new_win_shape[1], 3), dtype = np.uint8 )

        #Para cada subdivisión se crean los cuadros interiores
        for frame_r in range(frames_r-1):
            for frame_c in range(frames_c-1):
                
                #Se calculan las coordenadas para cada par de imágenes central-derecha y central-abajo
                frame_central_coord = [ frame_r * sld_step, frame_r*sld_step + win_size[0], frame_c*sld_step, frame_c*sld_step + win_size[1] ]
                frame_right_coord = [ frame_r * sld_step, frame_r*sld_step + win_size[0], ( frame_c + 1 )*sld_step, ( frame_c + 1 )*sld_step + win_size[1] ]
                frame_down_coord = [ ( frame_r + 1) * sld_step, ( frame_r + 1 )*sld_step + win_size[0], frame_c*sld_step, frame_c*sld_step + win_size[1] ]
                
                #Se crea un arreglo con las coordenadas de todos los cuadros para la posterior reconstrucción de la imagen, además del arreglo total de cuadros centrales
                frame_array = np.concatenate( (frame_array, np.reshape(frame_central_coord, ( 1, 4) ) ), axis = 0)
                frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) ), ( 1, new_win_shape[0], new_win_shape[1], 3 ) ) ), axis = 0 )
                #frame_array_tot[( frames_c - 1 )*frame_r + frame_c , :, :, :] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                #frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
                
                #OJO con el orden, es primero cuadro-cuadro de la derecha y luego cuadro-cuadro de abajo
                frame_array_in[0,  2*( ( frames_c - 1 )*frame_r + frame_c ), : ,: ,: ] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
                frame_array_in[1,  2*(  ( frames_c - 1 )*frame_r + frame_c ) , : ,: ,: ] = cv2.resize(img_rgb_new[ frame_right_coord[0] : frame_right_coord[1],\
                    frame_right_coord[2] : frame_right_coord[3] , :], (new_win_shape[0], new_win_shape[1]) )
                
                frame_array_in[0, 2*( ( frames_c - 1 )*frame_r + frame_c ) + 1, : ,: ,: ] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :] , (new_win_shape[0], new_win_shape[1]) ) 
                frame_array_in[1, 2*( ( frames_c - 1 )*frame_r + frame_c ) + 1, : ,: ,: ] = cv2.resize( img_rgb_new[ frame_down_coord[0] : frame_down_coord[1],\
                    frame_down_coord[2] : frame_down_coord[3] , :], ( new_win_shape[0], new_win_shape[1] ) )
                
                #Se adjuntan las coordenadas para asociar a predicciones de CSN más adelante
                frame_coordinates_in[0, 2*( ( frames_c - 1)*frame_r + frame_c ), :  ] = frame_central_coord
                frame_coordinates_in[1, 2*( ( frames_c - 1)*frame_r + frame_c ), :  ] = frame_right_coord
                
                frame_coordinates_in[0, 2*( ( frames_c - 1)*frame_r + frame_c ) + 1, : ] = frame_central_coord
                frame_coordinates_in[1, 2*( ( frames_c - 1)*frame_r + frame_c ) + 1, : ] = frame_down_coord   

        #Se rellenan los valores restantes de las coordenadas de los cuadros
        for frame_r in range(0, frames_r):
            
            for frame_c in range(frames_c-1, frames_c):
                
                frame_central_coord = [ frame_r * sld_step, frame_r*sld_step + win_size[0], frame_c*sld_step, frame_c*sld_step + win_size[1] ]
                #frame_array[ ( frames_c - 1)*frame_r + frame_c,:] = frame_central_coord
                frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) ), ( 1, new_win_shape[0], new_win_shape[1], 3 ) ) ), axis = 0 )
                frame_array = np.concatenate( (frame_array, np.reshape(frame_central_coord, ( 1, 4) ) ), axis = 0)
                #frame_array_tot[( frames_c - 1 )*frame_r + frame_c , :, :, :] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                #frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
        
        for frame_r in range(frames_r-1, frames_r):
            
            for frame_c in range(0, frames_c):
                
                frame_central_coord = [ frame_r * sld_step, frame_r*sld_step + win_size[0], frame_c*sld_step, frame_c*sld_step + win_size[1] ]
                frame_array = np.concatenate( (frame_array, np.reshape(frame_central_coord, ( 1, 4) ) ), axis = 0)
                
                frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) ), ( 1, new_win_shape[0], new_win_shape[1], 3 ) ) ), axis = 0 )
                #frame_array_tot[( frames_c - 1 )*frame_r + frame_c , :, :, :] = cv2.resize(img_rgb_new[ frame_central_coord[0] : frame_central_coord[1],\
                #frame_central_coord[2] : frame_central_coord[3], :], (new_win_shape[0], new_win_shape[1]) )
        
        #Se corta el primer cuadro del arreglo de cuadros centrales para eliminar el 0
        frame_array = frame_array[1:-1, :]
        frame_array = np.concatenate( ( frame_array, np.reshape( frame_array[-1, :] , ( 1, 4 ) ) ), axis = 0 )
        
        frame_array_tot = frame_array_tot[1:-1, :, :, :]
        frame_array_tot = np.concatenate( ( frame_array_tot, np.reshape( frame_array_tot[-1, :, :, :] , ( 1, frame_array_tot[-1, :, :, :].shape[0], frame_array_tot[-1, :, :, :].shape[1], frame_array_tot[-1, :, :, :].shape[2] ) ) ), axis = 0 )
       
        return frame_array_in, frame_coordinates_in, img_rgb_new, [frames_r, frames_c], frame_array, frame_array_tot
        