# Weed detection method
The main code used by the method described in the paper is uploaded in a separate folder, which contains both the libraries used to train the models and perform segmentation over full images. This folder contains also contains a simple example of use of the method over a folder of images (in this case the one in the database dubbed "Full Images") which can be used to assess the method in question. The code was tested using Numpy 1.19.5, TensorFlow-GPU 1.14.0 , Keras 2.2.3 and OpenCV 3.4.7.28.

# Weed detection database

This repo contains the database that was used for training both the region detection model (the CSN architecture) and the sub-frame detection model (the CNN architecture). The database is structured as follows:
1. First, images are divided between full pictures (from which examples come from) and the examples themselves
2. In the examples folders the pictures are then arranged into Region and Sub-Frame folders
3. In these folders examples are classified into Structured and Unstructured weeds, as well as into a Grass Subfolder
4. Finally, for each of the latter class-folders, sets of train and test images (which were generated randomly via a python code) are presented as were used during training, validation and testing stages

Additionally, outputs of the proposed method over selected testing images are available in the Segmented images folder.
