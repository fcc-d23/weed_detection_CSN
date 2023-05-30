# Weed detection database

This repo contains the database that was used for training both the region detection model (the CSN architecture) and the sub-frame detection model (the CNN architecture). The database is structured as follows:
1. First, images are divided between full pictures (from which examples come from) and the examples themselves
2. In the examples folders the pictures are then arranged into Region and Sub-Frame folders
3. In these folders examples are classified into Structured and Unstructured weeds, as well as into a Grass Subfolder
4. Finally, for each of the latter class-folders, sets of train and test images (which were generated randomly via a python code) are presented as were used during training, validation and testing stages

Additionally, outputs of the proposed method over selected testing images are available in the Segmented folder.