# Emotion_Detection
Emotion_Detection.ipynb - This .ipynb notebook gives the csv file of the training data from here. Initially there were only folders of images , now in the "train.csv" file, there are paths to all the images and corresponding emotions.We also did the image resizing in this notebook

new_try.ipynb - This .ipynb notebook takes in the csv file ("train.csv") and reads through all the path and loads the images and extracts the features from the images (img -> matrix conversion).
This also gives the model ("your_model.h5") which is trained by the images using CNN architrecture after splitting the csv into train_df and test_df
Finally it also gives the accuracy of the predicted and test_df. The predictions along with the labels are stored as a csv file named "test_with_predictions"

new_try_cv.py - This python file contains the opencv code to detect the live emotions shown by the person before the camera. This detects the face and captures the live image through video camera, and sends to the model which is already trained ("your_model.h5"). The model thus will detect the emotion and display it live.
