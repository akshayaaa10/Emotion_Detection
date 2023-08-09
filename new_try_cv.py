import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
from sklearn.model_selection import train_test_split
import pandas as pd
# import sklearn
# from new_try import train_labels
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model('your_model.h5')
le=LabelEncoder()
emotions = [ 'Angry', 'Contempt', 'Disgust',  'Fear',  'Happy',  'Neutral',  'Sad', 'Surprised']  # Add all your emotions here
le.fit(emotions)

# Open a handle to the default webcam
video_capture = cv2.VideoCapture(0)



while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()



    # Convert the image from BGR color (which OpenCV uses) to RGB color 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to the size your model expects
    frame_resized = cv2.resize(frame_rgb, (150, 150))  # replace with the input size your model expects

    # Preprocess the image to a 4D tensor
    img_array = img_to_array(frame_resized)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict the emotion of the image
    emotion_prediction = model.predict(img_batch)
    emotion_index = np.argmax(emotion_prediction)

    emotion = le.inverse_transform([emotion_index])

    # Display the resulting frame with predicted emotion
    cv2.putText(frame, "Emotion: " + str(emotion[0]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion recognition', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
