import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Load the trained model
model = load_model('/content/drive/MyDrive/Cap100/models/20231102-12231698907993-On-100-videos.h5')

# Define constants
IMG_SIZE = 224
SEQUENCE_LENGTH = 10

# Preprocess a single frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0  # Normalize the frame
    return frame

# Preprocess the video frames
def preprocess_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_frame(frame)
        frames.append(preprocessed_frame)

        if len(frames) == SEQUENCE_LENGTH:
            sequence = np.array(frames)
            sequence = np.expand_dims(sequence, axis=0)
            yield sequence

    cap.release()

# Save uploaded video
def save_uploaded_video(uploaded_file):
    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return video_path

# Make predictions on a video
def make_prediction(video_path):
    # Preprocess video frames
    video_frames = preprocess_video_frames(video_path)

    for frames in video_frames:
        prediction = model.predict(frames)

        if prediction > 0.00001:
            result = "Accident"
        else:
            result = "Non-Accident"

        return prediction, result

# Save uploaded video
def save_uploaded_video(uploaded_file):
    # Ensure the "uploads" directory exists
    os.makedirs("uploads", exist_ok=True)

    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return video_path


# Streamlit app
def main():
    st.title('Accident Detection with Video Input')
    uploaded_file = st.file_uploader('Upload a video:', type=['mp4'])

    if uploaded_file is not None:
        video_path = save_uploaded_video(uploaded_file)
        st.video(video_path, format='video/mp4')

        if st.button('Predict'):
            prediction, result = make_prediction(video_path)
            st.write(f'Prediction: {prediction}')
            st.write(f'Result: {result}')

if __name__ == '__main__':
    main()
