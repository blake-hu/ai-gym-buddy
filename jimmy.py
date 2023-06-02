import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from constants import exercises
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

model = tf.keras.models.load_model('model')

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    # Tensorflow integration for predicting exercise

    # Read landmarks into numpy array
    if not results.pose_landmarks:
        continue
    landmarks =  np.zeros(shape=(1, 132)).astype('float32')
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        landmarks[0][i * 4 + 0] = landmark.x
        landmarks[0][i * 4 + 1] = landmark.y
        landmarks[0][i * 4 + 2] = landmark.z
        landmarks[0][i * 4 + 3] = landmark.visibility

    print(landmarks)
    
    # Predict exercise using landmarks fed into neural network
    prediction_array = model.predict(tf.convert_to_tensor(landmarks))
    exercise_id = np.argmax(prediction_array)
    exercise = exercises[exercise_id]

    # Render exercise text on frame
    image = cv2.putText(image, exercise, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
            2.5, (240, 240, 240), 4, cv2.LINE_AA)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()