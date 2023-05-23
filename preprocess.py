# load time: 6 seconds
# processing speed: 11 images / sec

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import glob
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

exercises = {
  0: 'barbell_biceps_curl',
  1: 'bench_press',
  2: 'chest_fly_machine',
  3: 'deadlift',
  4: 'decline_bench_press',
  5: 'hammer_curl',
  6: 'hip_thrust',
  7: 'incline_bench_press',
  8: 'lat_pulldown',
  9: 'lateral_raises',
  10: 'leg_extension',
  11: 'leg_raises',
  12: 'plank',
  13: 'pull_up',
  14: 'romanian_deadlift',
  15: 'russian_twist',
  16: 'shoulder_press',
  17: 'squat',
  18: 't_bar_row',
  19: 'squat',
  20: 'tricep_dips',
  21: 'tricep_pushdown',
}

data_cols = []
row_num = 0
for i in range(33):
  data_cols.append(f'landmark{i}.x')
  data_cols.append(f'landmark{i}.y')
  data_cols.append(f'landmark{i}.z')
  data_cols.append(f'landmark{i}.v')
data_cols.append('exercise')
data = pd.DataFrame(columns=data_cols)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:

  for exercise_num, exercise in exercises.items():
    image_dir = f'./kaggle_workout_dataset/{exercise}/*.jpg'
    image_files = sorted(glob.glob(image_dir))

    for file_num, file in enumerate(image_files):
      print(f'{exercise}: Processing image {file_num} of {len(image_files)}')
      image = cv2.imread(file)
      image_height, image_width, _ = image.shape
      # Convert the BGR image to RGB before processing.
      results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      
      if not results.pose_landmarks:
        continue

      for landmark_num, landmark in enumerate(results.pose_landmarks.landmark):
        data.at[row_num, f'landmark{landmark_num}.x'] = landmark.x
        data.at[row_num, f'landmark{landmark_num}.y'] = landmark.y
        data.at[row_num, f'landmark{landmark_num}.z'] = landmark.z
        data.at[row_num, f'landmark{landmark_num}.v'] = landmark.visibility
      data.at[row_num, 'exercise'] = exercise_num

      row_num += 1

  data.to_pickle('./data.pkl')
