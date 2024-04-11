# import os
# import pickle
#
# import mediapipe as mp
# import cv2
#
# mp_hands = mp.solutions.hands
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# DATA_DIR = './data'
#
# data = []
# labels = []
# for gesture_label, dir_ in enumerate(os.listdir(DATA_DIR)):  # enumerate the directories
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#
#         x_ = []
#         y_ = []
#
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:  # Check if exactly 2 hands are detected
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#
#                     x_.append(x)
#                     y_.append(y)
#
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))
#
#             data.append(data_aux)
#             labels.append(gesture_label)  # Use the gesture_label as the label for each hand
#             print(len(data_aux))
#
# f = open('data3.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()
import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for gesture_label, dir_ in enumerate(os.listdir(DATA_DIR)):  # enumerate the directories
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
            while len(data_aux) < 84:
                data_aux.extend([0, 0])

            data.append(data_aux)
            labels.append(gesture_label)  # Use the gesture_label as the label for each hand
            print(len(data_aux))

f = open('data3.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
