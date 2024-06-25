import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import model_from_json

# Function to perform detection using MediaPipe Hands
def mediapipe_detection(image, hands):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if image is None or len(image.shape) < 3:
        return None, None

    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    return image, results

# Function to extract keypoints from the detection results
def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
    if keypoints:
        return np.array(keypoints).flatten()
    return np.zeros(21 * 3)  # 21 landmarks with x, y, z coordinates

# Load model architecture from JSON file
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()

# Load model weights
model = model_from_json(model_json)
model.load_weights("model.h5")

# Constants
actions = ['A', 'B', 'C']
threshold = 0.8

# Set up MediaPipe Hands model
mp_hands = mp.solutions.hands

# New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []

# Start video capture
cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, hands)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            try:
                input_data = np.expand_dims(sequence, axis=0)
                print(f"Input data shape: {input_data.shape}")
                res = model.predict(input_data)[0]
                print("Model prediction:", res)
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))

                        if len(sentence) > 1:
                            sentence = sentence[-1:]
                            accuracy = accuracy[-1:]
            except Exception as e:
                print(f"Error during prediction: {e}")
            
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: " + ' '.join(sentence) + ' ' + ''.join(accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
