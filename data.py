import os
import cv2
import numpy as np
import mediapipe as mp

# Function to perform detection using MediaPipe Hands
def mediapipe_detection(image, hands):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Check if the image is not None and has a valid shape
    if image is None or len(image.shape) < 3:
        return None, None

    # Convert the image to RGB format if it's in BGR format
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make the detection
    results = hands.process(image)

    # Convert the image back to BGR format if it was originally in BGR format
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks on the image
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
    return keypoints

# Constants
DATA_PATH = 'data'
actions = ['A', 'B', 'C']
no_sequences = 30
sequence_length = 30

# Set up MediaPipe Hands model
mp_hands = mp.solutions.hands

# Create directories to save data
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

# Loop through actions
for action in actions:
    print(f"Starting data collection for action: {action}")
    # Loop through sequences aka videos
    for sequence in range(no_sequences):
        print(f"  Sequence {sequence}/{no_sequences}")
        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            # Read image
            image_path = 'Image/{}/{}.png'.format(action, sequence)
            image = cv2.imread(image_path)

            # Check if the image is loaded properly
            if image is None:
                print(f"Image not found at path: {image_path}")
                continue

            # Perform detection
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                image, results = mediapipe_detection(image, hands)

            # Check if the detection results are valid
            if image is None or results is None:
                print(f"Detection failed for image at path: {image_path}")
                continue

            # Display status message
            if frame_num == 0:
                cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(200)
            else: 
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

            # Export keypoints
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # Break loop on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

# Release resources and close windows
cv2.destroyAllWindows()
