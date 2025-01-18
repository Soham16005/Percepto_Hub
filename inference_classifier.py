import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # To draw the landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # For default landmark styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a label dictionary (A-Z)
labels_dict = {i: chr(65 + i) for i in range(5)}  # Adjust for 5 labels, or change 5 to 26 if A-Z

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Initialize lists for storing the landmarks data
    data_aux = []
    x_ = []
    y_ = []

    # Capture a frame
    ret, frame = cap.read()

    # Get the height and width of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hand landmarks
    results = hands.process(frame_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        # Ensure that only the first hand is processed
        hand_landmarks = results.multi_hand_landmarks[0]  # Select the first hand detected

        # Draw hand landmarks and connections
        mp_drawing.draw_landmarks(
            frame,  # The frame to draw on
            hand_landmarks,  # The landmarks of the first hand
            mp_hands.HAND_CONNECTIONS,  # Hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),  # Landmark style
            mp_drawing_styles.get_default_hand_connections_style()  # Connection style
        )

        # Collect landmarks coordinates (x, y)
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        # Normalize the coordinates and append to data_aux
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))  # Normalize x-coordinates
            data_aux.append(y - min(y_))  # Normalize y-coordinates

        # Ensure that exactly 42 features (21 landmarks * 2 (x, y)) are passed to the model
        if len(data_aux) == 42:
            # Make prediction using the trained model
            prediction = model.predict([np.asarray(data_aux)])

            # Get the predicted character
            predicted_character = labels_dict[int(prediction[0])]

            # Get bounding box for the hand landmarks
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw bounding box and the predicted character on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame with landmarks and prediction
    cv2.imshow('frame', frame)

    # Press 'q' to quit the inference
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
