from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model_dict = pickle.load(open('./model/model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create label dictionary (A-Z)
labels_dict = {i: chr(65 + i) for i in range(5)}

# Initialize webcam
cap = cv2.VideoCapture(0)

def classify_frame(frame):
    """Process frame, extract hand landmarks, and make predictions."""
    data_aux = []
    x_ = []
    y_ = []

    # Convert to RGB (required by MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
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

        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            return labels_dict[int(prediction[0])]

    return "No Hand"

def generate_frames():
    """Continuously capture frames and send them to the frontend."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            prediction = classify_frame(frame)

            # Display prediction on the frame
            cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.3, (0, 255, 0), 3, cv2.LINE_AA)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Return the webcam video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    """Return the latest ASL sign prediction."""
    success, frame = cap.read()
    if success:
        prediction = classify_frame(frame)
        return jsonify({'prediction': prediction})
    return jsonify({'prediction': 'No frame'})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)

