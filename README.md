# Percepto_Hub
**Percepto Hub** is an **American Sign Language (ASL)** **recognition system** that captures, processes, and classifies ASL signs in real-time. The system operates in three main stages:

•	**Data Collection**: The **collect_imgs.py** script captures hand images using **cv2.VideoCapture(0)** for each ASL letter and stores them in a **pickle** dataset folder.

•	**Feature Extraction & Dataset Creation**: The **create_dataset.py** script processes the collected images, extracting **21 hand landmarks** (x, y coordinates) using **MediaPipe Hands**. These 42 normalized features are stored for training.

•	**Training & Real-time Classification**: The **app.py** script trains a **Multilayer Perceptron (MLP) classifier**, achieving **97.8% accuracy**. The model then classifies ASL signs in real time.

The trained model is deployed using **Flask**, with a web interface designed in** HTML** and **CSS**. The web application streams real-time video, processes frames using **classify_frame()**, and predicts ASL signs dynamically. Running on 0.0.0.0:5000, it allows users to interact with the ASL recognition system over a network.

This project seamlessly integrates **Machine Learning** and **Web Development** to create an interactive **real-time ASL recognition web app**.


