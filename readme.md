# Project Title: Eye Tracking-Based User Engagement Detection

## Overview

This project focuses on developing a system that analyzes eye movements to detect user engagement with video content. By using machine learning models, the system can classify whether a user is watching or not watching a screen based on real-time video input. The project involves data collection, training a neural network model, and applying this model to classify user engagement from test video data.

## Key Features

- **Data Collection**: Captures and processes frames from video input, detecting eye regions to create labeled datasets for training (watching vs. not watching).
- **Neural Network Model**: A convolutional neural network (CNN) is built using `Keras` and trained on the collected eye images to classify engagement.
- **Real-Time Application**: The trained model is applied to test videos to predict user engagement based on eye movement.
- **Eye Detection**: Utilizes Haar cascades to accurately detect and segment eye regions from video frames for further processing.

## Technologies Used

- **Python**: The main programming language for the entire pipeline, including data collection, model training, and prediction.
- **OpenCV**: For video capture, image processing, and eye detection using Haar cascades.
- **Keras with TensorFlow Backend**: To build, train, and deploy the neural network for classification tasks.
- **NumPy**: For efficient handling of large image datasets and numerical computations.


## How to Use

1. **Install Required Libraries**:
   Install the necessary Python libraries by running:
   ```
   pip install opencv-python keras tensorflow numpy
   ```

2. **Prepare Video Data**:
   Place the video files for training and testing in the `vids` folder. Ensure that Haar cascades for left and right eye detection (`haarcascade_lefteye_2splits.xml` and `haarcascade_righteye_2splits.xml`) are placed in the `res` directory.

3. **Run the Code**:
   Execute the `main.py` file to start the data collection, model training, and testing process:
   ```bash
   python main.py
   ```

4. **Data Collection**:
   The script will process the input video frame by frame, saving eye images to the `watching_data` and `not_watching_data` directories for training the model.

5. **Model Training**:
   The neural network is trained using the collected eye images, and the trained model is saved as `model.h5` for future use.

6. **Real-Time Prediction**:
   After training, the model is applied to test videos to classify user engagement based on eye movements.
