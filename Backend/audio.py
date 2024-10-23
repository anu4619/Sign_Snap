import cv2
import numpy as np
from keras.models import load_model
import pyttsx3  # Importing the text-to-speech library
import threading
import time

# Load the pre-trained model
model = load_model('traffic_classifier.h5')

# Initialize the TTS engine
engine = pyttsx3.init()

# Dictionary to label all traffic sign classes
classes = {
    1: 'Speed limit (20km/h)', 2: 'Speed limit (30km/h)', 3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)', 5: 'Speed limit (70km/h)', 6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)', 8: 'Speed limit (100km/h)', 9: 'Speed limit (120km/h)',
    10: 'No passing', 11: 'No passing veh over 3.5 tons', 12: 'Right-of-way at intersection',
    13: 'Priority road', 14: 'Yield', 15: 'Stop', 16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited', 18: 'No entry', 19: 'General caution',
    20: 'Dangerous curve left', 21: 'Dangerous curve right', 22: 'Double curve',
    23: 'Bumpy road', 24: 'Slippery road', 25: 'Road narrows on the right',
    26: 'Road work', 27: 'Traffic signals', 28: 'Pedestrians', 29: 'Children crossing',
    30: 'Bicycles crossing', 31: 'Beware of ice/snow', 32: 'Wild animals crossing',
    33: 'End speed + passing limits', 34: 'Turn right ahead', 35: 'Turn left ahead',
    36: 'Ahead only', 37: 'Go straight or right', 38: 'Go straight or left',
    39: 'Keep right', 40: 'Keep left', 41: 'Roundabout mandatory',
    42: 'End of no passing', 43: 'End no passing veh > 3.5 tons'
}

# Track the last detected sign, its detection time, and a 3-second display period
last_detected_sign = None
last_detection_time = 0
display_time_limit = 3  # seconds to persist the text
cooldown_period = 5  # seconds between new sign announcements
confidence_threshold = 0.75  # 75% confidence required
prediction_buffer = []  # Buffer to store predictions for smoothing
buffer_size = 5  # Number of frames to average over

def preprocess_image(img):
    img = cv2.resize(img, (30, 30))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img

def speak(sign_name):
    # Use TTS in a separate thread
    engine.say(sign_name)
    engine.runAndWait()

def update_heading_with_smoothing(sign_name, confidence):
    global last_detected_sign, last_detection_time, prediction_buffer

    # Add current prediction to the buffer
    prediction_buffer.append((sign_name, confidence))
    if len(prediction_buffer) > buffer_size:
        prediction_buffer.pop(0)  # Keep buffer size constant

    # Check if the buffer contains consistent predictions
    predicted_signs = [p[0] for p in prediction_buffer]
    most_common_sign = max(set(predicted_signs), key=predicted_signs.count)

    # Update the heading only if the most common prediction is consistent
    if predicted_signs.count(most_common_sign) > (buffer_size // 2):
        # Check cooldown period and if the sign has changed
        current_time = time.time()
        if most_common_sign != last_detected_sign or (current_time - last_detection_time) > cooldown_period:
            # Update the last detected sign and time
            last_detected_sign = most_common_sign
            last_detection_time = current_time

            # Announce the traffic sign using TTS (separate thread)
            threading.Thread(target=speak, args=(most_common_sign,)).start()

            return most_common_sign
    return None

def detect_sign(frame):
    global last_detected_sign, last_detection_time

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust this value based on your needs
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]

            # Preprocess the ROI
            preprocessed = preprocess_image(roi)

            # Make prediction
            prediction = model.predict(preprocessed)
            sign_class = np.argmax(prediction) + 1
            confidence = np.max(prediction)

            if confidence < confidence_threshold:
                continue  # Ignore weak predictions

            sign_name = classes[sign_class]

            # Use smoothing and update heading
            final_sign = update_heading_with_smoothing(sign_name, confidence)

            if final_sign:
                # Update detection time when a valid sign is detected
                last_detection_time = time.time()

    return frame

def display_persistent_sign(frame):
    global last_detected_sign, last_detection_time

    # If the last detected sign exists, check if it is within the display time limit
    current_time = time.time()
    if last_detected_sign and (current_time - last_detection_time) < display_time_limit:
        # Display the persistent sign
        cv2.putText(frame, last_detected_sign, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Main function to run the program
def main():
    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and recognize traffic signs
        frame = detect_sign(frame)

        # Display the detected sign for 3 seconds (persistence logic)
        frame = display_persistent_sign(frame)

        # Display the result
        cv2.imshow('Traffic Sign Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()