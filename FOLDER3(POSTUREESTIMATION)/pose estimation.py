import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point 'b' formed by the lines ab and cb.

    Parameters:
        a, b, c: Each is a tuple of (x, y) coordinates.

    Returns:
        Angle in degrees.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    ba = a - b
    bc = c - b

    # Calculate the cosine of the angle using the dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip for numerical stability
    return np.degrees(angle)

# Function to classify posture based on angles
def classify_posture(landmarks, image_width, image_height):
    """
    Classifies the posture as 'Good' or 'Bad' based on shoulder and hip angles.

    Parameters:
        landmarks: List of detected pose landmarks.
        image_width: Width of the image/frame.
        image_height: Height of the image/frame.

    Returns:
        Tuple containing the classification string and a dictionary of angles.
    """
    # Extract necessary landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

    # Convert normalized coordinates to pixel values
    def to_pixel(landmark):
        return (int(landmark.x * image_width), int(landmark.y * image_height))

    left_shoulder_px = to_pixel(left_shoulder)
    right_shoulder_px = to_pixel(right_shoulder)
    left_hip_px = to_pixel(left_hip)
    right_hip_px = to_pixel(right_hip)
    left_knee_px = to_pixel(left_knee)
    right_knee_px = to_pixel(right_knee)

    # Calculate angles
    shoulder_angle = calculate_angle(left_shoulder_px, left_hip_px, right_shoulder_px)
    hip_angle = calculate_angle(left_hip_px, left_shoulder_px, right_hip_px)
    knee_angle = calculate_angle(left_knee_px, left_hip_px, right_knee_px)

    angles = {
        'Shoulder Angle': shoulder_angle,
        'Hip Angle': hip_angle,
        'Knee Angle': knee_angle
    }

    # Define thresholds for good posture
    # These thresholds can be adjusted based on empirical data
    if (160 < shoulder_angle < 200) and (160 < hip_angle < 200) and (160 < knee_angle < 200):
        posture = "Good Posture"
    else:
        posture = "Bad Posture"

    return posture, angles

# Function to draw angles on the image
def draw_angles(image, angles):
    """
    Displays the calculated angles on the image.

    Parameters:
        image: The image/frame.
        angles: Dictionary containing angle measurements.
    """
    y0, dy = 30, 30
    for i, (key, value) in enumerate(angles.items()):
        y = y0 + i * dy
        cv2.putText(image, f'{key}: {int(value)}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Main function to run posture detection
def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize Pose estimator
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        # Initialize a deque to store posture history for smoothing
        posture_history = deque(maxlen=5)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Perform pose detection
            results = pose.process(image)

            # Convert back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image_height, image_width, _ = image.shape

            # Initialize default posture
            posture = "No Person Detected"

            if results.pose_landmarks:
                # Draw pose landmarks on the image
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Classify posture
                posture, angles = classify_posture(results.pose_landmarks.landmark, image_width, image_height)
                draw_angles(image, angles)

                # Append to posture history
                posture_history.append(posture)

                # Determine the most common posture in the history
                if len(posture_history) > 0:
                    current_posture = max(set(posture_history), key=posture_history.count)
                else:
                    current_posture = posture
            else:
                current_posture = "No Person Detected"

            # Display posture classification
            color = (0, 255, 0) if current_posture == "Good Posture" else (0, 0, 255)
            cv2.putText(image, current_posture, (10, image_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Front View Posture Detection', image)

            # Exit on pressing 'Esc'
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()