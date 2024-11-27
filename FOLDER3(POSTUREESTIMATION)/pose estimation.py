import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle ABC (in degrees) where A, B, and C are points.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Function to classify posture as Good or Bad
def detect_posture(landmarks, image):
    """
    Detects posture using head, shoulders, and alignment from landmarks.
    """
    height, width, _ = image.shape

    # Get necessary keypoints
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Convert to pixel coordinates
    left_shoulder_coords = (int(left_shoulder.x * width), int(left_shoulder.y * height))
    right_shoulder_coords = (int(right_shoulder.x * width), int(right_shoulder.y * height))
    nose_coords = (int(nose.x * width), int(nose.y * height))

    # Calculate shoulder alignment (vertical difference between shoulders)
    shoulder_slope = abs(left_shoulder_coords[1] - right_shoulder_coords[1])

    # Calculate head angle relative to shoulders
    head_angle = calculate_angle(left_shoulder_coords, nose_coords, right_shoulder_coords)

    # Debug drawing
    cv2.circle(image, left_shoulder_coords, 5, (255, 0, 0), -1)  # Left shoulder
    cv2.circle(image, right_shoulder_coords, 5, (255, 0, 0), -1)  # Right shoulder
    cv2.circle(image, nose_coords, 5, (0, 255, 0), -1)  # Nose
    cv2.line(image, left_shoulder_coords, right_shoulder_coords, (0, 255, 255), 2)  # Shoulder line

    # Adjusted thresholds for better classification
    if shoulder_slope > 30 or head_angle < 150:  # Relax thresholds
        posture = "Bad Posture"
    else:
        posture = "Good Posture"

    # Display debug information for thresholds
    cv2.putText(image, f"Shoulder Slope: {shoulder_slope:.2f}px", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"Head Angle: {head_angle:.2f}°", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return posture

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Detect posture
        posture = detect_posture(results.pose_landmarks.landmark, frame)

        # Display posture classification
        cv2.putText(frame, posture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if posture == "Good Posture" else (0, 0, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Posture Estimation', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()