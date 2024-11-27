import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Function to calculate gaze direction
def detect_gaze_direction(landmarks, frame):
    height, width, _ = frame.shape

    # Get eye and iris landmarks (MediaPipe indices)
    left_eye_indices = [33, 133]  # Left corner and right corner of the left eye
    right_eye_indices = [362, 263]  # Left corner and right corner of the right eye
    left_iris_indices = [468]  # Center of the left iris
    right_iris_indices = [473]  # Center of the right iris

    # Extract coordinates for left eye
    left_eye_left = landmarks[left_eye_indices[0]]
    left_eye_right = landmarks[left_eye_indices[1]]
    left_iris = landmarks[left_iris_indices[0]]

    left_eye_left_coords = (int(left_eye_left.x * width), int(left_eye_left.y * height))
    left_eye_right_coords = (int(left_eye_right.x * width), int(left_eye_right.y * height))
    left_iris_coords = (int(left_iris.x * width), int(left_iris.y * height))

    # Extract coordinates for right eye
    right_eye_left = landmarks[right_eye_indices[0]]
    right_eye_right = landmarks[right_eye_indices[1]]
    right_iris = landmarks[right_iris_indices[0]]

    right_eye_left_coords = (int(right_eye_left.x * width), int(right_eye_left.y * height))
    right_eye_right_coords = (int(right_eye_right.x * width), int(right_eye_right.y * height))
    right_iris_coords = (int(right_iris.x * width), int(right_iris.y * height))

    # Draw debug points
    cv2.circle(frame, left_eye_left_coords, 5, (255, 0, 0), -1)  # Blue
    cv2.circle(frame, left_eye_right_coords, 5, (0, 255, 0), -1)  # Green
    cv2.circle(frame, left_iris_coords, 5, (0, 0, 255), -1)  # Red

    cv2.circle(frame, right_eye_left_coords, 5, (255, 0, 0), -1)  # Blue
    cv2.circle(frame, right_eye_right_coords, 5, (0, 255, 0), -1)  # Green
    cv2.circle(frame, right_iris_coords, 5, (0, 0, 255), -1)  # Red

    # Calculate gaze ratio for left eye
    left_eye_width = max(1, left_eye_right_coords[0] - left_eye_left_coords[0])
    left_gaze_ratio = (left_iris_coords[0] - left_eye_left_coords[0]) / left_eye_width

    # Calculate gaze ratio for right eye
    right_eye_width = max(1, right_eye_right_coords[0] - right_eye_left_coords[0])
    right_gaze_ratio = (right_iris_coords[0] - right_eye_left_coords[0]) / right_eye_width

    # Combine gaze ratios (average for both eyes)
    avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2

    # Debugging: Show gaze ratios
    cv2.putText(frame, f"L-Ratio: {left_gaze_ratio:.2f}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"R-Ratio: {right_gaze_ratio:.2f}", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Determine gaze direction
    if avg_gaze_ratio < 0.4:  # Adjust threshold for "Looking Left"
        return "Looking Left"
    elif avg_gaze_ratio > 0.6:  # Adjust threshold for "Looking Right"
        return "Looking Right"
    else:
        return "Looking Forward"

# Initialize staring time variables
start_time = None
staring_time = 0

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

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Detect gaze direction
            gaze_direction = detect_gaze_direction(face_landmarks.landmark, frame)

            # Track staring time if the user is "Looking Forward"
            if gaze_direction == "Looking Forward":
                if start_time is None:  # Start timing
                    start_time = time.time()
                staring_time = time.time() - start_time
            else:
                start_time = None  # Reset if not looking forward

            # Display the gaze direction and staring time on the frame
            cv2.putText(frame, gaze_direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Staring Time: {staring_time:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Gaze Estimation', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()