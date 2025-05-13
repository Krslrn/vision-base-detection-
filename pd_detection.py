import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VisionBasedCalculator:
    """Calculate the vertical line-of-sight angle and viewing distance based on user's eye position."""

    def __init__(self):
        # Mediapipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.75
        )
        # Colors for feedback visualization
        self.colors = {"ok": (0, 255, 0), "adjust": (0, 0, 255)}

    @staticmethod
    def draw_feedback(image, distance, angle, status):
        """Display feedback for distance and angle at the top-left corner."""
        lines = [
            f"Distance: {int(distance)} cm - {'OK' if status['distance'] == 'OK' else 'Adjust'}",
            f"Angle: {int(angle)}° - {'OK' if status['angle'] == 'OK' else 'Adjust'}",
            f"Feedback: {status['feedback']}",
        ]
        y_offset = 20
        for i, line in enumerate(lines):
            color = (0, 255, 0) if "OK" in line else (0, 0, 255)
            cv2.putText(image, line, (10, y_offset + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def apply_correction(self, distance):
        """Apply a correction factor to reduce the error."""
        correction_factor = 74 / 66  # Based on observed error
        return distance * correction_factor

    def calculate_angle_and_distance(self, pixels, distances):
        """Calculate line-of-sight angle and distance based on detected facial landmarks."""
        # Fit a second-degree polynomial to map pixels to distance
        coefficients = np.polyfit(pixels, distances, 2)

        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame to mirror the image
            frame = cv2.flip(frame, 1)  # Flip horizontally for left-mirrored effect

            # Process the frame using Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            # Process only the first detection
            if results.detections:
                detection = results.detections[0]  # Only process the first detection
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape

                # Calculate bounding box dimensions
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                left_eye = detection.location_data.relative_keypoints[0]
                right_eye = detection.location_data.relative_keypoints[1]
                eye_center = (
                    int((left_eye.x + right_eye.x) / 2 * iw),
                    int((left_eye.y + right_eye.y) / 2 * ih),
                )
                nose = detection.location_data.relative_keypoints[2]
                nose_position = (int(nose.x * iw), int(nose.y * ih))

                # Calculate pixel distance between eyes
                eye_distance = np.sqrt(
                    (left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2
                ) * iw

                # Map pixel distance to real-world distance
                a, b, c = coefficients
                real_distance = a * eye_distance ** 2 + b * eye_distance + c

                # Apply correction to reduce error
                corrected_distance = self.apply_correction(real_distance)

                # Calculate the vertical line-of-sight angle with dynamic baseline
                dy = nose_position[1] - eye_center[1]  # Vertical offset
                baseline = eye_distance  # Use eye distance as the baseline
                vertical_angle = math.degrees(math.atan2(dy, baseline))  # Dynamic baseline

                # Determine feedback status
                status = {
                    "distance": "OK" if 40 <= corrected_distance <= 74 else "Adjust",
                    "angle": "OK" if 15 <= abs(vertical_angle) <= 30 else "Adjust",
                    "feedback": "OK" if 40 <= corrected_distance <= 74 and 15 <= abs(vertical_angle) <= 30 else "Adjust Position",
                }

                # Draw feedback and bounding box
                bbox_color = self.colors["ok"] if status["feedback"] == "OK" else self.colors["adjust"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
                self.draw_feedback(frame, corrected_distance, abs(vertical_angle), status)

            # Show the frame
            cv2.imshow("Line-of-Sight Detection", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load calibration data from CSV file
    csv_file = "enhanced_distance_xy.csv"
    if os.path.exists(csv_file):
        calibration_data = pd.read_csv(csv_file)
        pixels = calibration_data["distance_pixel"].to_numpy()
        distances = calibration_data["distance_cm"].to_numpy()

        # Create and run the line-of-sight calculator
        calculator = VisionBasedCalculator()
        calculator.calculate_angle_and_distance(pixels, distances)
    else:
        print(f"Calibration file '{csv_file}' not found. Ensure it exists in the working directory.")