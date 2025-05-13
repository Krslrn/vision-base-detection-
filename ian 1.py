import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import math
import requests
import time  # Added for rate limiting
import serial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Threshold for horizontal adjustment
CENTER_TOLERANCE = 50  # Pixels from center
PC_NOTIFICATION_URL = "http://192.168.1.100:5000/notify"  # Replace with your PC's IP

class VisionBasedCalculator:
    """Calculate the vertical line-of-sight angle and viewing distance based on user's eye position."""

    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.75
        )
        self.colors = {"ok": (0, 255, 0), "adjust": (0, 0, 255)}
        self.last_status = "OK"
        self.last_notification_time = 0  # Track last notification time

    @staticmethod
    def draw_feedback(image, distance, angle, status):
        """Display feedback for distance, angle, and horizontal position."""
        lines = [
            f"Distance: {int(distance)} cm - {'OK' if status['distance'] == 'OK' else 'Adjust'}",
            f"Angle: {int(angle)}° - {'OK' if status['angle'] == 'OK' else 'Adjust'}",
            f"Horizontal Position: {'OK' if status['horizontal'] == 'OK' else 'Adjust'}",
            f"Feedback: {status['feedback']}",
        ]
        y_offset = 20
        for i, line in enumerate(lines):
            color = (0, 255, 0) if "OK" in line else (0, 0, 255)
            cv2.putText(image, line, (10, y_offset + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def apply_correction(self, distance):
        """Apply a correction factor to reduce the error."""
        correction_factor = 74 / 65
        return distance * correction_factor

    def send_pc_notification(self, message):
        """Send a notification to the PC with rate limiting (10 seconds per message)."""
        current_time = time.time()
        if message == "Adjust Position":
            if self.last_status == "OK" or (current_time - self.last_notification_time >= 10):
                try:
                    requests.post(PC_NOTIFICATION_URL, json={"message": message}, timeout=1)
                    print(f"Notification sent: {message}")
                    self.last_notification_time = current_time
                except requests.exceptions.RequestException:
                    print("PC notification server unavailable, continuing detection...")

        self.last_status = message  # Update last status

    def calculate_angle_and_distance(self, pixels, distances):
        """Calculate line-of-sight angle, distance, and horizontal position."""
        coefficients = np.polyfit(pixels, distances, 2)
        cap = cv2.VideoCapture(1)
        screen_center_x = int(cap.get(3) / 2)  # Get screen center x-coordinate
        ser = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust port if needed

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                left_eye = detection.location_data.relative_keypoints[0]
                right_eye = detection.location_data.relative_keypoints[1]
                eye_center = (
                    int((left_eye.x + right_eye.x) / 2 * iw),
                    int((left_eye.y + right_eye.y) / 2 * ih),
                )
                nose = detection.location_data.relative_keypoints[2]
                nose_position = (int(nose.x * iw), int(nose.y * ih))

                eye_distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2) * iw
                a, b, c = coefficients
                real_distance = a * eye_distance ** 2 + b * eye_distance + c
                corrected_distance = self.apply_correction(real_distance)

                dy = nose_position[1] - eye_center[1]
                baseline = eye_distance
                vertical_angle = math.degrees(math.atan2(dy, baseline))

                if corrected_distance < 40:
                    horizontal_status = "OK"
                else:
                    horizontal_offset = nose_position[0] - screen_center_x
                    if abs(horizontal_offset) > CENTER_TOLERANCE:
                        horizontal_status = "Adjust"
                        direction = "Left" if horizontal_offset < 0 else "Right"
                        direction2 = 1 if horizontal_offset < 0 else 2
                    else:
                        horizontal_status = "OK"
                        direction2 = 0

                status = {
                    "distance": "OK" if 40 <= corrected_distance <= 74 else "Adjust",
                    "angle": "OK" if 15 <= abs(vertical_angle) <= 30 else "Adjust",
                    "horizontal": horizontal_status,
                    "feedback": "OK" if 40 <= corrected_distance <= 74 and 15 <= abs(vertical_angle) <= 30 and horizontal_status == "OK" else "Adjust Position",
                }

                # Send notification only for "Adjust Position"
                self.send_pc_notification(status["feedback"])

                bbox_color = self.colors["ok"] if status["feedback"] == "OK" else self.colors["adjust"]
                self.draw_feedback(frame, corrected_distance, abs(vertical_angle), status)

                x_steps = int(corrected_distance * 10)  # Calibration needed
                y_steps = int(vertical_angle * 10)  # Calibration needed
                z_steps = int(direction2)  # Calibration needed and Z axis definition needed.
                print(z_steps)

                data = f"X:{x_steps},Y:{y_steps},Z:{z_steps}\n"
                ser.write(data.encode())

            else:
                print("No user detected.")
                cv2.putText(frame, "No user detected.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)    

            cv2.imshow("Line-of-Sight Detection", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        ser.close()

if __name__ == "__main__":
    csv_file = "enhanced_distance_xy.csv"
    if os.path.exists(csv_file):
        calibration_data = pd.read_csv(csv_file)
        pixels = calibration_data["distance_pixel"].to_numpy()
        distances = calibration_data["distance_cm"].to_numpy()

        calculator = VisionBasedCalculator()
        calculator.calculate_angle_and_distance(pixels, distances)
    else:
        print(f"Calibration file '{csv_file}' not found. Ensure it exists in the working directory.")