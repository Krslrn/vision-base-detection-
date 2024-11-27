import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DistanceCalculator:
    """Calculate the distance (cm) between user's eyes and laptop screen using webcam"""

    # Colors to use (in BGR)
    colors = [(76, 168, 240), (255, 0, 255), (255, 255, 0)]
    # Instantiate face detection solution
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)

    @staticmethod
    def draw_bbox(img, bbox, color, l=30, t=5, rt=1):
        """Draw bounding box around user's face"""
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, color, rt)
        # Top left
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)
        # Top right
        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)
        # Bottom left
        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)
        # Bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

    @staticmethod
    def draw_dist_between_eyes(img, center_left, center_right, color, distance_value):
        """Draw a line between user's eyes and annotate the distance (pixels) between them"""
        # Mark eyes
        cv2.circle(img, center_left, 1, color, thickness=8)
        cv2.circle(img, center_right, 1, color, thickness=8)

        # Line between eyes
        cv2.line(img, center_left, center_right, color, 3)

        # Add distance value
        cv2.putText(img, f'{int(distance_value)}',
                    (center_left[0], center_left[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)

    def run_config(self):
        """Initial configuration to measure distances in cm for different distances in pixels"""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Mirror the image
            image = cv2.flip(image, 1)

            # Convert to RGB for Mediapipe
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            if results.detections:
                for detection in results.detections:
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                    bbox_list.append(bbox)

                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x * iw), int(left_eye.y * ih)),
                                      (int(right_eye.x * iw), int(right_eye.y * ih))])

            # Convert back to BGR for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for bbox, eye in zip(bbox_list, eyes_list):
                dist_between_eyes = np.sqrt((eye[0][1] - eye[1][1])**2 + (eye[0][0] - eye[1][0])**2)
                DistanceCalculator.draw_bbox(image, bbox, self.colors[0])
                DistanceCalculator.draw_dist_between_eyes(image, eye[0], eye[1], self.colors[0], dist_between_eyes)

            cv2.imshow('webcam', image)
            if cv2.waitKey(5) & 0xFF == ord('k'):
                break
        cap.release()

    def calculate_distance(self, distance_pixel, distance_cm):
        """Calculate distance in cm between user's eyes and laptop screen"""
        coff = np.polyfit(distance_pixel, distance_cm, 2)
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Mirror the image
            image = cv2.flip(image, 1)

            # Process image with Mediapipe
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            if results.detections:
                for detection in results.detections:
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                    bbox_list.append(bbox)

                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x * iw), int(left_eye.y * ih)),
                                      (int(right_eye.x * iw), int(right_eye.y * ih))])

            # Convert back to BGR for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for bbox, eye in zip(bbox_list, eyes_list):
                dist_between_eyes = np.sqrt((eye[0][1] - eye[1][1])**2 + (eye[0][0] - eye[1][0])**2)
                a, b, c = coff
                distance_cm = a * dist_between_eyes**2 + b * dist_between_eyes + c

                if distance_cm > 61:
                    DistanceCalculator.draw_bbox(image, bbox, self.colors[0])
                    cv2.putText(image, f'{int(distance_cm)} cm - too far', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, self.colors[0], 2)
                elif distance_cm > 51:
                    DistanceCalculator.draw_bbox(image, bbox, self.colors[2])
                    cv2.putText(image, f'{int(distance_cm)} cm - safe', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, self.colors[2], 2)
                else:
                    DistanceCalculator.draw_bbox(image, bbox, self.colors[1])
                    cv2.putText(image, f'{int(distance_cm)} cm - too close', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, self.colors[1], 2)

            cv2.imshow('webcam', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()

if __name__ == '__main__':
    distance_df = pd.read_csv('distance_xy.csv')
    eye_screen_distance = DistanceCalculator()
    eye_screen_distance.calculate_distance(distance_df['distance_pixel'], distance_df['distance_cm'])