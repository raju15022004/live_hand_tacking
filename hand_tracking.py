import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Deque to store hand movement points
pts = deque(maxlen=64)

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    image = cv2.flip(image, 1)

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results_hand = hands.process(image_rgb)

    # Convert back to BGR for display
    image.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                if idx == 8:  # Index Finger
                    pts.appendleft((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Draw hand movement trail
    for i in range(1, len(pts)):
        if pts[i - 1] is not None and pts[i] is not None:
            thick = int(np.sqrt(len(pts) / float(i + 1)) * 4.5)
            cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), thick)

    # Display the frame
    cv2.imshow("Hand Movement Tracking", image)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
