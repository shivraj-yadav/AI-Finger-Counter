import cv2
import mediapipe as mp
import time
import math

class FingerCounter:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

    def calculate_angle(self, a, b, c):
        """Calculate angle between 3 points in degrees."""
        ab = [b.x - a.x, b.y - a.y]
        cb = [b.x - c.x, b.y - c.y]
        dot = ab[0]*cb[0] + ab[1]*cb[1]
        mag_ab = math.hypot(ab[0], ab[1])
        mag_cb = math.hypot(cb[0], cb[1])
        angle_rad = math.acos(dot / (mag_ab * mag_cb + 1e-6))
        return math.degrees(angle_rad)

    def count_fingers(self, hand_landmarks):
        fingers = []

        # Thumb using angle between CMC, MCP, TIP
        angle = self.calculate_angle(
            hand_landmarks.landmark[1],  # CMC
            hand_landmarks.landmark[2],  # MCP
            hand_landmarks.landmark[4]   # TIP
        )
        if angle > 160:
            fingers.append(1)
        else:
            fingers.append(0)

        # Index, Middle, Ring, Pinky
        for tip_id in [8, 12, 16, 20]:
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[tip_id - 2]

            if tip.y < pip.y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers.count(1)

    def run(self):
        prev_time = 0

        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from camera.")
                break

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            total_fingers = 0

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Count fingers
                    finger_count = self.count_fingers(hand_landmarks)
                    total_fingers += finger_count

                    # Get wrist position
                    wrist = hand_landmarks.landmark[0]
                    h, w, c = img.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                    cv2.putText(
                        img,
                        f"Fingers: {finger_count}",
                        (wrist_x - 30, wrist_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

            # Show FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Total Fingers: {total_fingers}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Finger Counter", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        finger_counter = FingerCounter()
        finger_counter.run()
    except Exception as e:
        print(f"An error occurred: {e}")
