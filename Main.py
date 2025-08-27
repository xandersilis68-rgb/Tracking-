import cv2
import numpy as np
from collections import deque

# Initialize scores
score_a = 0  # Left side (Team A)
score_b = 0  # Right side (Team B)

# Ball color in HSV (for yellow/orange volleyball; adjust as needed)
lower_ball = np.array([20, 100, 100])
upper_ball = np.array([30, 255, 255])

# Buffer for ball positions (for tracking trail)
pts = deque(maxlen=64)

# Point detection cooldown (frames to wait after scoring to avoid multiple increments)
cooldown = 0
cooldown_max = 30  # Adjust based on FPS

# Previous y-position for velocity check
prev_y = None

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    mid_x = frame_width // 2
    ground_threshold = frame_height - 50  # Adjust based on camera view

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Mask for ball color
    mask = cv2.inRange(hsv, lower_ball, upper_ball)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_center = None

    # Get largest contour (assumed ball)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        if radius > 10:  # Minimum size to be considered ball
            ball_center = (int(x), int(y))
            cv2.circle(frame, ball_center, int(radius), (0, 255, 255), 2)

    # Update points trail
    pts.appendleft(ball_center)

    # Draw trail
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Scoring logic
    if cooldown > 0:
        cooldown -= 1
    elif ball_center:
        current_y = ball_center[1]
        if prev_y is not None and current_y > prev_y and current_y > ground_threshold:
            # Ball is falling and hit ground
            if ball_center[0] < mid_x:
                # Hit on left side: point for Team B
                score_b += 1
                print(f"Point for Team B! Scores: A {score_a} - B {score_b}")
            else:
                # Hit on right side: point for Team A
                score_a += 1
                print(f"Point for Team A! Scores: A {score_a} - B {score_b}")
            cooldown = cooldown_max
            pts.clear()  # Reset trail after point

        prev_y = current_y

    # Display scores and court divider
    cv2.line(frame, (mid_x, 0), (mid_x, frame_height), (255, 0, 0), 2)  # Net/mid line
    cv2.putText(frame, f"Team A: {score_a}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Team B: {score_b}", (frame_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("AI Volleyball Scoreboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
