import cv2
import mediapipe as mp

# initialise
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# start web camera
cap = cv2.VideoCapture(0)  # edit for other cameras

# initialise detector
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Cannot access camera")
            break

        # frame colours
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        # draw landmarks
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # display image
        cv2.imshow("Pose detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

# end
cap.release()
cv2.destroyAllWindows()