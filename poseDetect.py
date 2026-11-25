import cv2
import mss
import numpy as np
import mediapipe as mp

# initialise
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

MONITOR_INDEX = 1      # 1 = primary monitor, 2 = second monitor, 0 = all monitors combined
REGION = None          # or a region, e.g. {"left": 0, "top": 0, "width": 1280, "height": 720}
SCALE = 0.5            # shrink frames before processing; faster, lower = less accurate for small people
RECORD = False         # True to also save the annotated output to a file
OUTPUT_FILE = "screen_pose.mp4"
FPS = 20

with mss.mss() as sct:
    monitor = REGION or sct.monitors[MONITOR_INDEX]

    writer = None

    # initialise detector
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        cv2.namedWindow("Screen pose detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Screen pose detection", 640, 360)

        while True:
            # grab the screen (BGRA) and convert to BGR
            shot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

            if SCALE != 1.0:
                frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE,
                                   interpolation=cv2.INTER_AREA)

            h, w = frame.shape[:2]

            # frame colours
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # bounding box around the person
                xs = [lm.x * w for lm in results.pose_landmarks.landmark]
                ys = [lm.y * h for lm in results.pose_landmarks.landmark]
                x1, x2 = int(max(min(xs), 0)), int(min(max(xs), w - 1))
                y1, y2 = int(max(min(ys), 0)), int(min(max(ys), h - 1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                status = "Person detected"
            else:
                status = "No person"

            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # optional recording of the annotated frames
            if RECORD:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (w, h))
                writer.write(frame)

            # display image
            cv2.imshow("Screen pose detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
                break

    if writer is not None:
        writer.release()

# end
cv2.destroyAllWindows()