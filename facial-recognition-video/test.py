import numpy as np
import cv2
import time

# creating the videocapture object
# and reading from the input file
# Change it to 0 if reading from webcam

cap = cv2.VideoCapture(0)

# Reading the video file until finished
while (cap.isOpened()):
    start_time = time.time()

    # Capture frame-by-frame

    ret, frame = cap.read()

    # if video finished or no Video Input
    if not ret:
        break

    # Our operations on the frame come here
    gray = frame

    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX

    fps = 1 / (time.time() - start_time)

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # puting the FPS count on the frame
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # displaying the frame with fps
    cv2.imshow('frame', gray)

    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()
