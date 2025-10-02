import numpy as np
import cv2
import time
import os


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


Distraction_FRAME_THRESHOLD = 2 

MAX_Distraction_FRAMES = 100  
  


FACE_CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_FILENAME = 'haarcascade_eye.xml'

try:

    FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, FACE_CASCADE_FILENAME)
    EYE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, EYE_CASCADE_FILENAME)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    
    if face_cascade.empty() or eye_cascade.empty():
        raise FileNotFoundError(f"One or both Haar Cascade files are empty at path: {cv2.data.haarcascades}")

except Exception as e:
    print("FATAL ERROR: Could not load Haar Cascade files.")
    print("Please ensure OpenCV is installed correctly and the cascade files are present.")
    print(f"Error details: {e}")
    exit()



tracking_active = False

track_points = None

old_gray = None

Distraction_counter = 0

Distraction_closed_frames = 0 


lk_params = dict(winSize = (21, 21),   
                 maxLevel = 3,         
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))


TRACK_COLOR = (0, 255, 0) 
DETECTION_COLOR = (255, 165, 0) 
CIRCLE_RADIUS = 7         


def draw_status_message(frame, message, color=(255, 255, 255)):
    """Draws a status message at the bottom center of the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(
        message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    x = (frame.shape[1] - text_width) // 2
    y = frame.shape[0] - 20 
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def detect_and_initialize_eyes(frame_gray, frame_color):
    """
    Detects faces, then eyes, and returns the coordinates of two eyes if found.
    Returns (True, two_eye_points) if successful, otherwise (False, None).
    """
    global face_cascade, eye_cascade


    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=7, minSize=(100, 100))
    
    if len(faces) == 0:
        return False, None

    
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]


    cv2.rectangle(frame_color, (x, y), (x + w, y + h), DETECTION_COLOR, 2)

  
    roi_gray = frame_gray[y:y + h//2, x:x + w] 
    
   
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)) 
    
    
    if len(eyes) == 2:
        eye_points = []
        for (ex, ey, ew, eh) in eyes:
            # Calculate the center of the eye and convert coordinates back to frame size
            center_x = x + ex + ew // 2
            center_y = y + ey + eh // 2
            
            eye_points.append([center_x, center_y])
            

            cv2.rectangle(frame_color, (x + ex, y + ey), (x + ex + ew, y + ey + eh), TRACK_COLOR, 1)

        initial_points = np.array(eye_points, dtype=np.float32).reshape(-1, 1, 2)
        return True, initial_points
    
    return False, None



cv2.namedWindow('Lucas-Kanade Robust Eye Tracker')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("FATAL ERROR: Could not open webcam.")
    exit()

# Wait for camera warmup
time.sleep(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    status_color = TRACK_COLOR if tracking_active else DETECTION_COLOR
    
    
    cv2.putText(frame, f"Distractions: {Distraction_counter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    if not tracking_active or track_points is None or len(track_points) < 2:
        # --- Automatic Initialization/Re-detection Phase ---
        
        draw_status_message(frame, "STATUS: Auto-detecting eyes...", status_color)
        
        success, new_points = detect_and_initialize_eyes(frame_gray, frame)
        
        if success:
            # If two eyes are found, tracking starts or is restored
            track_points = new_points
            old_gray = frame_gray.copy()
            
            # Distraction COUNTING LOGIC: Only count if transitioning from a lost state (tracking_active == False)
            if tracking_active == False: 
                
                # Check if the closed duration was within the plausible range for a human Distraction (2-100 frames).
                if Distraction_closed_frames >= Distraction_FRAME_THRESHOLD and Distraction_closed_frames <= MAX_Distraction_FRAMES:
                    Distraction_counter += 1
                    print(f"Distraction detected! Total Distractions: {Distraction_counter} (Closed Duration: {Distraction_closed_frames} frames)")
                
                elif Distraction_closed_frames > 0:
                    # Provide feedback when a lost state recovers but doesn't count as a Distraction
                    print(f"Interruption detected ({Distraction_closed_frames} frames) but NOT counted as Distraction.")
                    if Distraction_closed_frames < Distraction_FRAME_THRESHOLD:
                         print(f"Reason: Duration was too short (less than {Distraction_FRAME_THRESHOLD} frames).")
                    elif Distraction_closed_frames > MAX_Distraction_FRAMES:
                         print(f"Reason: Duration was too long (more than {MAX_Distraction_FRAMES} frames).")
            
            tracking_active = True
            Distraction_closed_frames = 0 # Reset closed frame counter as eyes are open
            
        else:
            # If detection failed (eyes closed/moved). We are accumulating closed frames.
            Distraction_closed_frames += 1
            
    elif tracking_active and track_points is not None and len(track_points) == 2:
        # --- Tracking Logic (Active Phase) ---

        draw_status_message(frame, "STATUS: Tracking Eyes (Green Circles)", status_color)

        # 2. Lucas–Kanade estimates point movement
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, track_points, None, **lk_params
        )

        # Filter out points that were successfully tracked (status == 1)
        good_new = p1[status == 1]

        # 3. Drawing and Maintenance
        if len(good_new) == 2:
            # Tracking successful
            for new_point in good_new:
                x, y = new_point.ravel()
                # Draw a large, filled green circle
                cv2.circle(frame, (int(x), int(y)), CIRCLE_RADIUS, TRACK_COLOR, -1)
            
            # Update the previous frame and previous points for the next loop
            old_gray = frame_gray.copy()
            track_points = good_new.reshape(-1, 1, 2)
            Distraction_closed_frames = 0 # Reset closed frame counter if tracking is successful
            
        else:
            # Tracking lost: This is the trigger to enter the potential Distraction state
            print("TRACKING LOST: Less than 2 points survived. Entering potential Distraction/re-detection phase.")
            
            # Set active to False to transition into the re-detection block next frame
            tracking_active = False
            track_points = None


    # --- Display ---
    cv2.imshow('Lucas-Kanade Robust Eye Tracker', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
