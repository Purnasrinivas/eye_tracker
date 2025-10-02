import cv2
import mediapipe as mp
import numpy as np
import pymunk
import random
import time

# --- Configuration Constants ---
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 24

# Physics Constants
GRAVITY_Y = -100.0  # Bubbles float up (negative gravity)
BUBBLE_RADIUS_MIN = 30
BUBBLE_RADIUS_MAX = 60
BUBBLE_MASS = 1.0
BUBBLE_ELASTICITY = 0.9  # Bouncy bubbles

# Game Spawning
SPAWN_INTERVAL = 1.0  # Seconds between new bubble spawns

# Gesture Logic
POP_DISTANCE_THRESHOLD = 30  # Max distance (pixels) between fingertip and bubble center to count as a pop

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Pymunk Space Setup
space = pymunk.Space()
space.gravity = (0, GRAVITY_Y)

# --- Global Game State ---
bubbles = []
score = 0
last_spawn_time = time.time()

# --- Physics Helpers ---

class Bubble:
    """Manages a single bubble (Pymunk Body + Shape)."""
    def __init__(self, x, y, radius):
        self.radius = radius
        
        # 1. Create Pymunk Body (Physics)
        moment = pymunk.moment_for_circle(BUBBLE_MASS, 0, radius)
        self.body = pymunk.Body(BUBBLE_MASS, moment)
        self.body.position = x, y
        
        # 2. Create Pymunk Shape (Collision)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = BUBBLE_ELASTICITY
        self.shape.collision_type = 1 # Custom collision type for hands/fingers
        self.shape.bubble = self # Reference back to the class instance
        
        # Add shape and body to the physics space
        space.add(self.body, self.shape)

def create_new_bubble(width, height):
    """Spawns a new bubble randomly at the bottom of the screen."""
    x = random.randint(int(width * 0.1), int(width * 0.9))
    y = int(height * 0.9)  # Spawn near the bottom (Pymunk's Y is inverted relative to OpenCV)
    radius = random.randint(BUBBLE_RADIUS_MIN, BUBBLE_RADIUS_MAX)
    
    new_bubble = Bubble(x, y, radius)
    bubbles.append(new_bubble)

def remove_bubble(bubble):
    """Removes bubble from the game and physics space."""
    space.remove(bubble.body, bubble.shape)
    if bubble in bubbles:
        bubbles.remove(bubble)

# --- Gesture Helpers ---

def get_hand_landmarks_2D(results, hand_index=0):
    """Extracts landmark coordinates for a specific hand."""
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) <= hand_index:
        return None
    
    landmarks = results.multi_hand_landmarks[hand_index].landmark
    
    # Scale coordinates from normalized [0, 1] to screen resolution [0, W]
    coords = {}
    for i, landmark in enumerate(landmarks):
        # Stores (x, y) tuple, where y is index 1
        coords[i] = (int(landmark.x * FRAME_WIDTH), int(landmark.y * FRAME_HEIGHT)) 
    return coords

def is_closed_fist(landmarks):
    """
    Checks for the pop gesture: A closed fist with the index finger extended.
    Landmark coordinates are (x, y) tuples, so y is accessed via index [1].
    """
    
    # Check if the tips of the middle, ring, and pinky fingers are below their MCP joints (Y-coord increases downwards)
    is_fist = (
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP][1] and
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.RING_FINGER_MCP][1] and
        landmarks[mp_hands.HandLandmark.PINKY_TIP][1] > landmarks[mp_hands.HandLandmark.PINKY_MCP][1]
    )
    # Check if index finger tip (8) is above its MCP joint (5) 
    is_pop_gesture = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP][1])
    
    # A pop gesture is a closed fist with an extended index finger
    return is_fist and is_pop_gesture

# --- Main Game Loop ---

def run_game():
    global last_spawn_time, score

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize a static shape for the ground (to prevent bubbles from floating away)
    ground_height = 20
    # In Pymunk, Y=0 is the bottom. Our ground is placed at the bottom 
    ground = pymunk.Segment(space.static_body, (0, ground_height), (FRAME_WIDTH, ground_height), ground_height)
    ground.elasticity = 0.8
    space.add(ground)

    # Convert Pymunk coordinates (Y=0 is bottom) to OpenCV coordinates (Y=0 is top)
    def pymunk_to_cv(y_coord):
        return FRAME_HEIGHT - y_coord

    print("--- Bubble Pop Game Running ---")
    print("👊 Pop: Closed fist + index finger near bubble.")
    print("✋ Move: Open palm to push bubbles.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect (easier for user interaction)
        frame = cv2.flip(frame, 1) 
        
        # Convert the BGR frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 1. Physics Step (Use sub-steps for smoother integration with CV)
        dt = 1.0 / FPS
        # Running the simulation 10 times per frame smooths out motion and collisions
        for _ in range(10): 
            space.step(dt / 10.0)

        # 2. Spawn Logic
        if time.time() - last_spawn_time > SPAWN_INTERVAL:
            # Spawn near the top of the Pymunk space (bottom of the OpenCV window)
            create_new_bubble(FRAME_WIDTH, FRAME_HEIGHT) 
            last_spawn_time = time.time()

        # 3. Gesture Processing (for up to 2 hands)
        if results.multi_hand_landmarks:
            for hand_index in range(len(results.multi_hand_landmarks)):
                landmarks = get_hand_landmarks_2D(results, hand_index)
                
                if not landmarks: continue

                # Coordinates are (x, y) tuples from get_hand_landmarks_2D
                index_tip = np.array(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
                palm_center = np.array(landmarks[mp_hands.HandLandmark.WRIST])
                
                # --- Gesture: POP (Closed Fist + Index Finger) ---
                if is_closed_fist(landmarks):
                    cv2.circle(frame, tuple(index_tip), 15, (0, 0, 255), 2) # Red outline for pop gesture
                    
                    for bubble in bubbles[:]: # Iterate over a copy for safe removal
                        bubble_pos_cv = np.array((bubble.body.position.x, pymunk_to_cv(bubble.body.position.y)))
                        
                        # Calculate distance from index tip to bubble center
                        distance = np.linalg.norm(index_tip - bubble_pos_cv)
                        
                        if distance < bubble.radius + POP_DISTANCE_THRESHOLD:
                            # POP SUCCESS!
                            score += 1
                            remove_bubble(bubble)
                            print(f"POP! Score: {score}")

                # --- Gesture: PUSH/MOVE (Open Palm) ---
                else: 
                    cv2.circle(frame, tuple(palm_center), 20, (255, 255, 0), 2) # Yellow outline for push gesture

                    # Apply force to any bubble near the palm center
                    for bubble in bubbles:
                        bubble_pos_cv = np.array((bubble.body.position.x, pymunk_to_cv(bubble.body.position.y)))
                        distance = np.linalg.norm(palm_center - bubble_pos_cv)

                        if distance < bubble.radius + 50:
                            # Calculate force vector (simple push away from palm)
                            direction = bubble_pos_cv - palm_center
                            # Convert to Pymunk coordinate system for force application (Y-axis inverted)
                            force_vector = (direction[0] * 1000, -direction[1] * 1000) 
                            
                            bubble.body.apply_force_at_world_point(force_vector, (bubble.body.position.x, bubble.body.position.y))


        # 4. Rendering and Cleanup
        
        # Draw the ground line (Pymunk Y=ground_height, OpenCV Y=FRAME_HEIGHT - ground_height)
        cv2.line(frame, (0, FRAME_HEIGHT - ground_height), (FRAME_WIDTH, FRAME_HEIGHT - ground_height), (0, 200, 0), 5)

        # Draw Bubbles
        for bubble in bubbles[:]:
            p = bubble.body.position
            x, y = int(p.x), int(pymunk_to_cv(p.y))
            
            # Check if bubble escaped the top (lost point)
            if y < -bubble.radius:
                remove_bubble(bubble)
                continue

            # Draw the bubble
            cv2.circle(frame, (x, y), bubble.radius, (100, 255, 255), 2) # Light blue outline
            cv2.circle(frame, (x, y), bubble.radius - 1, (10, 50, 80), -1) # Dark interior for visibility

        # Display Score and FPS
        cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the result
        cv2.imshow('Gesture Bubble Pop', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    run_game()
