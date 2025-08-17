"""
Accessible Gesture Mouse Control
================================

A computer vision-based mouse control system designed for accessibility,
using simple hand gestures to control cursor movement, clicking, and scrolling.

Author: [Your Name]
License: MIT
Version: 1.0.0
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
from collections import deque
import sys
import os


# Configuration class for easy customization
class GestureConfig:
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # Gesture thresholds (adjustable for different users)
    FIST_THRESHOLD = 0.6
    OPEN_HAND_THRESHOLD = 0.8
    FINGER_BEND_THRESHOLD = 0.7

    # Swipe detection parameters
    SWIPE_HISTORY_SIZE = 10
    SWIPE_MIN_DISTANCE = 0.05
    SWIPE_VELOCITY_THRESHOLD = 0.02

    # Timing controls
    GESTURE_HOLD_TIME = 0.3
    CLICK_COOLDOWN = 0.5
    SCROLL_COOLDOWN = 0.1

    # Smoothing
    SMOOTHING_FRAMES = 8




def print_startup_info():
    """Print startup information and instructions"""
    print("=" * 60)
    print("🤚 ACCESSIBLE GESTURE MOUSE CONTROL")
    print("=" * 60)
    print("🎯 Designed for accessibility and ease of use")
    print("\n📋 GESTURE CONTROLS:")
    print("   ✋ Open Hand        → Move cursor")
    print("   ✊ Closed Fist      → Left click")
    print("   👆 Point (1 finger) → Right click")
    print("   ✌️  Two Fingers      → Scroll (swipe up/down)")
    print("\n⌨️  KEYBOARD CONTROLS:")
    print("   SPACE → Toggle mouse control on/off")
    print("   Q     → Quit application")
    print("\n🔧 TIPS:")
    print("   • Keep hand within the green control zone")
    print("   • Hold gestures briefly for activation")
    print("   • Use SPACE to disable if needed")
    print("   • Ensure good lighting for best results")
    print("=" * 60)


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'pyautogui': 'pyautogui',
        'numpy': 'numpy'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   pip install {package}")
        sys.exit(1)


# Initialize configuration
config = GestureConfig()

# Check dependencies and initialize
check_dependencies()
print_startup_info()

# Global state variables
mouse_enabled = True
current_gesture = "none"
gesture_start_time = 0
last_action_time = 0
last_scroll_time = 0

# Disable PyAutoGUI failsafe for smoother operation
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # Use lighter model for better performance
)
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=0  # Use lighter model for better performance
)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()
print(f"🖥️  Screen resolution: {screen_width}x{screen_height}")


# Camera setup with error handling
def initialize_camera():
    """Initialize camera with proper error handling"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        print("💡 Please check:")
        print("   • Camera is connected and not in use by another app")
        print("   • Camera permissions are enabled")
        sys.exit(1)

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    # Verify camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Camera initialized: {actual_width}x{actual_height}")

    return cap


cap = initialize_camera()

# Enhanced smoothing system
cursor_history = deque(maxlen=config.SMOOTHING_FRAMES)
gesture_history = deque(maxlen=5)  # For gesture stability

# Swipe tracking
swipe_history = deque(maxlen=config.SWIPE_HISTORY_SIZE)
two_finger_positions = deque(maxlen=config.SWIPE_HISTORY_SIZE)


def calculate_hand_openness(landmarks):
    """Calculate how open the hand is (0 = closed fist, 1 = fully open)"""

    # Key points for measuring hand openness
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Fingertips and their corresponding base joints
    fingertips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    finger_bases = [
        mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    # Calculate distances from wrist to fingertips vs finger bases
    tip_distances = []
    base_distances = []

    for tip, base in zip(fingertips, finger_bases):
        tip_point = landmarks.landmark[tip]
        base_point = landmarks.landmark[base]

        tip_dist = math.sqrt((tip_point.x - wrist.x) ** 2 + (tip_point.y - wrist.y) ** 2)
        base_dist = math.sqrt((base_point.x - wrist.x) ** 2 + (base_point.y - wrist.y) ** 2)

        tip_distances.append(tip_dist)
        base_distances.append(base_dist)

    # Calculate openness ratio
    avg_tip_distance = np.mean(tip_distances)
    avg_base_distance = np.mean(base_distances)

    if avg_base_distance == 0:
        return 0

    openness = avg_tip_distance / avg_base_distance
    return min(1.0, max(0.0, (openness - 0.8) / 0.4))  # Normalize to 0-1


def detect_two_fingers_extended(landmarks):
    """Check if exactly two fingers (index and middle) are extended"""

    # Get finger landmarks
    fingers = [
        # Thumb (special case - compare x coordinate)
        landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
        # Index finger
        landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[
            mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        # Middle finger
        landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks.landmark[
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        # Ring finger
        landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks.landmark[
            mp_hands.HandLandmark.RING_FINGER_PIP].y,
        # Pinky
        landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    ]

    # Count extended fingers
    extended_count = sum(fingers)

    # Check if index and middle are extended, others are not
    index_extended = fingers[1]  # Index finger
    middle_extended = fingers[2]  # Middle finger
    others_folded = not any([fingers[0], fingers[3], fingers[4]])  # Thumb, ring, pinky folded

    return index_extended and middle_extended and others_folded, extended_count


def get_two_finger_center(landmarks):
    """Get the center point between index and middle fingertips"""
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    center_x = (index_tip.x + middle_tip.x) / 2
    center_y = (index_tip.y + middle_tip.y) / 2

    return center_x, center_y


def detect_swipe_gesture():
    """Detect swipe direction from finger movement history"""
    if len(two_finger_positions) < 5:
        return "none", 0

    # Get recent positions
    recent_positions = list(two_finger_positions)[-5:]

    # Calculate vertical movement
    y_positions = [pos[1] for pos in recent_positions]
    y_start = y_positions[0]
    y_end = y_positions[-1]
    y_movement = y_end - y_start

    # Calculate velocity (movement per frame)
    velocity = abs(y_movement) / len(recent_positions)

    # Check if movement is significant enough
    if abs(y_movement) > config.SWIPE_MIN_DISTANCE and velocity > config.SWIPE_VELOCITY_THRESHOLD:
        if y_movement < -config.SWIPE_MIN_DISTANCE:  # Moving up
            return "swipe_up", velocity
        elif y_movement > config.SWIPE_MIN_DISTANCE:  # Moving down
            return "swipe_down", velocity

    return "none", 0


def detect_simple_gesture(landmarks):
    """Detect simple gestures suitable for accessibility"""

    # Check for two-finger gesture first
    two_fingers, finger_count = detect_two_fingers_extended(landmarks)

    if two_fingers:
        return "two_fingers"

    openness = calculate_hand_openness(landmarks)

    # Gesture detection based on hand openness
    if openness < 0.3:
        return "fist"  # Closed fist - for clicking
    elif openness > 0.7:
        return "open"  # Open palm - for moving cursor
    elif 0.3 <= openness <= 0.5:
        return "point"  # Pointing gesture - for right click
    else:
        return "neutral"


def smooth_coordinates(x, y):
    """Apply enhanced smoothing to cursor coordinates"""
    cursor_history.append((x, y))

    if len(cursor_history) < 2:
        return x, y

    # Weighted average with more weight on recent positions
    weights = np.linspace(0.5, 1.0, len(cursor_history))
    weights = weights / np.sum(weights)

    smooth_x = int(np.average([pos[0] for pos in cursor_history], weights=weights))
    smooth_y = int(np.average([pos[1] for pos in cursor_history], weights=weights))

    return smooth_x, smooth_y


def map_to_screen(x, y):
    """Map hand position to entire screen coordinates"""
    # Clamp values to [0,1] range
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    # Add dead zone in center for stability
    dead_zone = 0.05
    if abs(x - 0.5) < dead_zone:
        x = 0.5
    if abs(y - 0.5) < dead_zone:
        y = 0.5

    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)

    return screen_x, screen_y


def draw_ui(frame, gesture, openness, mouse_pos, swipe_info=None):
    """Draw user interface with accessibility features"""
    global mouse_enabled

    height, width = frame.shape[:2]

    # Draw full-screen border indicator
    border_color = (0, 255, 0) if mouse_enabled else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color, 4)

    # Status text at top
    status_y = 30
    cv2.putText(frame, f"Mouse: {'ON' if mouse_enabled else 'OFF'}",
                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if mouse_enabled else (0, 0, 255), 2)

    # Gesture display
    gesture_display = gesture.upper()
    gesture_color = (255, 255, 0)

    if gesture == "two_fingers":
        gesture_display = "TWO FINGERS (READY TO SCROLL)"
        gesture_color = (0, 255, 255)
    elif gesture == "open":
        gesture_display = "OPEN HAND (MOVING)"
        gesture_color = (0, 255, 0)
    elif gesture == "fist":
        gesture_display = "FIST (CLICKING)"
        gesture_color = (255, 0, 0)
    elif gesture == "point":
        gesture_display = "POINTING (RIGHT CLICK)"
        gesture_color = (0, 165, 255)

    cv2.putText(frame, f"Gesture: {gesture_display}",
                (10, status_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)

    # Swipe information
    if swipe_info:
        swipe_direction, velocity = swipe_info
        if swipe_direction != "none":
            swipe_text = f"SCROLLING {swipe_direction.split('_')[1].upper()}"
            cv2.putText(frame, swipe_text, (10, status_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Cursor position indicator
    if mouse_pos:
        screen_x, screen_y = mouse_pos
        cv2.putText(frame, f"Cursor: ({screen_x}, {screen_y})",
                    (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Bottom instructions
    instructions = [
        "CONTROLS: SPACE=toggle mouse  Q=quit",
        "GESTURES: ✋=Move  ✊=Click  👆=RightClick  ✌️=SwipeScroll"
    ]

    for i, instruction in enumerate(instructions):
        y_pos = height - 30 - (len(instructions) - i - 1) * 30
        cv2.putText(frame, instruction, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


def draw_landmarks_simple(frame, landmarks):
    """Draw simplified hand landmarks"""

    height, width = frame.shape[:2]

    # Only draw key points for clarity
    key_points = [
        mp_hands.HandLandmark.WRIST,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    colors = {
        mp_hands.HandLandmark.WRIST: (255, 0, 255),
        mp_hands.HandLandmark.INDEX_FINGER_TIP: (0, 255, 0),
        mp_hands.HandLandmark.THUMB_TIP: (0, 0, 255),
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP: (255, 0, 0),
        mp_hands.HandLandmark.RING_FINGER_TIP: (0, 255, 255),
        mp_hands.HandLandmark.PINKY_TIP: (255, 255, 0)
    }

    for point in key_points:
        landmark = landmarks.landmark[point]
        x, y = int(landmark.x * width), int(landmark.y * height)
        color = colors.get(point, (255, 255, 255))
        cv2.circle(frame, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)


# Main execution loop
def main():
    """Main application loop"""
    global current_gesture, gesture_start_time, last_action_time, last_scroll_time, mouse_enabled

    try:
        print("\n🚀 Starting gesture control... Press SPACE to toggle, Q to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_time = time.time()
            mouse_pos = None
            swipe_info = None

            if results.multi_hand_landmarks and mouse_enabled:
                hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand only

                # Draw simplified landmarks
                draw_landmarks_simple(frame, hand_landmarks)

                # Get hand center (wrist position)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # Detect gesture
                openness = calculate_hand_openness(hand_landmarks)
                gesture = detect_simple_gesture(hand_landmarks)

                # Stabilize gesture detection
                gesture_history.append(gesture)
                if len(gesture_history) >= 3:
                    # Use most common gesture in recent history
                    gesture_counts = {g: gesture_history.count(g) for g in gesture_history}
                    stable_gesture = max(gesture_counts, key=gesture_counts.get)
                else:
                    stable_gesture = gesture

                # Handle two-finger swipe gestures
                if stable_gesture == "two_fingers":
                    # Track two-finger center position
                    center_x, center_y = get_two_finger_center(hand_landmarks)
                    two_finger_positions.append((center_x, center_y))

                    # Detect swipe
                    swipe_direction, velocity = detect_swipe_gesture()
                    swipe_info = (swipe_direction, velocity)

                    if swipe_direction != "none" and current_time - last_scroll_time > config.SCROLL_COOLDOWN:
                        # Perform scrolling based on swipe direction and velocity
                        scroll_amount = int(3 + velocity * 100)  # Base scroll + velocity-based boost

                        if swipe_direction == "swipe_up":
                            pyautogui.scroll(scroll_amount)
                            print(f"📜 Scroll Up (amount: {scroll_amount})")
                        elif swipe_direction == "swipe_down":
                            pyautogui.scroll(-scroll_amount)
                            print(f"📜 Scroll Down (amount: {scroll_amount})")

                        last_scroll_time = current_time

                    # Draw two-finger indicator
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    index_pos = (int(index_tip.x * width), int(index_tip.y * height))
                    middle_pos = (int(middle_tip.x * width), int(middle_tip.y * height))

                    # Draw line between fingers
                    cv2.line(frame, index_pos, middle_pos, (0, 255, 255), 3)
                    cv2.circle(frame, index_pos, 12, (0, 255, 0), -1)
                    cv2.circle(frame, middle_pos, 12, (0, 255, 0), -1)

                # Handle other gestures
                elif stable_gesture == "open":
                    # Move cursor smoothly
                    screen_x, screen_y = map_to_screen(wrist.x, wrist.y)
                    smooth_x, smooth_y = smooth_coordinates(screen_x, screen_y)
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
                    mouse_pos = (smooth_x, smooth_y)

                elif stable_gesture == "fist":
                    # Left click with hold prevention
                    if current_gesture != "fist":
                        gesture_start_time = current_time
                        current_gesture = "fist"
                    elif current_time - gesture_start_time > config.GESTURE_HOLD_TIME:
                        if current_time - last_action_time > config.CLICK_COOLDOWN:
                            pyautogui.click(button='left')
                            last_action_time = current_time
                            print("🖱️  Left Click")

                elif stable_gesture == "point":
                    # Right click
                    if current_gesture != "point":
                        gesture_start_time = current_time
                        current_gesture = "point"
                    elif current_time - gesture_start_time > config.GESTURE_HOLD_TIME:
                        if current_time - last_action_time > config.CLICK_COOLDOWN:
                            pyautogui.click(button='right')
                            last_action_time = current_time
                            print("🖱️  Right Click")
                else:
                    current_gesture = "none"
                    # Clear two-finger history when not in two-finger mode
                    if stable_gesture != "two_fingers":
                        two_finger_positions.clear()

                # Draw UI with current info
                draw_ui(frame, stable_gesture, openness, mouse_pos, swipe_info)

            else:
                # No hand detected or mouse disabled - clear tracking data
                two_finger_positions.clear()
                draw_ui(frame, "none", 0, None)

            # Show frame
            cv2.imshow('Accessible Gesture Mouse', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord(' '):  # Space to toggle mouse
                mouse_enabled = not mouse_enabled
                status = "enabled" if mouse_enabled else "disabled"
                print(f"🖱️  Mouse {status}")
                time.sleep(0.2)  # Prevent rapid toggling

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        print("🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Gesture Mouse Control stopped.")


if __name__ == "__main__":
    main()