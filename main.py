import cv2
import mediapipe as mp
import pyautogui
import math
import time


FACE_MODEL = "face_landmarker.task"
HAND_MODEL = "hand_landmarker.task"


# ==========================================
# MediaPipe Setup
# ==========================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


# ==========================================
# Face Landmarker
# ==========================================

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(
    face_options
)


# ==========================================
# Hand Landmarker
# ==========================================

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

hand_landmarker = HandLandmarker.create_from_options(
    hand_options
)


# ==========================================
# Camera
# ==========================================

camera = cv2.VideoCapture(0)

timestamp = 0


# ==========================================
# Screen
# ==========================================

screen_width, screen_height = pyautogui.size()


# ==========================================
# MODE SYSTEM
# ==========================================

modes = [
    "MOUSE",
    "KEYBOARD",
    "MEDIA"
]

current_mode_index = 0

current_mode = modes[current_mode_index]


# ==========================================
# Gesture Stabilization
# ==========================================

previous_gesture = "NONE"

stable_gesture = "NONE"

gesture_frames = 0

REQUIRED_FRAMES = 8


# ==========================================
# Mode Lock
# ==========================================

mode_locked = False


# ==========================================
# Action State
# ==========================================

last_action_time = 0

ACTION_COOLDOWN = 0.8

current_action = "Waiting..."


# ==========================================
# Mouse State
# ==========================================

previous_mouse_x = screen_width / 2
previous_mouse_y = screen_height / 2

smooth_factor = 0.25

pinching = False
dragging = False

pinch_start_time = 0

PINCH_THRESHOLD = 0.045
DRAG_DELAY = 0.35


# ==========================================
# Face
# ==========================================

def get_head_direction(face):

    nose = face[1]

    left_face = face[234]
    right_face = face[454]

    center = (
        left_face.x +
        right_face.x
    ) / 2

    difference = nose.x - center

    if difference < -0.035:
        return "LEFT"

    elif difference > 0.035:
        return "RIGHT"

    return "CENTER"


# ==========================================
# Distance
# ==========================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ==========================================
# Fingers
# ==========================================

def fingers_up(hand, handedness):

    fingers = []

    # Thumb

    if handedness == "Right":

        if hand[4].x < hand[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    else:

        if hand[4].x > hand[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Index

    fingers.append(
        1 if hand[8].y < hand[6].y
        else 0
    )

    # Middle

    fingers.append(
        1 if hand[12].y < hand[10].y
        else 0
    )

    # Ring

    fingers.append(
        1 if hand[16].y < hand[14].y
        else 0
    )

    # Pinky

    fingers.append(
        1 if hand[20].y < hand[18].y
        else 0
    )

    return fingers


# ==========================================
# Gesture Detection
# ==========================================

def detect_gesture(hand, handedness):

    fingers = fingers_up(
        hand,
        handedness
    )

    thumb = fingers[0]
    index = fingers[1]
    middle = fingers[2]
    ring = fingers[3]
    pinky = fingers[4]

    # FIST

    if [
        thumb,
        index,
        middle,
        ring,
        pinky
    ] == [0, 0, 0, 0, 0]:

        return "FIST"

    # THUMBS UP

    if [
        thumb,
        index,
        middle,
        ring,
        pinky
    ] == [1, 0, 0, 0, 0]:

        return "THUMBS UP"

    # ONE

    if [
        thumb,
        index,
        middle,
        ring,
        pinky
    ] == [0, 1, 0, 0, 0]:

        return "ONE"

    # PEACE

    if [
        index,
        middle,
        ring,
        pinky
    ] == [1, 1, 0, 0]:

        return "PEACE"

    # THREE

    if [
        index,
        middle,
        ring,
        pinky
    ] == [1, 1, 1, 0]:

        return "THREE"

    # OPEN PALM

    if [
        thumb,
        index,
        middle,
        ring,
        pinky
    ] == [1, 1, 1, 1, 1]:

        return "OPEN PALM"

    return "UNKNOWN"


# ==========================================
# Stable Gesture
# ==========================================

def update_stable_gesture(raw_gesture):

    global previous_gesture
    global stable_gesture
    global gesture_frames

    if raw_gesture == previous_gesture:

        gesture_frames += 1

    else:

        previous_gesture = raw_gesture
        gesture_frames = 1

    if gesture_frames >= REQUIRED_FRAMES:

        stable_gesture = raw_gesture

    return stable_gesture


# ==========================================
# Change Mode
# ==========================================

def change_mode():

    global current_mode_index
    global current_mode
    global current_action

    current_mode_index += 1

    if current_mode_index >= len(modes):

        current_mode_index = 0

    current_mode = modes[
        current_mode_index
    ]

    current_action = (
        f"MODE → {current_mode}"
    )

    print(
        f"MODE CHANGED → {current_mode}"
    )


# ==========================================
# Mode Controller
# ==========================================

def handle_mode_control(gesture):

    global mode_locked

    if gesture == "ONE":

        if not mode_locked:

            change_mode()

            mode_locked = True

        return

    if gesture != "ONE":

        mode_locked = False


# ==========================================
# Mouse Controller
# ==========================================

def mouse_controller(hand):

    global previous_mouse_x
    global previous_mouse_y

    global pinching
    global dragging

    global pinch_start_time
    global current_action

    index_tip = hand[8]
    thumb_tip = hand[4]

    target_x = (
        index_tip.x *
        screen_width
    )

    target_y = (
        index_tip.y *
        screen_height
    )

    mouse_x = (
        previous_mouse_x +
        (
            target_x -
            previous_mouse_x
        ) *
        smooth_factor
    )

    mouse_y = (
        previous_mouse_y +
        (
            target_y -
            previous_mouse_y
        ) *
        smooth_factor
    )

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    is_pinching = (
        pinch_distance <
        PINCH_THRESHOLD
    )

    if not is_pinching:

        pyautogui.moveTo(
            mouse_x,
            mouse_y,
            duration=0
        )

        previous_mouse_x = mouse_x
        previous_mouse_y = mouse_y

    if (
        is_pinching
        and not pinching
    ):

        pinching = True

        pinch_start_time = time.time()

    if (
        is_pinching
        and pinching
    ):

        hold_time = (
            time.time() -
            pinch_start_time
        )

        if (
            hold_time >= DRAG_DELAY
            and not dragging
        ):

            dragging = True

            pyautogui.mouseDown()

            current_action = (
                "MOUSE → DRAGGING"
            )

    if (
        is_pinching
        and dragging
    ):

        pyautogui.moveTo(
            mouse_x,
            mouse_y,
            duration=0
        )

    if (
        not is_pinching
        and pinching
    ):

        pinching = False

        if not dragging:

            pyautogui.click()

            current_action = (
                "MOUSE → LEFT CLICK"
            )

        else:

            pyautogui.mouseUp()

            dragging = False

            current_action = (
                "MOUSE → DROP"
            )

    return (
        is_pinching,
        pinch_distance
    )


# ==========================================
# Keyboard Controller
# ==========================================

def keyboard_controller(
    gesture,
    head
):

    global last_action_time
    global current_action

    now = time.time()

    if (
        now - last_action_time
        < ACTION_COOLDOWN
    ):

        return

    if gesture == "THUMBS UP":

        pyautogui.press("enter")

        current_action = (
            "KEYBOARD → ENTER"
        )

        last_action_time = now

    elif gesture == "PEACE":

        pyautogui.press("space")

        current_action = (
            "KEYBOARD → SPACE"
        )

        last_action_time = now

    elif gesture == "FIST":

        pyautogui.press("backspace")

        current_action = (
            "KEYBOARD → BACKSPACE"
        )

        last_action_time = now

    elif (
        gesture == "THREE"
        and head == "LEFT"
    ):

        pyautogui.press("left")

        current_action = (
            "KEYBOARD → LEFT"
        )

        last_action_time = now

    elif (
        gesture == "THREE"
        and head == "RIGHT"
    ):

        pyautogui.press("right")

        current_action = (
            "KEYBOARD → RIGHT"
        )

        last_action_time = now


# ==========================================
# MEDIA CONTROLLER
# ==========================================

def media_controller(gesture):

    global last_action_time
    global current_action

    now = time.time()

    if (
        now - last_action_time
        < ACTION_COOLDOWN
    ):

        return

    # ======================================
    # THUMBS UP
    # ======================================

    if gesture == "THUMBS UP":

        pyautogui.press(
            "playpause"
        )

        current_action = (
            "MEDIA → PLAY / PAUSE"
        )

        last_action_time = now

        print(
            "MEDIA → PLAY / PAUSE"
        )


    # ======================================
    # PEACE
    # ======================================

    elif gesture == "PEACE":

        pyautogui.press(
            "nexttrack"
        )

        current_action = (
            "MEDIA → NEXT TRACK"
        )

        last_action_time = now

        print(
            "MEDIA → NEXT TRACK"
        )


    # ======================================
    # FIST
    # ======================================

    elif gesture == "FIST":

        pyautogui.press(
            "prevtrack"
        )

        current_action = (
            "MEDIA → PREVIOUS TRACK"
        )

        last_action_time = now

        print(
            "MEDIA → PREVIOUS TRACK"
        )


    # ======================================
    # OPEN PALM
    # ======================================

    elif gesture == "OPEN PALM":

        pyautogui.press(
            "volumemute"
        )

        current_action = (
            "MEDIA → MUTE / UNMUTE"
        )

        last_action_time = now

        print(
            "MEDIA → MUTE / UNMUTE"
        )


# ==========================================
# Main Loop
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "Camera error"
        )

        break

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    # ======================================
    # RGB
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    # ======================================
    # Face
    # ======================================

    face_result = (
        face_landmarker.detect_for_video(
            mp_image,
            timestamp
        )
    )

    # ======================================
    # Hand
    # ======================================

    hand_result = (
        hand_landmarker.detect_for_video(
            mp_image,
            timestamp
        )
    )

    # ======================================
    # Defaults
    # ======================================

    head = "NONE"

    raw_gesture = "NONE"

    stable_gesture = "NONE"

    is_pinching = False

    pinch_distance = 0

    # ======================================
    # Face Processing
    # ======================================

    if face_result.face_landmarks:

        face = (
            face_result.face_landmarks[0]
        )

        head = get_head_direction(
            face
        )

        for landmark in face:

            x = int(
                landmark.x *
                width
            )

            y = int(
                landmark.y *
                height
            )

            cv2.circle(
                frame,
                (x, y),
                1,
                (255, 0, 255),
                -1
            )

    # ======================================
    # Hand Processing
    # ======================================

    if hand_result.hand_landmarks:

        hand = (
            hand_result.hand_landmarks[0]
        )

        handedness = (
            hand_result.handedness[0][0]
            .category_name
        )

        raw_gesture = detect_gesture(
            hand,
            handedness
        )

        stable_gesture = (
            update_stable_gesture(
                raw_gesture
            )
        )

        # Draw hand

        for landmark in hand:

            x = int(
                landmark.x *
                width
            )

            y = int(
                landmark.y *
                height
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # ==================================
        # Mode
        # ==================================

        handle_mode_control(
            stable_gesture
        )

        # ==================================
        # MOUSE
        # ==================================

        if current_mode == "MOUSE":

            (
                is_pinching,
                pinch_distance
            ) = mouse_controller(
                hand
            )

        # ==================================
        # KEYBOARD
        # ==================================

        elif current_mode == "KEYBOARD":

            if dragging:

                pyautogui.mouseUp()

                dragging = False

            keyboard_controller(
                stable_gesture,
                head
            )

        # ==================================
        # MEDIA
        # ==================================

        elif current_mode == "MEDIA":

            if dragging:

                pyautogui.mouseUp()

                dragging = False

            media_controller(
                stable_gesture
            )

    else:

        # ==================================
        # No hand
        # ==================================

        raw_gesture = "NONE"

        stable_gesture = "NONE"

        previous_gesture = "NONE"

        gesture_frames = 0

        mode_locked = False

        pinching = False

        if dragging:

            pyautogui.mouseUp()

            dragging = False

    # ======================================
    # UI
    # ======================================

    cv2.putText(
        frame,
        f"MODE: {current_mode}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Raw: {raw_gesture}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Stable: {stable_gesture}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {gesture_frames}/{REQUIRED_FRAMES}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Head: {head}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2
    )

    if current_mode == "MOUSE":

        if dragging:

            status = "DRAGGING"

        elif is_pinching:

            status = "PINCH"

        else:

            status = "MOUSE ACTIVE"

    elif current_mode == "KEYBOARD":

        status = "KEYBOARD ACTIVE"

    else:

        status = "MEDIA ACTIVE"

    cv2.putText(
        frame,
        status,
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Action: {current_action}",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # ======================================
    # Show
    # ======================================

    cv2.imshow(
        "AI Multi Mode Controller",
        frame
    )

    # ======================================
    # Q = Quit
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# Safety Cleanup
# ==========================================

if dragging:

    pyautogui.mouseUp()

face_landmarker.close()

hand_landmarker.close()

camera.release()

cv2.destroyAllWindows()