import cv2
import mediapipe as mp
import pyautogui
import math
import time


HAND_MODEL = "hand_landmarker.task"


# ==========================================
# MediaPipe
# ==========================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
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
# Modes
# ==========================================

current_mode = "MOUSE"


# ==========================================
# Security
# ==========================================

system_armed = False

last_security_action = 0

SECURITY_COOLDOWN = 1.2


# ==========================================
# Gesture History
# ==========================================

gesture_history = {
    "Left": [],
    "Right": []
}

HISTORY_SIZE = 5


# ==========================================
# Mode Lock
# ==========================================

mode_change_locked = False

last_mode_change = 0

MODE_COOLDOWN = 1.0


# ==========================================
# Action Cooldown
# ==========================================

last_action_time = 0

ACTION_COOLDOWN = 0.8


# ==========================================
# Mouse
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
# Status
# ==========================================

current_action = "SYSTEM LOCKED"


# ==========================================
# Fingers
# ==========================================

def fingers_up(hand, handedness):

    fingers = []


    # Thumb

    if handedness == "Right":

        fingers.append(
            1 if hand[4].x < hand[3].x
            else 0
        )

    else:

        fingers.append(
            1 if hand[4].x > hand[3].x
            else 0
        )


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
# Gesture Recognition
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


    raised = (
        thumb +
        index +
        middle +
        ring +
        pinky
    )


    # FIST

    if raised == 0:

        return "FIST"


    # OPEN PALM

    if raised == 5:

        return "OPEN PALM"


    # ONE

    if (
        thumb == 0
        and index == 1
        and middle == 0
        and ring == 0
        and pinky == 0
    ):

        return "ONE"


    # PEACE

    if (
        index == 1
        and middle == 1
        and ring == 0
        and pinky == 0
    ):

        return "PEACE"


    # THREE

    if (
        index == 1
        and middle == 1
        and ring == 1
        and pinky == 0
    ):

        return "THREE"


    # THUMBS UP

    if (
        thumb == 1
        and index == 0
        and middle == 0
        and ring == 0
        and pinky == 0
    ):

        if hand[4].y < hand[3].y:

            return "THUMBS UP"


    return "UNKNOWN"


# ==========================================
# Stable Gesture
# ==========================================

def get_stable_gesture(
    gesture,
    handedness
):

    history = gesture_history[
        handedness
    ]


    history.append(
        gesture
    )


    if len(history) > HISTORY_SIZE:

        history.pop(0)


    if len(history) < HISTORY_SIZE:

        return "NONE"


    counts = {}

    for item in history:

        counts[item] = (
            counts.get(item, 0) + 1
        )


    stable = max(
        counts,
        key=counts.get
    )


    confidence = (
        counts[stable] /
        len(history)
    )


    if confidence >= 0.8:

        return stable


    return "UNKNOWN"


# ==========================================
# Security Controller
# ==========================================

def security_controller(
    left_gesture,
    right_gesture
):

    global system_armed
    global last_security_action
    global current_action


    now = time.time()


    if (
        now - last_security_action
        < SECURITY_COOLDOWN
    ):

        return


    # ======================================
    # ARM
    # ======================================

    if (
        left_gesture == "OPEN PALM"
        and right_gesture == "OPEN PALM"
    ):

        system_armed = True

        current_action = (
            "SYSTEM → ARMED"
        )

        last_security_action = now

        print(
            "SYSTEM ARMED"
        )


    # ======================================
    # LOCK
    # ======================================

    elif (
        left_gesture == "FIST"
        and right_gesture == "FIST"
    ):

        system_armed = False

        current_action = (
            "SYSTEM → LOCKED"
        )

        last_security_action = now

        print(
            "SYSTEM LOCKED"
        )


# ==========================================
# Mode Controller
# ==========================================

def mode_controller(
    left_gesture
):

    global current_mode

    global mode_change_locked

    global last_mode_change

    global current_action


    now = time.time()


    if (
        now - last_mode_change
        < MODE_COOLDOWN
    ):

        return


    if left_gesture == "ONE":

        current_mode = "MOUSE"


    elif left_gesture == "PEACE":

        current_mode = "KEYBOARD"


    elif left_gesture == "THREE":

        current_mode = "MEDIA"


    else:

        return


    last_mode_change = now

    mode_change_locked = True

    current_action = (
        f"MODE → {current_mode}"
    )


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


    pinch_distance = math.sqrt(
        (
            thumb_tip.x -
            index_tip.x
        ) ** 2
        +
        (
            thumb_tip.y -
            index_tip.y
        ) ** 2
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
                "MOUSE → DRAG"
            )


        if dragging:

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


        if dragging:

            pyautogui.mouseUp()

            dragging = False

            current_action = (
                "MOUSE → DROP"
            )

        else:

            pyautogui.click()

            current_action = (
                "MOUSE → CLICK"
            )


# ==========================================
# Keyboard Controller
# ==========================================

def keyboard_controller(
    gesture
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

        pyautogui.press(
            "enter"
        )

        current_action = (
            "KEYBOARD → ENTER"
        )


    elif gesture == "PEACE":

        pyautogui.press(
            "space"
        )

        current_action = (
            "KEYBOARD → SPACE"
        )


    elif gesture == "FIST":

        pyautogui.press(
            "backspace"
        )

        current_action = (
            "KEYBOARD → BACKSPACE"
        )


    else:

        return


    last_action_time = now


# ==========================================
# Media Controller
# ==========================================

def media_controller(
    gesture
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

        pyautogui.press(
            "playpause"
        )

        current_action = (
            "MEDIA → PLAY / PAUSE"
        )


    elif gesture == "PEACE":

        pyautogui.press(
            "nexttrack"
        )

        current_action = (
            "MEDIA → NEXT"
        )


    elif gesture == "FIST":

        pyautogui.press(
            "prevtrack"
        )

        current_action = (
            "MEDIA → PREVIOUS"
        )


    elif gesture == "OPEN PALM":

        pyautogui.press(
            "volumemute"
        )

        current_action = (
            "MEDIA → MUTE"
        )


    else:

        return


    last_action_time = now


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
    # Hand Detection
    # ======================================

    result = (
        hand_landmarker.detect_for_video(
            mp_image,
            timestamp
        )
    )


    left_gesture = "NONE"

    right_gesture = "NONE"

    left_stable = "NONE"

    right_stable = "NONE"

    left_hand = None

    right_hand = None


    hand_count = 0


    # ======================================
    # Process Hands
    # ======================================

    if result.hand_landmarks:

        hand_count = len(
            result.hand_landmarks
        )


        for hand_index, hand in enumerate(
            result.hand_landmarks
        ):

            handedness = (
                result.handedness[
                    hand_index
                ][0].category_name
            )


            gesture = detect_gesture(
                hand,
                handedness
            )


            stable = get_stable_gesture(
                gesture,
                handedness
            )


            if handedness == "Left":

                left_gesture = gesture

                left_stable = stable

                left_hand = hand


            else:

                right_gesture = gesture

                right_stable = stable

                right_hand = hand


            # Draw landmarks

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


    # ======================================
    # TWO-HAND SECURITY
    # ======================================

    if (
        left_stable != "NONE"
        and right_stable != "NONE"
    ):

        security_controller(
            left_stable,
            right_stable
        )


    # ======================================
    # MODE SELECTION
    # ======================================

    if system_armed:

        if left_stable in [
            "ONE",
            "PEACE",
            "THREE"
        ]:

            if not mode_change_locked:

                mode_controller(
                    left_stable
                )

        else:

            mode_change_locked = False


    # ======================================
    # RIGHT HAND ACTION
    # ======================================

    if (
        system_armed
        and right_hand is not None
    ):

        # Mouse

        if current_mode == "MOUSE":

            mouse_controller(
                right_hand
            )


        # Keyboard

        elif current_mode == "KEYBOARD":

            if dragging:

                pyautogui.mouseUp()

                dragging = False


            keyboard_controller(
                right_stable
            )


        # Media

        elif current_mode == "MEDIA":

            if dragging:

                pyautogui.mouseUp()

                dragging = False


            media_controller(
                right_stable
            )


    else:

        # Safety

        if dragging:

            pyautogui.mouseUp()

            dragging = False


        pinching = False


    # ======================================
    # UI
    # ======================================

    if system_armed:

        security_text = "🔓 ARMED"

    else:

        security_text = "🔒 LOCKED"


    cv2.putText(
        frame,
        security_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"HANDS: {hand_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"LEFT: {left_gesture}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RIGHT: {right_gesture}",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"MODE: {current_mode}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"ACTION: {current_action}",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        "AI Secure Two Hand Controller",
        frame
    )


    # ======================================
    # Q
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# Cleanup
# ==========================================

if dragging:

    pyautogui.mouseUp()


hand_landmarker.close()

camera.release()

cv2.destroyAllWindows()