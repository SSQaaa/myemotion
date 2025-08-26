import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
               max_num_hands=2,  # 识别手的最大数量
               min_detection_confidence=0.5,  # 识别置信度阈值
               min_tracking_confidence=0.5)  # 追踪置信度阈值，越低越灵敏，但也更容易误识别，越高的话重复检测的概率越大，越慢
mpDraw = mp.solutions.drawing_utils
handlmsStyles = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
handconStyles = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)

pTime = 0
cTime = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    # print(results.multi_hand_landmarks)
    # if results.multi_hand_landmarks:
        # for handLms in results.multi_hand_landmarks:
        #     for id, lm in enumerate(handLms.landmark):
        #         h, w, c = frame.shape
        #         cx, cy = int(lm.x * w), int(lm.y * h)
        #         print(id, cx, cy)
        #         if id == 8:  # Index finger tip
        #             cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        #     mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handlmsStyles, handconStyles)

    tip_ids = [4, 8, 12, 16, 20] # 分别是大拇指，食指，中指，无名指，小拇指的指尖id
    if results.multi_hand_landmarks:
        fingers = []
        # 大拇指
        if results.multi_hand_landmarks[0].landmark[tip_ids[0]].x > results.multi_hand_landmarks[0].landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
        # 其他四个手指
        for id in range(1, 5):  # 1-4分别代表食指到小拇指,不包括大拇指
            if results.multi_hand_landmarks[0].landmark[tip_ids[id]].y < results.multi_hand_landmarks[0].landmark[tip_ids[id] - 2].y:
                fingers.append(1)  # 手指张开
            else:
                fingers.append(0)  # 手指关闭
        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        cv2.putText(frame, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handlmsStyles, handconStyles)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    # Display the frame
    cv2.imshow('MediaPipe Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break