import cv2
import mediapipe as mp
import time
import math
import imutils

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

def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def angle(a, b, c):
    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc) + 1e-6
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))

def fingers(results):
    tip_ids = [4, 8, 12, 16, 20] # 分别是大拇指，食指，中指，无名指，小拇指的指尖id
    all_fingers = []
    totalFingers = 0
    for i in range(len(results.multi_hand_landmarks)):
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # print(handedness.classification[0].label)  # 打印左右手
            # print(handedness.classification[0].score)  # 打印置信度
            handType = handedness.classification[0].label
            # print(handType)
            fingers = []
            basicLength = dist(results.multi_hand_landmarks[i].landmark[0], results.multi_hand_landmarks[i].landmark[9])

            # 大拇指
            if angle(results.multi_hand_landmarks[i].landmark[2], results.multi_hand_landmarks[i].landmark[3], results.multi_hand_landmarks[0].landmark[4]) < 150 or dist(results.multi_hand_landmarks[i].landmark[17], results.multi_hand_landmarks[i].landmark[4]) < dist(results.multi_hand_landmarks[i].landmark[17], results.multi_hand_landmarks[i].landmark[5]):
                fingers.append(0)
            else:
                fingers.append(1)
            # 其他四个手指
            for id in range(1, 5):  # 1-4分别代表食指到小拇指,不包括大拇指
                if angle(results.multi_hand_landmarks[i].landmark[tip_ids[id]-1], results.multi_hand_landmarks[i].landmark[tip_ids[id]-2], results.multi_hand_landmarks[0].landmark[tip_ids[id]-3]) < 90:
                    fingers.append(0) 
                elif basicLength < dist(results.multi_hand_landmarks[i].landmark[tip_ids[id]], results.multi_hand_landmarks[i].landmark[0]) < basicLength * 2.2:
                    fingers.append(1) 
                else:
                    fingers.append(0)
            #print(fingers)
            #totalFingers = fingers.count(1)
            # print(totalFingers)
            #cv2.putText(frame, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        all_fingers.append(fingers)
    #print(all_fingers)
    return all_fingers
        
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 1 表示水平翻转（镜像），0 表示垂直翻转，-1 表示水平垂直翻转
    frame = imutils.resize(frame, width=300)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        if [1, 0, 0, 0, 0] in fingers(results):
            cv2.putText(frame, 'Thumbs_UP', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        elif [0, 1, 1, 0, 0] in fingers(results):
            cv2.putText(frame, 'Victory', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        elif [1, 1, 1, 1, 1] in fingers(results):
            cv2.putText(frame, 'Hi', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        elif [0, 0, 0, 0, 0] in fingers(results):
            cv2.putText(frame, 'Fighting', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handlmsStyles, handconStyles)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    #print(fps)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    # Display the frame
    cv2.imshow('MediaPipe Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break