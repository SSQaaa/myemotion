from tensorflow.keras.utils import img_to_array
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time
import serial
import mediapipe as mp
import math
from threading import Thread, Lock
from queue import Queue
from collections import Counter

pTime = 0
cTime = 0
imshow_flag = True

# =================================================== 串口发送设置 =========================================================================
class SerialComm:
    def __init__(self):
        self.ser_32_0 = serial.Serial('/dev/ttyS0', 115200, timeout=0.1) 
        self.ser_51_4 = serial.Serial('/dev/ttyS4', 115200, timeout=0.1)
        self.ser_mipi_3 = serial.Serial('/dev/ttyS3', 115200, timeout=0.1)
        self.lock = Lock()
        self.last_type = 0
        self.mytimeout = 0.5  # 优先级超时时间（秒）
        self.last_time = time.time()  # 上次更新优先级的时间
        time.sleep(0.5)
        
    def send(self, data_byte):
        with self.lock:
            if self.ser_51_4.in_waiting:  # 检查是否有数据
                data = self.ser_51_4.read(self.ser_51_4.in_waiting)  # 读出全部数据
                #print(f"[S4] Received: {data.hex()}")  # 打印十六进制
                self.ser_32_0.write(data)  # 原样转发
                print(f"[S0] Forwarded: {data.hex()}")
            
            else:
                current_time = time.time()
                if current_time - self.last_time > self.mytimeout:
                    self.last_type = 0  # 超时后重置优先级

                if data_byte >= self.last_type:
                    self.last_time = current_time
                    packet_0 = bytes([data_byte, data_byte, data_byte, data_byte, data_byte])
                    packet_4 = str(data_byte)  
                    #packet_4 = data_byte.to_bytes(1, byteorder='big')
                    packet_3 = f"page {str(data_byte)}\r\n"
                    self.ser_32_0.write(packet_0)
                    self.ser_51_4.write(packet_4.encode("utf-8"))
                    #self.ser_51_4.write(packet_4)
                    self.ser_mipi_3.write(packet_3.encode("utf-8"))
                    # print(data_byte)
                # else:
                #     if current_time - self.last_time > self.mytimeout:
                #         self.last_type = data_byte
                #         self.last_time = current_time
                #         packet_4 = bytes([0x55, 0x55, data_byte, 0x55, 0x55])
                #         packet_0 = chr(data_byte + 0x30)   # 0x30 是字符 '0' 的 ASCII 码
                #         packet_3 = f"page {str(data_byte)}\n"
                #         self.ser_32_4.write(packet_4)
                #         self.ser_51_0.write(packet_0.encode("utf-8"))
                #         self.ser_mipi_3.write(packet_3.encode("utf-8"))
                #         print(data_byte)
                       


            time.sleep(0.01)
            self.last_type = data_byte
            
            
    def close(self):
        self.ser_32_0.close()
        self.ser_51_4.close()
        self.ser_mipi_3.close()

# =================================================== 串口发送设置 =========================================================================

# =================================================== 获取画面设置 =========================================================================
class FrameGrabber:
    def __init__(self):
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.thread = Thread(target=self._grab_frames, daemon=True)
        self.thread.start()

    def _grab_frames(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)


    def get_frame(self):
        return self.frame_queue.get()

    def close(self):
        self.running = False
        self.thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("Frame grabber closed")

# =================================================== 获取画面设置 =========================================================================

# =================================================== 手势识别设置 =========================================================================
class GestureDetector:
    def __init__(self, serial_comm):
        self.serial_comm = serial_comm
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.thread = Thread(target=self._detect_loop, daemon=True)
        self.thread.start()
        self.gesture = None 
        self.last_seen_time = 0
        mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,  # 识别手的最大数量
                    min_detection_confidence=0.7,  # 识别置信度阈值
                    min_tracking_confidence=0.7)  # 追踪置信度阈值，越低越灵敏，但也更容易误识别，越高的话重复检测的概率越大，越慢
        self.mpDraw = mp.solutions.drawing_utils
        self.handlmsStyles = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
        self.handconStyles = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
            
        
    def update_frame(self, frame):
        if not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        self.frame_queue.put((frame))

    def dist(self, a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def angle(self, a, b, c):
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.hypot(*ba) * math.hypot(*bc) + 1e-6
        return math.degrees(math.acos(max(-1, min(1, dot / mag))))

    def fingers(self, results):
        tip_ids = [4, 8, 12, 16, 20] # 分别是大拇指，食指，中指，无名指，小拇指的指尖id
        all_fingers = []
        for i in range(len(results.multi_hand_landmarks)):
            for self.handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # print(handedness.classification[0].emotion)  # 打印左右手
                # print(handedness.classification[0].score)  # 打印置信度
                self.handType = handedness.classification[0].label
                # print(handType)
                fingers = []
                basicLength = self.dist(results.multi_hand_landmarks[i].landmark[0], results.multi_hand_landmarks[i].landmark[9])

                # 大拇指
                if self.angle(results.multi_hand_landmarks[i].landmark[2], results.multi_hand_landmarks[i].landmark[3], results.multi_hand_landmarks[0].landmark[4]) < 150 or self.dist(results.multi_hand_landmarks[i].landmark[17], results.multi_hand_landmarks[i].landmark[4]) < self.dist(results.multi_hand_landmarks[i].landmark[17], results.multi_hand_landmarks[i].landmark[5]):
                    fingers.append(0)
                else:
                    fingers.append(1)
                # 其他四个手指
                for id in range(1, 5):  # 1-4分别代表食指到小拇指,不包括大拇指
                    if self.angle(results.multi_hand_landmarks[i].landmark[tip_ids[id]-1], results.multi_hand_landmarks[i].landmark[tip_ids[id]-2], results.multi_hand_landmarks[0].landmark[tip_ids[id]-3]) < 90:
                        fingers.append(0) 
                    elif basicLength < self.dist(results.multi_hand_landmarks[i].landmark[tip_ids[id]], results.multi_hand_landmarks[i].landmark[0]) < basicLength * 2.2:
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
        
    def _detect_loop(self):       # 检测
        # mpDraw = mp.solutions.drawing_utils
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                frame = cv2.flip(frame, 1)  # 1 表示水平翻转（镜像），0 表示垂直翻转，-1 表示水平垂直翻转
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    self.last_seen_time = time.time()

                    #  绘制 21 个关键点和连线
                    for handLms in results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(
                            frame, 
                            handLms, 
                            mp.solutions.hands.HAND_CONNECTIONS,
                            self.handlmsStyles, 
                            self.handconStyles
                        )

                    if [1, 0, 0, 0, 0] in self.fingers(results):
                        # cv2.putText(frameClone, 'Thumbs_UP', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
                        self.gesture = "Thumbs_UP"
                    elif [0, 1, 1, 0, 0] in self.fingers(results):
                        # cv2.putText(frameClone, 'Victory', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
                        self.gesture = "Victory"
                    elif [1, 1, 1, 1, 1] in self.fingers(results):
                        # cv2.putText(frameClone, 'Hi', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
                        self.gesture = "Hi"
                    elif [0, 0, 0, 0, 0] in self.fingers(results):
                        # cv2.putText(frameClone, 'Fighting', (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
                        self.gesture = "Fighting"

                    # for handLms in results.multi_hand_landmarks:
                    #     self.mpDraw.draw_landmarks(frameClone, handLms, self.mpHands.HAND_CONNECTIONS, self.handlmsStyles, self.handconStyles)
                else:
                    # 超过2秒无手势，清除数据
                    if time.time() - getattr(self, 'last_seen_time', 0) > 0.2:
                        self.gesture = None

            except Exception as e:
                # print("[GestureDetector] Error:", e)
                continue


                
    def close(self):
        self.running = False
        self.thread.join()
# =================================================== 手势识别设置 =========================================================================

# =================================================== 情绪识别设置 =========================================================================
class EmotionDetector:
    def __init__(self, serial_comm):
        self.serial_comm = serial_comm
        self.frame_queue = Queue(maxsize=1)
        self.running = True
        self.thread = Thread(target=self._detect_loop, daemon=True)
        self.thread.start()
        self.emotion = None
        self.fX, self.fY, self.fW, self.fH = 0, 0, 0, 0
        self.last_seen_time = 0
         # parameters for loading data and images
        detection_model_path = '/home/orangepi/Desktop/myemotion/Emotion-recognition-master/haarcascade_files/haarcascade_frontalface_default.xml'
        emotion_model_path = '/home/orangepi/Desktop/myemotion/Emotion-recognition-master/models/_mini_XCEPTION.102-0.66.hdf5'

        # loading models
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        assert not self.face_detection.empty(), "Haar 模型加载失败：检查 haarcascade_files 路径"
        self.emotion_classifier = load_model(emotion_model_path, compile=False)

        
    def update_frame(self, frame):
        if not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        self.frame_queue.put((frame))
        
    def _detect_loop(self):     
        while self.running:
            try:
                EMOTIONS = ["angry","disgust","scared","happy","sad","surprised","neutral"]

                # feelings_faces = []
                # for emotion in EMOTIONS:
                #     feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))
                frame = self.frame_queue.get(timeout=0.1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detection.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=6, minSize=(60,60),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                canvas = np.zeros((250, 300, 3), dtype="uint8")
                # frameClone = frame.copy()
                preds = np.zeros(len(EMOTIONS), dtype=np.float32)  # 无人脸时显示全0

                if len(faces) > 0:
                    self.last_seen_time = time.time()
                    # 选面积最大的人脸（w*h）
                    (self.fX, self.fY, self.fW, self.fH) = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                    if self.fW * self.fH < 5000:
                        self.emotion = None
                        continue

                    # 边界检查
                    if self.fW > 0 and self.fH > 0 and self.fY + self.fH <= gray.shape[0] and self.fX+self.fW <= gray.shape[1]:
                        roi = gray[self.fY:self.fY+self.fH, self.fX:self.fX+self.fW]
                        if roi.size > 0:
                            roi = cv2.resize(roi, (64, 64))
                            roi = roi.astype("float") / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)
                            preds = self.emotion_classifier.predict(roi, verbose=0)[0]
                            self.emotion = EMOTIONS[preds.argmax()]
                    else:
                        self.emotion = None
                    # # ROI 预处理与推理
                    # roi = gray[self.fY + self.fH:self.fX + self.fW]
                    # roi = cv2.resize(roi, (64, 64))
                    # roi = roi.astype("float") / 255.0
                    # roi = img_to_array(roi)
                    # roi = np.expand_dims(roi, axis=0)

                    # preds = self.emotion_classifier.predict(roi, verbose=0)[0]
                    # self.emotion = EMOTIONS[preds.argmax()]

                    # 画框与标签
                    # cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                    # cv2.putText(frameClone, emotion, (fX, fY - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                else:
                    # 超过2秒无人脸，清除数据
                    if time.time() - getattr(self, 'last_seen_time', 0) > 0.2:
                        self.emotion = None
                        self.fX, self.fY, self.fW, self.fH = 0, 0, 0, 0
            except Exception as e:
                # print("[EmotionDetector] Error:", e)
                continue


                
    def close(self):
        self.running = False
        self.thread.join()
# =================================================== 情绪识别设置 =========================================================================

def find_max(lst):
    """返回出现次数最多的一个元素和计数"""
    if not lst:
        return None, 0
    counter = Counter(lst)
    most_common = counter.most_common(1)  # 返回 [(element, count)]
    if most_common:
        return most_common[0]  # 返回 (element, count)
    else:
        return None, 0

data_arr = []
frame_count = 0
timers = {
    "send": 0,
    "neutral": time.time(),
    "emotion_none": time.time(),
    "same_type": time.time()
}
last_type = 0
current_time = 0
noface_flag = 0
# last_send_time = 0
# send_cooldown = 5.0  # 单位：秒
# last_send_netrual_time = 0
# last_send_emotion_time = 0

if __name__ == '__main__':
    serial_comm = SerialComm()
    frame_grabber = FrameGrabber()
    gesture_detector = GestureDetector(serial_comm)
    emotion_detector = EmotionDetector(serial_comm)
    while True:
        cTime = time.time()
        current_time = time.time()
        fps = 1 / (cTime - pTime + 1e-6)
        frame = frame_grabber.get_frame()
        frame_for_gesture = frame.copy()
        frame_for_emotion = frame.copy()
        gesture_detector.update_frame(frame_for_gesture)
        emotion_detector.update_frame(frame_for_emotion)
        frameClone = frame.copy()


        gesture= gesture_detector.gesture  
        fX, fY, fW, fH = emotion_detector.fX, emotion_detector.fY, emotion_detector.fW, emotion_detector.fH
        emotion = emotion_detector.emotion      

        
        # -------------------- 串口发送 --------------------
        if gesture == "Thumbs_UP":
            data_byte = 0x04
        elif gesture == "Victory":        
            data_byte = 0x03
        elif gesture == "Hi":
            data_byte = 0x01
        elif gesture == "Fighting":
            data_byte = 0x02
        elif emotion == "happy":
            data_byte = 0x05

        # neutral 情绪30秒冷却
        if emotion == "neutral":
            if current_time - timers["neutral"] >= 30.0:
                data_byte = 0x06
                timers["neutral"] = current_time
            else:
                data_byte = 0x00

        # 无情绪120秒冷却
        if emotion is None:
            if current_time - timers["emotion_none"] >= 120.0:
                data_byte = 0x07
                serial_comm.send(data_byte)
                print(data_byte)
                timers["emotion_none"] = current_time
                noface_flag = 1

        if gesture is not None or emotion is not None:
            data_arr.append(data_byte)
            frame_count += 1
            print(data_byte, gesture, emotion)
            cv2.putText(frameClone, gesture, (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
            #gesture_detector.gesture = None  # 发送后清除，避免重复发送
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
            cv2.putText(frameClone, emotion, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            #emotion_detector.emotion = None  # 发送后清除，避免重复发送  
    
        if frame_count == 30:
            current_time = time.time()
            if current_time - timers["send"] >= 5.0:
                (data_final, cnt) = find_max(data_arr)
                if cnt >= 15 and noface_flag == 1 and emotion is not None:
                    noface_flag = 0
                    data_final = 0x08
                    serial_comm.send(data_final)
                    print(data_final)
                if cnt >= 15 and data_final != last_type:   # 至少15帧检测到且不等于上次发送，没有检测到就不发送
                  last_type = data_final
                  if data_final != 0x00:
                    serial_comm.send(data_final)
                    print(data_final)
                    timers["send"] = current_time  # 记录上次发送时间
                else:
                    print("not enough")
            else:
                print("waiting......")
                # print(f"[COOLDOWN] Wait {send_cooldown - (current_time - last_send_time):.1f}s")
            
            data_arr = []
            frame_count = 0


        # 检查连续未检测超时
        last_seen_gesture = getattr(gesture_detector, 'last_seen_time', 0)
        last_seen_emotion = getattr(emotion_detector, 'last_seen_time', 0)
        # print(f"Last seen gesture: {last_seen_gesture}, Last seen emotion: {last_seen_emotion}")
        now = time.time()
        # print(f"Now: {now}")
        if (now - last_seen_gesture > 1) and (now - last_seen_emotion > 1):
            if frame_count > 0 or len(data_arr) > 0:
                print("[RESET] ")
            data_arr = []
            frame_count = 0
       
        # 20s重置相同类型计时器
        if current_time - timers["same_type"] >= 20.0:
            last_type = 0
            timers["same_type"] = current_time
            print("[SAME_TYPE_RESET]")

        if serial_comm.ser_51_4.in_waiting:
            data = serial_comm.ser_51_4.read(serial_comm.ser_51_4.in_waiting)
            serial_comm.ser_32_0.write(data)
            print(f"[S0] Forwarded: {data.hex()}")

        # -------------------- 串口发送 --------------------

        #print(fps)
        pTime = cTime
        # cv2.putText(frameClone, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
      
        # 始终刷新窗口（不再因无人脸而 continue）
        if imshow_flag:
            cv2.imshow('your_face', frameClone)
            # cv2.imshow("Probabilities", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cv2.destroyAllWindows()
