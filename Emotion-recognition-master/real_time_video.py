from tensorflow.keras.utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import time
import serial
 
ser = serial.Serial('/dev/ttyS4', 115200, timeout=0.1)  # 或 /dev/ttyUSB0

# parameters for loading data and images
detection_model_path = '/home/orangepi/Desktop/myemotion/Emotion-recognition-master/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = '/home/orangepi/Desktop/myemotion/Emotion-recognition-master/models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
assert not face_detection.empty(), "Haar 模型加载失败：检查 haarcascade_files 路径"
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["angry","disgust","scared","happy","sad","surprised","neutral"]

feelings_faces = []
for emotion in EMOTIONS:
    feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
# 在 Linux 更稳：指定 V4L2；如果在 Windows，可改为 cv2.VideoCapture(0)
try:
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
except:
    camera = cv2.VideoCapture(0)

# 可选：强制一组通用参数，很多UVC相机更稳
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    raise RuntimeError("Camera open failed")

# 预热丢几帧，避免刚打开时为空
for _ in range(8):
    camera.read()

cv2.namedWindow('your_face')

while True:
    ok, frame = camera.read()
    if not ok or frame is None:
        # 偶发丢帧：跳过本轮，不要崩
        time.sleep(0.01)
        continue

    # ✅ 只在确认拿到帧后再 resize
    # frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    preds = np.zeros(len(EMOTIONS), dtype=np.float32)  # 无人脸时显示全0
    label = "no face"

    if len(faces) > 0:
        # 选面积最大的人脸（w*h）
        (fX, fY, fW, fH) = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]

        # ROI 预处理与推理
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi, verbose=0)[0]
        label = EMOTIONS[preds.argmax()]

        # 画框与标签
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    # 概率条（无人脸时为全0，也会正常刷新）
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
        text = "{}: {:.2f}%".format(emotion, prob * 100)
        w = int(prob * 300)
        cv2.rectangle(canvas, (7, i * 35 + 5), (w, i * 35 + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, i * 35 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    # 始终刷新窗口（不再因无人脸而 continue）
    cv2.imshow('your_face', frameClone)
    # cv2.imshow("Probabilities", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
