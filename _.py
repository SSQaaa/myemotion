import cv2

cap = cv2.VideoCapture(0)


while True:
  ret, frame = cap.read()
  if not ret:
    print("failed")
  cv2.imshow("ok", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.destroyAllWindows()