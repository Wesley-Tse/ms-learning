import cv2

print("start creating videocapture")
cap = cv2.VideoCapture(0)
print("create successfuly")
print("press 'q' to exit")
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
