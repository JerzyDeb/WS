import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
scalling_factor = 0.5
frame = cv2.imread('twarz.jpeg')
frame = cv2.resize(
    frame,
    None,
    fx=scalling_factor,
    fy=scalling_factor,
    interpolation=cv2.INTER_AREA,
)
face_rects = face_cascade.detectMultiScale(
    frame,
    scaleFactor=1.3,
    minNeighbors=5,
)

for (x, y, w, h) in face_rects:
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        3,
    )

cv2.imshow('Twarz', frame)
cv2.waitKey(0)
