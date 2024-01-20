import cv2
import numpy

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


def find_face(img_path, scaling_factor=0.5, scale_factor=1.3, min_neighbours=3):
    frame = cv2.imread(img_path)
    frame = cv2.resize(
        frame,
        None,
        fx=scaling_factor,
        fy=scaling_factor,
        interpolation=cv2.INTER_AREA,
    )
    face_rects = face_cascade.detectMultiScale(
        frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbours,
    )

    for (x, y, w, h) in face_rects:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3,
        )
    cv2.imshow('Found face', frame)


def find_smiles(img_path):
    image = cv2.imread(img_path)
    gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_filter, 1.2, 8)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )
        roi_gray = gray_filter[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 55, minSize=(25, 25))
        eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 6)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(
                roi_color,
                (sx, sy),
                (sx + sw, sy + sh),
                (0, 255, 0),
                1
            )
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 0, 255),
                1
            )
    print(f'Znaleziono: {len(faces)} osób')
    cv2.imshow('Smiles', image)


def capture_face_from_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        face_rects = face_cascade.detectMultiScale(
            frame,
            scaleFactor=2,
            minNeighbors=3,
        )
        for (x, y, w, h) in face_rects:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3,
            )
        cv2.imshow('Video', frame)
        print(f'Osoby na video: {len(face_rects)}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def find_people_on_video():
    cap = cv2.VideoCapture('video.mp4')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 560))
        gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = hog.detectMultiScale(gray_filter, winStride=(8, 8), scale=1.1)
        boxes = numpy.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(
                frame,
                (xa, ya),
                (xb, yb),
                (0, 255, 0),
                1,
            )
        cv2.imshow('Video', frame)
        print(f'Osoby na video: {len(boxes)}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


find_face('osoba.jpg')
find_smiles('osoby.jpg')
capture_face_from_video()
find_people_on_video()
cv2.waitKey(0)
