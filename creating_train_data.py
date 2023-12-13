import os
import shutil
import cv2
import numpy as np


def read_data(path, type):
    data_img = []
    data_labels = []

    files = os.listdir(path)

    for file in files:
        data_img.append(cv2.imread(path + file))
        data_labels.append(type)

    return data_img, data_labels


left_eye_save, watching_data = None, None
def detect_eyes(img):
    global left_eye_save, right_eye_save
    img_save = img.copy()
    right_eye_cascade = cv2.CascadeClassifier('res/haarcascade_righteye_2splits.xml')
    right_eye = right_eye_cascade.detectMultiScale(img, 1.1, 4)
    left_eye_cascade = cv2.CascadeClassifier('res/haarcascade_lefteye_2splits.xml')
    left_eye = left_eye_cascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in right_eye:
        for (x2, y2, w2, h2) in left_eye:
            if x2+w2 < x or x+w < x2 or y2+h2 < y or y+h < y2:
                if w2*h2/w*h < 0.7 or w*h/w2*h2 < 0.7:
                    right_eye_save = [x, y, w, h]
                    left_eye_save = [x2, y2, w2, h2]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                    cv2.putText(img, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return False, img_save, left_eye_save, right_eye_save
                right_eye_save = [x, y, w, h]
                left_eye_save = [x2, y2, w2, h2]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                cv2.putText(img, "True", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return True, img_save, left_eye_save, right_eye_save
    cv2.putText(img, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return False, img_save, left_eye_save, right_eye_save


def main():
    cap = cv2.VideoCapture("vids/vid_train.mp4")

    k = 1
    try:
        os.mkdir("not_watching_data")
        os.mkdir("watching_data")
    except OSError:
        # recreate
        shutil.rmtree("not_watching_data")
        os.mkdir("not_watching_data")
        shutil.rmtree("watching_data")
        os.mkdir("watching_data")

    while True:
        ret, frame = cap.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        frame = cv2.resize(frame, (812, 512))

        value, img_save, left, right = detect_eyes(frame)
        all_eyes = None
        if left is not None and right is not None:
            try:
                left = img_save[left[1]:left[1] + left[3], left[0]:left[0] + left[2]]
                right = img_save[right[1]:right[1] + right[3], right[0]:right[0] + right[2]]
                left = cv2.resize(left, (128, 128))
                right = cv2.resize(right, (128, 128))
                all_eyes = np.concatenate((right, left), axis=1)
                if not value:
                    if all_eyes is not None: cv2.imwrite("not_watching_data/frame%d.jpg" % k, all_eyes)
                    k += 1
                else:
                    if all_eyes is not None: cv2.imwrite("watching_data/frame%d.jpg" % k, all_eyes)
                    k += 1
            except:
                print("err")

        cv2.imshow('process of creating training data', frame)

        if cv2.waitKey(1) == ord('q'):
            break