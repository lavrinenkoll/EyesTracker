import os
import shutil
import cv2
import numpy as np


# save the position of eyes in the last frame
left_eye_save, watching_data = None, None


def detect_eyes(img):
    global left_eye_save, right_eye_save
    # save the original image
    img_save = img.copy()
    # use the cascade to detect eyes in the image and return the position of eyes
    right_eye_cascade = cv2.CascadeClassifier('res/haarcascade_righteye_2splits.xml')
    right_eye = right_eye_cascade.detectMultiScale(img, 1.1, 4)
    left_eye_cascade = cv2.CascadeClassifier('res/haarcascade_lefteye_2splits.xml')
    left_eye = left_eye_cascade.detectMultiScale(img, 1.1, 4)

    # find the position of eyes if there are two eyes, and return the result of watching
    for (x, y, w, h) in right_eye:
        for (x2, y2, w2, h2) in left_eye:
            # if the eyes are too close to each other, return false
            if x2+w2 < x or x+w < x2 or y2+h2 < y or y+h < y2:
                # if the eyes are too different in size, return false
                if w2*h2/w*h < 0.7 or w*h/w2*h2 < 0.7:
                    right_eye_save = [x, y, w, h]
                    left_eye_save = [x2, y2, w2, h2]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                    cv2.putText(img, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return False, img_save, left_eye_save, right_eye_save
                # return true if the eyes are normal
                right_eye_save = [x, y, w, h]
                left_eye_save = [x2, y2, w2, h2]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                cv2.putText(img, "True", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return True, img_save, left_eye_save, right_eye_save
    # return false if there is only one eye or no eye
    cv2.putText(img, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return False, img_save, left_eye_save, right_eye_save


def main(path):
    # open the video
    cap = cv2.VideoCapture(path)

    # set counter
    k = 1
    # create folders to save data if not exist
    try:
        os.mkdir("not_watching_data")
        os.mkdir("watching_data")
    except OSError:
        # recreate folders if exist
        shutil.rmtree("not_watching_data")
        os.mkdir("not_watching_data")
        shutil.rmtree("watching_data")
        os.mkdir("watching_data")

    # read the video frame by frame
    while True:
        # read the frame
        ret, frame = cap.read()
        # break if the video is over
        if not ret:
            cv2.destroyAllWindows()
            break
        # resize the frame
        frame = cv2.resize(frame, (812, 512))

        # detect eyes in the frame
        value, img_save, left, right = detect_eyes(frame)
        all_eyes = None

        # save the eyes if there are two eyes
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

        # show the frame
        cv2.imshow('process of creating training data', frame)

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break