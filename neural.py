import os
import cv2
import keras
import numpy as np

from creating_train_data import detect_eyes


def create_model():
    # create a model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 128*2, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])

    # compile the model with Adam optimizer
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


def train_model(model, train_images, train_labels):
    # train the model
    model.fit(train_images, train_labels, epochs=100)
    return model


def predict(model, test_images):
    # predict the result
    predictions = model.predict(test_images)
    return predictions


def load_train_data():
    # load the data from the folders
    images_path = 'not_watching_data/'
    images = []
    labels = []
    # read the images from the folder
    for filename in os.listdir(images_path):
        # read the image
        img = cv2.imread(os.path.join(images_path, filename))
        # resize the image
        img = cv2.resize(img, (128*2, 128))
        # if the image is not None, append it to the list
        if img is not None:
            images.append(img)
            labels.append(1)

    # convert the list to numpy array
    images = np.array(images)
    labels = np.array(labels)

    # read the images from the folder
    images_fake = []
    labels_fake = []
    images_path = 'watching_data/'
    n = 0 #counter

    # read the images from the folder
    for filename in os.listdir(images_path):
        # read the image every 15 frames
        if n % 15 !=0:
            n += 1
            continue
        n += 1
        # read the image
        img = cv2.imread(os.path.join(images_path, filename))
        # resize the image
        img = cv2.resize(img, (128*2, 128))
        # if the image is not None, append it to the list
        if img is not None:
            images_fake.append(img)
            labels_fake.append(0)

    # convert the list to numpy array
    images_fake = np.array(images_fake)
    labels_fake = np.array(labels_fake)

    # concatenate the two arrays together
    images_all = np.concatenate((images, images_fake), axis=0)
    labels_all = np.concatenate((labels, labels_fake), axis=0)

    return images_all, labels_all


def init_model(images_all, labels_all):
    # create and train the model
    model = create_model()
    model = train_model(model, images_all, labels_all)
    # save the model
    model.save('model.h5')


def load_model():
    # load the model
    return keras.models.load_model('model.h5')


def main(path):
    # load the data
    img_all, labels_all = load_train_data()
    # create and train the model, then save it
    init_model(img_all, labels_all)

    # load the model
    model = load_model()

    # open the video
    vid = cv2.VideoCapture(path)
    # read the video frame by frame
    while True:
        # read the frame
        ret, frame = vid.read()
        # break if the video is over
        if not ret:
            cv2.destroyAllWindows()
            break
        # resize the frame
        frame = cv2.resize(frame, (812, 512))
        # detect eyes in the frame
        value, img_save, left, right = detect_eyes(frame)

        # if there are two eyes, predict the result
        if left is not None and right is not None:
            left = img_save[left[1]:left[1] + left[3], left[0]:left[0] + left[2]]
            right = img_save[right[1]:right[1] + right[3], right[0]:right[0] + right[2]]
            left = cv2.resize(left, (128, 128))
            right = cv2.resize(right, (128, 128))
            eyes = np.concatenate((right, left), axis=1)
            eyes = np.expand_dims(eyes, axis=0)

            predictions = predict(model, eyes)
            print(predictions)
            # show the result on the frame
            if predictions[0][0] < predictions[0][1]:
                cv2.putText(img_save, "Not watching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img_save, "Watching", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # show the frame
        cv2.imshow("applying a neural model to video", img_save)

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break