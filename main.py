import creating_train_data
import neural

if __name__ == '__main__':
    # collecting data
    creating_train_data.main("vids/vid_train.mp4")
    # create and use model
    neural.main("vids/vid_test.mp4")
