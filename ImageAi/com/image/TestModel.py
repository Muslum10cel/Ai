import numpy as np
import cv2 as cv
from keras.preprocessing.image import img_to_array
from keras.models import load_model

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "/", "+", "-", "="]


def temp2():
    im = cv.imread("7_2898.jpg")
    im = cv.resize(im, (28, 28))
    cv.imwrite("im77.jpg", im)
    im = im.astype("float") / 255.0
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)
    cl = model.predict(im)[0]
    print(cl)
    print(max(cl))
    print(np.argmax(cl))
    print(classes[np.argmax(cl)])
    for i in range(len(cl)):
        print("{0} --> {1}".format(i, cl[i]))


def temp1():
    capture = cv.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        cv.imshow('image', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.imwrite("test_55.jpg", frame)
            # im = cv.resize(frame, (28, 28))
            # im = im.astype("float") / 255.0
            # im = img_to_array(im)
            # im = np.expand_dims(im, axis=0)
            # cl = model.predict(im)[0]q
            # print(cl)
            # print(max(cl))
            # print(classes[np.argmax(cl)])
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    model = load_model("my_model3")
    temp2()
