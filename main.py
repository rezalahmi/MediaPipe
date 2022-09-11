# This is a sample Python script.
import FaceLandMarkDetection
import matplotlib.pyplot as plt
import os
import cv2
import edit_facial_items

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = cv2.imread('images/reza.JPG')
    l, r = edit_facial_items.eyebrows_detection(image)
    hull = edit_facial_items.convert_landmark_to_point(l, image.shape)
    # result = edit_facial_items.remove_eyebrow(image, hull)
    result = edit_facial_items.blur_eyebrow(image, hull)
    result = cv2.resize(result, (500, 400))
    cv2.imshow('result', result)

    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
