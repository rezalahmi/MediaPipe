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
    image = cv2.imread('images/raheleh.JPG')
    l, r = edit_facial_items.eyebrows_detection(image)
    hull = edit_facial_items.convert_landmark_to_point(l, image.shape)
    result = edit_facial_items.remove_eyebrow(image, hull)
    plt.title("Result")
    plt.axis('off')
    plt.imshow(result)

    plt.show()
    # cv2.imwrite(os.path.join('result', 'annotation.jpg'), annotated_image)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
