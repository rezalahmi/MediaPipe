# This is a sample Python script.
import FaceGeometryDetection
import FaceLandMarkDetection
import matplotlib.pyplot as plt
import os
import cv2

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = FaceGeometryDetection.load_image('images/reza.JPG')
    plt.figure(figsize=[10, 10])
    plt.title("Sample Image")
    plt.axis('off')
    plt.imshow(image[:, :, ::-1])
    plt.show()
    results = FaceLandMarkDetection.landMarkDetector(image)
    annotated_image = FaceLandMarkDetection.show_LandMark(image, results)
    plt.title("Result")
    plt.axis('off')
    plt.imshow(annotated_image)
    plt.show()
    cv2.imwrite(os.path.join('result', 'annotation.jpg'), annotated_image)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
