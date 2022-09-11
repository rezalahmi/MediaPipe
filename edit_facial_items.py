import mediapipe as mp
import numpy as np
import itertools
from scipy.spatial import Delaunay
import cv2


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def eyebrows_detection(image):
    left_eyebrow_point = []
    right_eyebrow_point = []
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                             min_detection_confidence=0.5)
    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
    RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            for RIGHT_EYEBROW_INDEX in RIGHT_EYEBROW_INDEXES:
                right_eyebrow_point.append(face_landmarks.landmark[RIGHT_EYEBROW_INDEX])
            for LEFT_EYEBROW_INDEX in LEFT_EYEBROW_INDEXES:
                left_eyebrow_point.append(face_landmarks.landmark[LEFT_EYEBROW_INDEX])
    return left_eyebrow_point, right_eyebrow_point


def find_forehead(image):
    forehead = []
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                             min_detection_confidence=0.5)
    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            return face_landmarks.landmark[10].x * image.shape[1], face_landmarks.landmark[10].y * image.shape[0]


def convert_landmark_to_point(landmarks, shape):
    xy_points = []
    for landmark in landmarks:
        xy_points.append((landmark.x * shape[1], landmark.y * shape[0]))
    return np.array(xy_points)


def remove_eyebrow(image, hull):
    pixels_index = np.zeros((image.shape[0] * image.shape[1], 2))
    x, y = find_forehead(image)
    color = image[int(y), int(x)]
    for loop1 in range(image.shape[0]):
        for loop2 in range(image.shape[1]):
            pixels_index[loop1 * image.shape[1] + loop2, 0] = loop1
            pixels_index[loop1 * image.shape[1] + loop2, 1] = loop2
    # if pixel in hull, return True
    points = in_hull(pixels_index, hull)
    for loop1 in range(image.shape[0]):
        for loop2 in range(image.shape[1]):
            if points[loop1 * image.shape[1] + loop2]:
                image[loop2, loop1] = color
    return image


def blur_eyebrow(image, hull):
    pixels_index = np.zeros((image.shape[0] * image.shape[1], 2))
    color = (255, 255, 255)
    mask = np.zeros(image.shape, dtype=np.uint8)
    for loop1 in range(image.shape[0]):
        for loop2 in range(image.shape[1]):
            pixels_index[loop1 * image.shape[1] + loop2, 0] = loop1
            pixels_index[loop1 * image.shape[1] + loop2, 1] = loop2
    # if pixel in hull, return True
    points = in_hull(pixels_index, hull)
    for loop1 in range(image.shape[0]):
        for loop2 in range(image.shape[1]):
            if points[loop1 * image.shape[1] + loop2]:
                mask[loop2, loop1] = color
    blurred_img = cv2.GaussianBlur(image, (205, 205), 0)
    result = np.where(mask == np.array([255, 255, 255]), blurred_img, image)
    return result
