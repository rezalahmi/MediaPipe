import mediapipe as mp
import numpy as np
import itertools


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


def convert_landmark_to_point(landmarks, shape):
    xy_points = []
    for landmark in landmarks:
        xy_points.append((landmark.x * shape[1], landmark.y * shape[0]))
    return np.array(xy_points)
