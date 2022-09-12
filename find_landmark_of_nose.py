import cv2
import mediapipe as mp


def landmark_nose(image):
    nose = [2,
            94, 19, 1, 4, 5, 195, 197, 6, 168,
            193, 122, 196, 3, 51, 45, 44, 125, 141,
            417, 351, 419, 248, 281, 275, 274, 354, 370,
            465, 412, 399, 456, 363, 440, 461, 462,
            344, 360, 420, 458, 250,
            237, 220, 134, 236, 174, 241, 242,
            218, 115, 131, 238, 20]
    nose_landmarks = []
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             min_detection_confidence=0.3)
    results = face_mesh_images.process(image)
    if results.multi_face_landmarks:
        for face_np, face_landmarks in enumerate(results.multi_face_landmarks):
            for idx in nose:
                nose_landmarks.append((int(face_landmarks.landmark[idx].x * image.shape[1]),
                                       int(face_landmarks.landmark[idx].y * image.shape[0])))
    return nose_landmarks
