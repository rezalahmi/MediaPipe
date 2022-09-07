import itertools
import mediapipe as mp


def landMarkDetector(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                             min_detection_confidence=0.5)

    face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
    LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
    RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
    LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
    RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
    if face_mesh_results.multi_face_landmarks:
        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            print(f'FACE NUMBER: {face_no + 1}')
            print('-----------------------')
            print(f'LEFT EYE LANDMARKS:n')
            for LEFT_EYE_INDEX in LEFT_EYE_INDEXES:
                print(face_landmarks.landmark[LEFT_EYE_INDEX])
            print(f'RIGHT EYE LANDMARKS:n')
            for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES:
                print(face_landmarks.landmark[RIGHT_EYE_INDEX])
            print(f'LEFT EYEBROW LANDMARKS:n')
            for LEFT_EYEBROW_INDEX in LEFT_EYEBROW_INDEXES:
                print(face_landmarks.landmark[LEFT_EYEBROW_INDEX])
            print(f'RIGHT EYEBROW LANDMARKS:n')
            for RIGHT_EYEBROW_INDEX in RIGHT_EYEBROW_INDEXES:
                print(face_landmarks.landmark[RIGHT_EYEBROW_INDEX])
    return face_mesh_results


def show_LandMark(image, face_mesh_results):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    img_copy = image[:, :, ::-1].copy()
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=img_copy,
                                      landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    return img_copy
