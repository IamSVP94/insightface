import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
from scipy.spatial.distance import cdist
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math

PARENT_DIR = Path('/home/vid/hdd/projects/PycharmProjects/insightface/')
bright_etalon = 150  # constant 150
turnmetric = 20  # constant 20


def get_random_color():
    randomcolor = (random.randint(100, 255), random.randint(100, 255), random.randint(0, 100))
    return randomcolor


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def draw_(img, landmarks):
    (right_eye_x, right_eye_y), (left_eye_x, left_eye_y), nose, right_mouth, left_mouth = landmarks.astype(np.int32)
    cimg = img.copy()
    cv2.circle(cimg, (right_eye_x, right_eye_y), 1, (255, 255, 0), 1)
    cv2.circle(cimg, (left_eye_x, left_eye_y), 1, (255, 255, 0), 2)
    cv2.circle(cimg, nose, 1, (0, 255, 0), 3)
    cv2.circle(cimg, right_mouth, 1, (255, 0, 0), 1)
    cv2.circle(cimg, left_mouth, 1, (255, 0, 0), 2)
    return cimg


# this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure_old(img, landmark):
    landmark = landmark.astype(np.int32)
    right_eye, left_eye, nose, right_mouth, left_mouth = landmark
    e_center = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))  # eyes center
    m_center = (int((right_mouth[0] + left_mouth[0]) / 2), int((right_mouth[1] + left_mouth[1]) / 2))  # mouth center

    # -----------------------
    upside_down = False
    if m_center[1] < e_center[1]:
        upside_down = True
    # -----------------------
    # find rotation direction

    if m_center[0] > e_center[0]:
        point_3rd = (m_center[0], e_center[1])
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (e_center[0], m_center[1])
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges
    a = findEuclideanDistance(np.array(e_center), np.array(point_3rd))
    b = findEuclideanDistance(np.array(m_center), np.array(point_3rd))
    c = findEuclideanDistance(np.array(m_center), np.array(e_center))
    # -----------------------

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle_rad = np.arccos(cos_a)  # angle in radian
        angle = (angle_rad * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == 1:
            angle = 90 - angle

        if upside_down == True:
            angle = 90 - angle

        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))

        # uncomment for draw
        # cimg = draw_(img, landmark)
        # cv2.circle(cimg, e_center, 1, (100, 100, 100), 3)
        # cv2.circle(cimg, m_center, 1, (100, 100, 100), 3)
        # cv2.circle(cimg, point_3rd, 1, (200, 200, 200), 3)

        # params for rotate transformation
        if direction == 1:
            rot_mat_angle = angle
            turn_landmark = lambda x, y: (x * cos_a + y * sin_a, -x * sin_a + y * cos_a)
        else:
            rot_mat_angle = -angle
            turn_landmark = lambda x, y: (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        # rotate img (around img center)
        rot_mat = cv2.getRotationMatrix2D(image_center, rot_mat_angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        # rotate landmark (around img center)
        new_landmark = []
        for x, y in landmark:
            x, y = turn_landmark(x - image_center[0], y - image_center[1])
            x += image_center[0]
            y += image_center[1]
            new_landmark.append((x, y))

        if new_landmark:
            landmark = np.array(new_landmark)

            # uncomment for draw
            # dimg = draw_(img, landmark)
            # vis = np.concatenate((cimg, dimg), axis=1)
            # title = f'"{direction}" "{angle}"'
            # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            # plt.title(title)
            # plt.show()
            # cv2.imshow(title, vis)
            # cv2.waitKey()

    # -----------------------
    return img, landmark  # return img & landmark anyway


def alignment_procedure_new(full_img, box, landmark):
    def _get_new_landmark(landmark, image_center, box):
        new_landmark = []
        for x, y in landmark:
            x, y = turn_landmark(x - image_center[0], y - image_center[1])
            x += image_center[0]
            y += image_center[1]
            new_landmark.append((x, y))
        if new_landmark:
            xmin, ymin, xmax, ymax = box
            # return np.array(new_landmark)
            return np.array([[k[0] - xmin, k[1] - ymin] for k in new_landmark])
        return landmark

    def _get_new_box(box, image_center, cos):
        orig_box_w = box[2] - box[0]
        orig_box_h = box[3] - box[1]

        tl = (box[0], box[1])
        tr = (box[2], box[1])
        dl = (box[0], box[3])
        dr = (box[2], box[3])

        new_box = []
        for x, y in [tl, tr, dl, dr]:
            x, y = turn_landmark(x - image_center[0], y - image_center[1])
            x += image_center[0]
            y += image_center[1]
            new_box.append((x, y))
        if new_box:
            xmin = max(0, min([i[0] for i in new_box]))
            ymin = max(0, min([i[1] for i in new_box]))
            xmax = min(full_img.shape[1], max([i[0] for i in new_box]))
            ymax = min(full_img.shape[0], max([i[1] for i in new_box]))

            half_diff_w = int((orig_box_w - (xmax - xmin)) / 2 / cos)
            half_diff_h = int((orig_box_h - (ymax - ymin)) / 2)
            return np.array([xmin - half_diff_w, ymin - half_diff_h, xmax + half_diff_w, ymax + half_diff_h])
        else:
            return box

    landmark = landmark.astype(np.int32)
    img = full_img[box[1]:box[3], box[0]:box[2]]
    right_eye, left_eye, nose, right_mouth, left_mouth = landmark
    e_center = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))  # eyes center
    m_center = (int((right_mouth[0] + left_mouth[0]) / 2), int((right_mouth[1] + left_mouth[1]) / 2))  # mouth center

    # -----------------------
    upside_down = False
    if m_center[1] < e_center[1]:
        upside_down = True
    # -----------------------
    # find rotation direction

    if m_center[0] > e_center[0]:
        point_3rd = (m_center[0], e_center[1])
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (e_center[0], m_center[1])
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges
    a = findEuclideanDistance(np.array(e_center), np.array(point_3rd))
    b = findEuclideanDistance(np.array(m_center), np.array(point_3rd))
    c = findEuclideanDistance(np.array(m_center), np.array(e_center))
    # -----------------------

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle_rad = np.arccos(cos_a)  # angle in radian
        angle = (angle_rad * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == 1:
            angle = 90 - angle

        if upside_down == True:
            angle = 90 - angle

        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))

        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        # image_center = int(image_center[0] + box[0]), int(image_center[1] + box[1])
        image_center = int((image_center[0] + box[0]) / cos_a), int((image_center[1] + box[1]) / cos_a)

        # params for rotate transformation
        if direction == 1:
            rot_mat_angle = angle
            turn_landmark = lambda x, y: (int(x * cos_a + y * sin_a), int(-x * sin_a + y * cos_a))
        else:
            rot_mat_angle = -angle
            turn_landmark = lambda x, y: (int(x * cos_a - y * sin_a), int(x * sin_a + y * cos_a))

        rot_mat = cv2.getRotationMatrix2D(image_center, rot_mat_angle, 1.0)
        # rotate img (around img center)
        full_img = cv2.warpAffine(full_img, rot_mat, full_img.shape[1::-1], flags=cv2.INTER_LINEAR)
        # rotate box (around img center)
        box = _get_new_box(box, image_center, cos=cos_a)
        # rotate landmark (around img center)
        landmark = _get_new_landmark(landmark, image_center, box)

        img = full_img[box[1]:box[3], box[0]:box[2]]

    # -----------------------
    return img, landmark  # return img & landmark anyway


def alignment_procedure(full_img, box, landmark):
    def _get_new_landmark(landmark, image_center, box):
        new_landmark = []
        for x, y in landmark:
            x, y = turn_landmark(x - image_center[0], y - image_center[1])
            x += image_center[0]
            y += image_center[1]
            new_landmark.append((x, y))
        if new_landmark:
            xmin, ymin, xmax, ymax = box
            return np.array([[k[0] - xmin, k[1] - ymin] for k in new_landmark])
        return landmark

    def _get_new_box(box, image_center, cos):
        orig_box_w = box[2] - box[0]
        orig_box_h = box[3] - box[1]

        tl = (box[0], box[1])
        tr = (box[2], box[1])
        dl = (box[0], box[3])
        dr = (box[2], box[3])

        new_box = []
        for x, y in [tl, tr, dl, dr]:
            x, y = turn_landmark(x - image_center[0], y - image_center[1])
            x += image_center[0]
            y += image_center[1]
            new_box.append((x, y))
        if new_box:
            xmin = max(0, min([i[0] for i in new_box]))
            ymin = max(0, min([i[1] for i in new_box]))
            xmax = min(full_img.shape[1], max([i[0] for i in new_box]))
            ymax = min(full_img.shape[0], max([i[1] for i in new_box]))

            half_diff_w = int((orig_box_w - (xmax - xmin)) / 2 / cos)
            half_diff_h = int((orig_box_h - (ymax - ymin)) / 2)
            return np.array([xmin - half_diff_w, ymin - half_diff_h, xmax + half_diff_w, ymax + half_diff_h])
        else:
            return box

    def _get_rotate_params(m_center, e_center):
        upside_down = False
        if m_center[1] < e_center[1]:
            upside_down = True
        # find rotation direction
        if m_center[0] > e_center[0]:
            direction = -1  # rotate same direction to clock
            point_3rd = (m_center[0], e_center[1])
        else:
            direction = 1  # rotate inverse direction of clock
            point_3rd = (e_center[0], m_center[1])
        return upside_down, direction, point_3rd

    def _get_angle(m_center, e_center, point_3rd):
        # find length of triangle edges
        a = findEuclideanDistance(np.array(e_center), np.array(point_3rd))
        b = findEuclideanDistance(np.array(m_center), np.array(point_3rd))
        c = findEuclideanDistance(np.array(m_center), np.array(e_center))
        # apply cosine rule
        if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
            # In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
            cos_a = min(1.0, max(-1.0, cos_a))

            angle_rad = np.arccos(cos_a)  # angle in radian
            angle = (angle_rad * 180) / math.pi  # radian to degree
            return angle
        else:
            return None

    landmark = landmark.astype(np.int32)
    right_eye, left_eye, nose, right_mouth, left_mouth = landmark
    e_center = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))  # eyes center
    m_center = (int((right_mouth[0] + left_mouth[0]) / 2), int((right_mouth[1] + left_mouth[1]) / 2))  # mouth center

    upside_down, direction, point_3rd = _get_rotate_params(m_center, e_center)

    angle = _get_angle(m_center, e_center, point_3rd)
    if angle is None:
        xmin, ymin, xmax, ymax = box
        landmark = np.array([[k[0] - xmin, k[1] - ymin] for k in landmark])
        img = full_img[ymin:ymax, xmin:xmax]
        return img, landmark

    if direction == 1:
        angle = 90 - angle

    if upside_down == True:
        angle = 90 - angle  # TODO: test it

    cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))

    image_center = int((box[2] - box[0]) / 2 + box[0]), int((box[3] - box[1]) / 2 + box[1])

    # params for rotate transformation
    if direction == 1:
        rot_mat_angle = angle
        turn_landmark = lambda x, y: (int(x * cos_a + y * sin_a), int(-x * sin_a + y * cos_a))
    else:
        rot_mat_angle = -angle
        turn_landmark = lambda x, y: (int(x * cos_a - y * sin_a), int(x * sin_a + y * cos_a))

    rot_mat = cv2.getRotationMatrix2D(image_center, rot_mat_angle, 1.0)
    # rotate img (around img center)
    full_img = cv2.warpAffine(full_img, rot_mat, full_img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # rotate box (around img center)
    box = _get_new_box(box, image_center, cos=cos_a)
    # rotate landmark (around img center)
    landmark = _get_new_landmark(landmark, image_center, box)

    img = full_img[box[1]:box[3], box[0]:box[2]]

    # -----------------------
    return img, landmark  # return img & landmark anyway


class ArcFaceONNXVL(ArcFaceONNX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, img, face=None, show=False):
        if face:
            aimg, face.kps = alignment_procedure(img, landmark=face.kps)
        else:
            aimg = cv2.resize(img, (112, 112))
        if show:
            cv2.imshow('35 aimg.shape', aimg)
            cv2.waitKey()
        embedding = self.get_feat(aimg).flatten()
        if face:
            face.embedding = embedding
        return np.expand_dims(embedding, axis=0)


class RetinaDetector(FaceAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, img, max_num=0, change_kpss_for_crop=True):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            if change_kpss_for_crop:
                xmin, ymin, xmax, ymax = bbox
                kps = np.array([[k[0] - xmin, k[1] - ymin] for k in kps])
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces, show_kps=False, whoes=None, show=False):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = whoes[i][2]
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 1)
            if show_kps and face.kps is not None:
                kps = face.kps.astype(np.int32)
                for l in range(kps.shape[0]):
                    color_kps = (0, 0, 255)
                    if l == 0 or l == 3:
                        color_kps = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color_kps, 1)
            if whoes:
                label = whoes[i][0]
                recog_dist = whoes[i][1]
                cv2.putText(dimg, f'"{label}", {recog_dist} brightness={face.brightness}',
                            (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)
        if show:
            # dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
            # plt.imshow(dimg)
            # plt.show()
            cv2.imshow('dimg', dimg)
            cv2.waitKey()
        return dimg


from retinaface import RetinaFace

detect_model = RetinaFace.build_model()


def find_face(filename, threshold=0.3, model=detect_model):
    image = cv2.imread(str(filename))
    face_landmarks = RetinaFace.detect_faces(str(filename), model=model, threshold=threshold)
    facelist = []
    for face in face_landmarks.values():
        facecl = Face(
            bbox=np.array(face['facial_area']).astype(np.int32),
            kps=np.array(list(face['landmarks'].values())),
            det_score=face['score'])
        crop_face = image[facecl.bbox[1]:facecl.bbox[3], facecl.bbox[0]:facecl.bbox[2]]
        facecl.embedding = recognator.get(crop_face, show=False)
        facelist.append(facecl)
    return facelist


detector = RetinaDetector(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition'],
    det_name='retinaface_mnet025_v2', rec_name='arcface_r100_v1',
)
detector.prepare(ctx_id=0, det_thresh=0.5)  # 0.5

session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)


class Person:
    def __init__(self, path=None, img=None, embedding=None,
                 makemask=False, align=True, kps=None,
                 label='Unknown', color=(0, 0, 255), emb_net=recognator,
                 change_brightness=True, show=False, show_kps=False):
        self.path = str(path) if path else None
        self.color = color
        self.kps = kps
        self.label = label
        self.embedding = embedding
        if img is None:
            img = cv2.imread(self.path)
        else:
            img = img
        self._full_img = img
        if makemask:
            faces = detector.get(img, change_kpss_for_crop=False)
            # faces = detector.get(img, change_kpss_for_crop=True)

            # faces = find_face(str(self.path))

            # if len(faces) != 1:  # TODO: add this condition for prod!!!
            #     vis = img
            #     h, w, _ = img.shape
            #     for face in faces:
            #         box = face.bbox.astype(np.int32)
            #         crop_face = img[box[1]:box[3], box[0]:box[2]]
            #         cimg = draw_(crop_face, face.kps)
            #         curr_h, curr_w, _ = cimg.shape
            #         top = int((h - curr_h) / 2)
            #         bottom = h - curr_h - top
            #         left, right = 10, 10
            #         cimg = cv2.copyMakeBorder(cimg, top, bottom, left, right, cv2.BORDER_CONSTANT)
            #         vis = np.concatenate((vis, cimg), axis=1)
            #
            #     h, w, _ = vis.shape
            #     vis = cv2.resize(vis, (int(w / 2), int(h / 2)))
            #     cv2.imshow(f'there is {len(faces)} faces, Should be only 1!!!', vis)
            #     cv2.waitKey()
            #     assert len(
            #         faces) == 1, f'"{len(faces)}" persons on the photo "{self.path}"'  # only 1 face for etalon!!!
            #
            face = faces[0]
            box = face.bbox.astype(np.int32)
            self.kps = face.kps
            img = img[box[1]:box[3], box[0]:box[2]]
            if align and self.kps is not None and box is not None:
                img, self.kps = alignment_procedure(self._full_img, box=box, landmark=self.kps)
                # img, self.kps = alignment_procedure_old(img, landmark=self.kps)
        if change_brightness:
            self.img = self.change_brightness(img, etalon=bright_etalon)
        else:
            self.img = img
        if show and self.img is not None:
            cv2.imshow(f'"{self.label}" {self.path}', self.img)
            cv2.waitKey()
        self.emb_net = emb_net

        if show_kps:
            draw_img = draw_(self.img, self.kps)
            plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
            plt.title(f'"{self.label}" before emb')
            plt.show()

        if self.embedding is None:
            self.embedding = self._get_embedding()
            self.brightness = self.get_brightness(self.img)
        else:
            self.brightness = bright_etalon

    def _get_embedding(self):
        return self.emb_net.get(self.img, show=False)

    def get_label(self, persons, threshold=0.7, metric='cosine', face=None, show=False):
        dists = []
        for person in persons:
            dist = cdist(self.embedding, person.embedding, metric=metric)[0][0]
            dists.append(dist)
        who = np.argmin(dists)
        self.turn = 'center'
        self.turn_param = -999
        if face is not None:
            self.turn_param, self.turn = Person.get_turn(self.kps, self.img, treshold=turnmetric, show=show)
        if dists[who] < threshold and self.turn == 'center':
            self.label = persons[who].label
            self.color = persons[who].color
        return dists[who]

    @staticmethod
    def get_brightness(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return int(np.mean(v))

    @staticmethod
    def change_brightness(img, etalon=None, diff=None, show=False):  # etalon=150
        assert (etalon is not None) or (diff is not None), f'You shold set etalon or diff'
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        orig_br = int(np.mean(v))
        if etalon:
            value = etalon - orig_br
            v = cv2.add(v, value)
            v[v > 255] = 255
            v[v < 0] = 0
            hsv = cv2.merge((h, s, v))
        if diff:
            v = cv2.add(v, diff)
            v[v > 255] = 255
            v[v < 0] = 0
            hsv = cv2.merge((h, s, v))
        final_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if show:
            vis = np.concatenate((img, final_img), axis=1)
            cv2.imshow(f'before {orig_br}:after ~{etalon}', vis)
            cv2.waitKey()
        return final_img

    @staticmethod
    def get_turn(kps, img, treshold=10, show=False):
        (r_eye_x, r_eye_y), (l_eye_x, l_eye_y), (nose_x, nose_y), \
        (r_m_x, r_m_y), (l_m_x, l_m_y) = kps.astype(np.int32)

        img_center_X = int(img.shape[1] / 2)
        e_center = (int((l_eye_x - r_eye_x) / 2 + r_eye_x), int(np.abs(l_eye_y - r_eye_y) / 2 + min(r_eye_y, l_eye_y)))
        m_center = (int((l_m_x - r_m_x) / 2 + r_m_x), int(np.abs(l_m_y - r_m_y) / 2 + min(r_m_y, l_m_y)))
        c_center = (int((e_center[0] + m_center[0]) / 2), int((e_center[1] + m_center[1]) / 2))  # eye-mouth center

        c_n = (int((c_center[0] + nose_x) / 2), int((c_center[1] + nose_y) / 2))  # eye-mouth and nose center

        turn_param = 100 / img_center_X * c_n[0] - 100

        if - treshold < turn_param < treshold:
            turn = 'center'
        elif c_center[0] < img_center_X:
            turn = 'right'
        else:
            turn = 'left'
        if show and img is not None:
            cimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            plt.title(f'"{turn}"\nturn_param={round(turn_param, 3)}% treshold={treshold}')
            cv2.circle(cimg, (r_eye_x, r_eye_y), 1, (255, 255, 0), 1)
            cv2.circle(cimg, (l_eye_x, l_eye_y), 1, (255, 255, 0), 2)
            cv2.circle(cimg, (nose_x, nose_y), 1, (0, 255, 0), 1)
            cv2.circle(cimg, (r_m_x, r_m_y), 1, (255, 0, 0), 1)
            cv2.circle(cimg, (l_m_x, l_m_y), 1, (255, 0, 0), 2)

            cv2.line(cimg, (img_center_X, img.shape[0]), (img_center_X, 0), (200, 200, 200), 1)
            cv2.circle(cimg, c_n, 1, (200, 200, 200), 3)

            cv2.line(cimg, e_center, m_center, (0, 0, 255), 1)
            cv2.circle(cimg, c_center, 1, (0, 0, 255), 3)
            cv2.circle(cimg, e_center, 1, (255, 255, 0), 2)
            cv2.circle(cimg, m_center, 1, (255, 0, 0), 2)

            plt.imshow(cimg)
            plt.show()
        return int(turn_param), turn


def persons_list_from_csv(df_path):
    df_persons = pd.read_csv(df_path, index_col=0)
    persons = []
    for label, line in df_persons.iterrows():
        emb = np.array([line.to_list()[1:]])
        person = Person(label=label, color=get_random_color(), embedding=emb, change_brightness=False)
        persons.append(person)
    return persons
