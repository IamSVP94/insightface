import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math

PARENT_DIR = Path('/home/psv/PycharmProjects/insightface/')

align = True
bright_etalon = 100  # 150
n_persons = 2500


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(img, landmark):
    landmark = landmark.astype(np.int32)
    right_eye, left_eye, nose, right_mouth, left_mouth = landmark
    e_center = (int((right_eye[0] + left_eye[0]) / 2), int((right_eye[1] + left_eye[1]) / 2))
    m_center = (int((right_mouth[0] + left_mouth[0]) / 2), int((right_mouth[1] + left_mouth[1]) / 2))
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
                turn_param = whoes[i][3]
                cv2.putText(dimg, f'"{label}", {recog_dist} brightness={face.brightness} turn={turn_param}%',
                            (box[0] - 1, box[1] - 4), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)
        if show:
            # dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
            # plt.imshow(dimg)
            # plt.show()
            cv2.imshow('dimg', dimg)
            cv2.waitKey()
        return dimg


detector = RetinaDetector(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition'],
    det_name='retinaface_mnet025_v2', rec_name='arcface_r100_v1',
)
detector.prepare(ctx_id=0, det_thresh=0.5)

session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)


class Person:
    def __init__(self, path=None, img=None,
                 makemask=False, align=True, kps=None,
                 label='Unknown', color=(0, 0, 255), emb_net=recognator,
                 change_brightness=True, show=False):
        self.path = str(path) if path else None
        self.color = color
        self.kps = kps
        self.label = label
        if img is None:
            img = cv2.imread(self.path)
        else:
            img = img
        if makemask:
            faces = detector.get(img)
            face = faces[0]
            box = face.bbox.astype(np.int32)
            self.kps = face.kps
            img = img[box[1]:box[3], box[0]:box[2]]
        if align and self.kps is not None:
            img, self.kps = alignment_procedure(img, landmark=self.kps)
        if change_brightness:
            self.img = self.change_brightness(img, etalon=bright_etalon)
        else:
            self.img = img
        self.brightness = self.get_brightness(self.img)
        if show:
            cv2.imshow(f'"{self.label}" {self.path}', self.img)
            cv2.waitKey()
        self.emb_net = emb_net
        self.embedding = self._get_embedding()

    def _get_embedding(self):
        return self.emb_net.get(self.img, show=False)

    @staticmethod
    def get_brightness(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return int(np.mean(v))

    @staticmethod
    def change_brightness(img, etalon=None, diff=None, show=False):  # etalon=150
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
    def get_turn(kps, img, treshold=10):
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
        return int(turn_param), turn


cols = ['path']
cols.extend([i for i in range(0, 512)])
df = pd.DataFrame(columns=cols)

all_persons = []
PERSON_DIR = Path('/home/psv/Downloads/datasets/lfw1img/')
person_dirs = list(PERSON_DIR.glob('*'))
p_bar = tqdm(person_dirs[:n_persons], colour='yellow')
for dir in p_bar:
    p_bar.set_description(f'{str(dir)}')
    img_path = list(dir.glob('*.jpg'))[0]
    label = img_path.parts[-2]
    person = Person(path=img_path, label=label, makemask=True, change_brightness=True)
    for_write = [person.path]
    for_write.extend(person.embedding.tolist()[0])
    df.loc[person.label, :] = for_write
    all_persons.append(person)

DIR = Path('/home/psv/file/project/recog_datasets/LABELED_FACES/LABELED_full/')
person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', makemask=True)
person2 = Person(path=DIR / 'Vladislav/fas.jpg', label='Vladislav', makemask=True)
person3 = Person(path=DIR / 'Farid/fas.jpg', label='Farid', makemask=True)
person4 = Person(path=DIR / 'Denis/fas.jpg', label='Denis', makemask=True)
person5 = Person(path=DIR / 'Anton/fas.jpg', label='Anton', makemask=True)
person6 = Person(path=DIR / 'Alexandr/fas.jpg', label='Alexandr', makemask=True)
person7 = Person(path=DIR / 'Putin/fas.jpg', label='Putin', makemask=True)
person8 = Person(path=DIR / 'Irina/fas.jpg', label='Irina', makemask=True)
person9 = Person(path=DIR / 'Korzh/fas.jpg', label='Korzh', makemask=True)
person10 = Person(path=DIR / 'Bruce/fas.jpg', label='Bruce', makemask=True)
person11 = Person(path=DIR / 'Alena/fas.jpg', label='Alena', makemask=True)
person12 = Person(path=DIR / 'Dmitryi/fas.jpg', label='Dmitryi', makemask=True)
person13 = Person(path=DIR / 'VladislavV/fas.jpg', label='VladislavV', makemask=True)
all_persons.extend(
    [person1, person2, person3, person4, person5, person6, person7, person8, person9, person10, person11, person12,
     person13])
for person in all_persons:
    for_write = [person.path]
    for_write.extend(person.embedding.tolist()[0])
    df.loc[person.label, :] = for_write

df.to_csv(f'n_persons={n_persons + len(all_persons)}_bright_etalon={bright_etalon}_embeddings.csv')
# TODO: save all crop face in special dir
