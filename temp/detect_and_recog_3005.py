import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import random
from PIL import Image
import math

PARENT_DIR = Path('/home/psv/PycharmProjects/insightface/')

align = True
recog_tresh = 0.65
turnmetric = 8
bright_etalon = 150
n_persons = 1505

new_img_dir_path = PARENT_DIR / 'temp' / f'bright_{bright_etalon}_3005_n_persons={n_persons}_turn={turnmetric}_recog_tresh={recog_tresh}'
new_img_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_img_dir_path}')


def get_random_color():
    randomcolor = (random.randint(100, 255), random.randint(100, 255), random.randint(0, 100))
    return randomcolor


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


# this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(img, landmark):
    (right_eye_x, right_eye_y), (left_eye_x, left_eye_y), nose, _, _ = landmark.astype(np.int32)  # (X,Y)
    # -----------------------
    upside_down = False
    if nose[1] < left_eye_y or nose[1] < right_eye_y:
        upside_down = True
    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array((left_eye_x, left_eye_y)), np.array(point_3rd))
    b = findEuclideanDistance(np.array((right_eye_x, right_eye_y)), np.array(point_3rd))
    c = findEuclideanDistance(np.array((right_eye_x, right_eye_y)), np.array((left_eye_x, left_eye_y)))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        if upside_down == True:
            angle = angle + 90

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * (-angle)))
    # -----------------------

    return img  # return img anyway


class ArcFaceONNXVL(ArcFaceONNX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, img, face=None, show=False):
        if face:
            aimg = alignment_procedure(img, landmark=face.kps)
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
                 makemask=False, align=True,
                 label='Unknown', color=(0, 0, 255), emb_net=recognator,
                 change_brightness=True, show=False):
        self.path = str(path) if path else None
        self.color = color
        self.label = label
        if img is None:
            img = cv2.imread(self.path)
        else:
            img = img
        if makemask:
            faces = detector.get(img)
            face = faces[0]
            # assert len(faces) == 1, f'"{len(faces)}" persons on the photo "{self.path}"'  # TODO: add condition
            box = face.bbox.astype(np.int32)
            img = img[box[1]:box[3], box[0]:box[2]]
            if align:
                img = alignment_procedure(img, landmark=face.kps)

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

    def get_label(self, persons, threshold=0.7, metric='cosine', face=None, full_img=None, show=False):
        dists = []
        for person in persons:
            dist = cdist(self.embedding, person.embedding, metric=metric)[0][0]
            dists.append(dist)
        who = np.argmin(dists)

        turn = 'center'
        if full_img is not None and face is not None:
            turn = Person.get_turn(face, treshold=turnmetric, img=full_img, show=show)
        if dists[who] < threshold and turn == 'center':
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
    def get_turn(face, treshold=10, img=None, show=False):
        (r_eye_x, r_eye_y), (l_eye_x, l_eye_y), \
        (nose_x, nose_y), \
        (r_m_x, r_m_y), (l_m_x, l_m_y) = face.kps.astype(np.int32)  # (X,Y)

        e_center = (int((l_eye_x - r_eye_x) / 2 + r_eye_x), int(np.abs(l_eye_y - r_eye_y) / 2 + min(r_eye_y, l_eye_y)))
        m_center = (int((l_m_x - r_m_x) / 2 + r_m_x), int(np.abs(l_m_y - r_m_y) / 2 + min(r_m_y, l_m_y)))

        turn_eye_X_center = e_center[0] / nose_x * 1000 - 1000
        turn_m_X_center = m_center[0] / nose_x * 1000 - 1000
        turn_X_center = (turn_eye_X_center + turn_m_X_center) / 2
        if - treshold < turn_X_center < treshold:
            turn = 'center'
        elif treshold > turn_eye_X_center:
            turn = 'left'
        else:
            turn = 'right'
        if show and img is not None:
            em_center = (int((e_center[0] + m_center[0]) / 2), int((e_center[1] + m_center[1]) / 2))
            cimg = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            plt.title(
                f'"{turn}"\ns={round(turn_X_center, 3)} e={round(turn_eye_X_center, 3)} m={round(turn_m_X_center, 3)}'
            )
            cv2.circle(cimg, (r_eye_x, r_eye_y), 1, (255, 255, 0), 1)
            cv2.circle(cimg, (l_eye_x, l_eye_y), 1, (255, 255, 0), 2)
            cv2.circle(cimg, (nose_x, nose_y), 1, (0, 255, 0), 1)
            cv2.circle(cimg, (r_m_x, r_m_y), 1, (255, 0, 0), 1)
            cv2.circle(cimg, (l_m_x, l_m_y), 1, (255, 0, 0), 2)
            cv2.line(cimg, e_center, m_center, (0, 0, 255), 1)
            cv2.circle(cimg, em_center, 1, (0, 0, 255), 2)

            box = face.bbox.astype(np.int32)
            crop_face = cimg[box[1]:box[3], box[0]:box[2]]

            plt.imshow(crop_face)
            plt.show()
        return turn


show_persons = False
all_persons = []
PERSON_DIR = Path('/home/psv/Downloads/datasets/lfw1img/')
person_dirs = list(PERSON_DIR.glob('*'))
p_bar = tqdm(person_dirs[:n_persons], colour='yellow')
for dir in p_bar:
    p_bar.set_description(f'{str(dir)}')
    img_path = list(dir.glob('*.jpg'))[0]
    label = img_path.parts[-2]
    person = Person(path=img_path, label=label, color=get_random_color(),
                    makemask=True, change_brightness=True,
                    show=show_persons)
    all_persons.append(person)

DIR = Path('/home/psv/file/project/recog_datasets/LABELED_FACES/LABELED_full/')
person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', color=(0, 255, 0), makemask=True, show=show_persons)
person2 = Person(path=DIR / 'Vladislav/fas.jpg', label='Vladislav', color=(255, 0, 0), makemask=True, show=show_persons)
person3 = Person(path=DIR / 'Farid/fas.jpg', label='Farid', color=(180, 237, 140), makemask=True, show=show_persons)
person4 = Person(path=DIR / 'Denis/fas.jpg', label='Denis', color=(80, 127, 200), makemask=True, show=show_persons)
person5 = Person(path=DIR / 'Anton/fas.jpg', label='Anton', color=(213, 147, 138), makemask=True, show=show_persons)
person6 = Person(path=DIR / 'Alexandr/fas.jpg', label='Alexandr', color=(131, 158, 101), makemask=True, show=show_persons)
person7 = Person(path=DIR / 'Putin/fas.jpg', label='Putin', color=(113, 47, 38), makemask=True, show=show_persons)
person8 = Person(path=DIR / 'Irina/fas.jpg', label='Irina', color=(31, 58, 61), makemask=True, show=show_persons)
person9 = Person(path=DIR / 'Korzh/fas.jpg', label='Korzh', color=(137, 40, 80), makemask=True, show=show_persons)
person10 = Person(path=DIR / 'Bruce/fas.jpg', label='Bruce', color=(180, 40, 137), makemask=True, show=show_persons)
all_persons.extend([person1, person2, person3, person4, person5, person6, person7, person8, person9, person10])

if __name__ == '__main__':
    # DATASET_DIR = Path('/home/psv/Pictures/mpv/goods_cooler/')
    # imgs = list(DATASET_DIR.glob('*.jpg'))
    DATASET_DIR = Path('/home/psv/Pictures/mpv/office_cooler_persons/')
    imgs = list(DATASET_DIR.glob('**/*.jpg'))

    # imgs.extend(list(PERSON_DIR.glob('**/*.jpg')))

    p_bar = tqdm(imgs, colour='green')
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.imread(str(img_path))

        # img = cv2.flip(img, 1)  # TODO: del this str!!!

        faces = detector.get(img)
        whoes = []
        for face_idx, face in enumerate(faces):
            emb = np.expand_dims(face['embedding'], axis=0)
            box = face.bbox.astype(np.int32)
            crop_face = img[box[1]:box[3], box[0]:box[2]]
            try:
                crop_face = Person.change_brightness(crop_face, etalon=bright_etalon)
            except cv2.error:
                continue
            unknown = Person(img=crop_face, change_brightness=True, show=False)
            face.brightness = unknown.brightness
            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, full_img=img, face=face, show=False)
            if unknown.label != 'Unknown':
                print('\n', img_path, unknown.label)
            whoes.append((unknown.label, round(near_dist, 4), unknown.color))

        if len(faces) == len(whoes):
            dimg = detector.draw_on(img, faces, whoes=whoes, show_kps=False, show=False)
            cv2.imwrite(str(new_img_dir_path / f'{img_path.name}'), dimg)
        # if img_idx > 50:
        #     exit()
# '''
