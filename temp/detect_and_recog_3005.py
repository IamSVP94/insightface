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

PARENT_DIR = Path('/home/psv/PycharmProjects/insightface/')

padding = 0
padding_type = '%'
align = True
mode = 'mean'
metric = 'cosine'
recog_tresh = 0.725
turnmetric = 10
bright_etalon = 150

new_img_dir_path = PARENT_DIR / 'temp' / f'bright4_1305_turn={turnmetric}_recog_tresh={recog_tresh}'
new_img_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_img_dir_path}')


class ArcFaceONNXVL(ArcFaceONNX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, img, face=None, show=False):
        if face:
            from insightface.utils import face_align
            aimg = face_align.norm_crop(img, landmark=face.kps)
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
    def __init__(self, path=None, img=None, label='Unknown', color=(0, 0, 255), emb_net=recognator,
                 change_brightness=True, show=False):
        self.path = str(path) if path else None
        self.color = color
        self.label = label
        if img is None:
            img = cv2.imread(self.path)
        else:
            img = img
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
        return self.emb_net.get(self.img)

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


DIR = Path('/home/psv/file/project/recog_datasets/LABELED_FACES/CROP_FACES/')
person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', color=(0, 255, 0), show=False)
person2 = Person(path=DIR / 'Vladislav/fas.jpg', label='Vladislav', color=(255, 0, 0), show=False)
person3 = Person(path=DIR / 'Putin/fas.jpg', label='Putin', color=(113, 47, 38), show=False)
person4 = Person(path=DIR / 'Irina/fas.jpg', label='Irina', color=(31, 58, 61), show=False)
person5 = Person(path=DIR / 'Korzh/fas.jpg', label='Korzh', color=(137, 40, 80), show=False)
person6 = Person(path=DIR / 'Bruce/fas.jpg', label='Bruce', color=(180, 40, 137), show=False)
person7 = Person(path=DIR / 'Farid/fas.jpg', label='Farid', color=(180, 237, 140), show=False)
persons = [person1, person2, person3, person4, person5, person6, person7]

if __name__ == '__main__':
    DATASET_DIR = Path('/home/psv/Pictures/mpv/goods_cooler/')
    # imgs = list(DATASET_DIR.glob('**/*.jpg'))
    imgs = list(DATASET_DIR.glob('*.jpg'))

    random.shuffle(imgs)

    p_bar = tqdm(imgs)
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.imread(str(img_path))
        faces = detector.get(img)
        whoes = []
        for face_idx, face in enumerate(faces):
            emb = np.expand_dims(face['embedding'], axis=0)
            box = face.bbox.astype(np.int32)
            crop_face = img[box[1]:box[3], box[0]:box[2]]
            crop_face = Person.change_brightness(crop_face, etalon=bright_etalon)
            unknown = Person(img=crop_face, change_brightness=True, show=False)
            face.brightness = unknown.brightness
            near_dist = unknown.get_label(persons, threshold=recog_tresh, full_img=img, face=face, show=False)
            whoes.append((unknown.label, round(near_dist, 4), unknown.color))
        dimg = detector.draw_on(img, faces, whoes=whoes, show_kps=False, show=False)
        cv2.imwrite(str(new_img_dir_path / f'{img_path.name}'), dimg)
        # exit()
        if img_idx > 50:
            exit()
# '''
