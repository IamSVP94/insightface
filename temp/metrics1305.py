import cv2
import numpy as np
import pandas as pd
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shutil

metric = 'cosine'
threshold = 0.725
change_brightness = None
turntreshold = 9999

PARENT_DIR = Path('/home/psv/PycharmProjects/insightface/')
SAVE_DIR = PARENT_DIR / f'temp/metrics/'


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
            cv2.imshow('aimg', aimg)
            cv2.waitKey()
        embedding = self.get_feat(aimg).flatten()
        if face:
            face.embedding = embedding
        return np.expand_dims(embedding, axis=0)


session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
)
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)


class Person:
    def __init__(self, path=None, img=None, label='Unknown', color=(0, 0, 255), emb_net=recognator,
                 change_brightness=None, show=False):
        self.path = str(path) if path else None
        self.color = color
        self.label = label
        if img is None:
            img = cv2.imread(self.path)
        else:
            img = img
        if change_brightness:
            self.img = self.change_brightness(img, etalon=change_brightness)
        else:
            self.img = img
        self.brightness = self.get_brightness(self.img)
        if show:
            cv2.imshow(f'"{self.label}" brightness={self.brightness} {self.path}', self.img)
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
            turn = Person.get_turn(face, treshold=turntreshold, img=full_img, show=show)
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
    def change_brightness(img, etalon=150, show=False):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        orig_br = int(np.mean(v))
        value = etalon - orig_br
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        final_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        if show:
            vis = np.concatenate((img, final_img), axis=1)
            cv2.imshow(f'before {orig_br}:after ~{etalon}', vis)
            cv2.waitKey()
        return final_img

    @staticmethod
    def change_brightness1(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def get_turn(face, treshold=10, img=None, show=False):
        r_eye_x, r_eye_y, l_eye_x, l_eye_y, nose_x, nose_y, r_m_x, r_m_y, l_m_x, l_m_y = face  # (X,Y)

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


DIR = Path('/home/psv/file/project/LABELED_FACES/')
show = False
person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', change_brightness=change_brightness, show=show)
person2 = Person(path=DIR / 'Vladislav/fas.jpg', label='Vladislav', change_brightness=change_brightness, show=show)
person3 = Person(path=DIR / 'Putin/fas.jpg', label='Putin', change_brightness=change_brightness, show=show)
person4 = Person(path=DIR / 'Irina/fas.jpg', label='Irina', change_brightness=change_brightness, show=show)
person5 = Person(path=DIR / 'Korzh/fas.jpg', label='Korzh', change_brightness=change_brightness, show=show)
person6 = Person(path=DIR / 'Bruce/fas.jpg', label='Bruce', change_brightness=change_brightness, show=show)
# person7 = Person(path=DIR / 'Petrovich/fas0.jpg', label='Petrovich')
# person8 = Person(path=DIR / 'Semenich/fas0.jpg', label='Semenich')
persons = [person1, person2, person3, person4, person5, person6]

df = pd.DataFrame(columns=['GT', 'PRED', 'dist', 'img_shape'])
if __name__ == '__main__':
    DATASET_DIR = Path('/home/psv/file/project/office_faces/')
    df_kps = pd.read_csv(DATASET_DIR / 'df_kps.csv', index_col='file').astype(int)
    imgs = list(DATASET_DIR.glob('**/*.jpg'))
    p_bar = tqdm(imgs)
    bad_results = []
    for idx, img_path in enumerate(p_bar):
        kps = df_kps.loc[img_path.stem, :]
        p_bar.set_description(f'{img_path}')
        GT = img_path.parent.parts[-1]
        unknown = Person(path=img_path, change_brightness=change_brightness)
        near_dist = unknown.get_label(persons, face=kps, full_img=unknown.img, threshold=threshold, metric=metric)
        df.loc[img_path.stem, :] = [GT, unknown.label, near_dist, unknown.img.shape[:2]]
        if GT != unknown.label:
            bad_results.append(img_path)
        if idx > 15:
            break

    top1 = 100 - len(df[df['GT'] != df['PRED']]) / (len(df) / 100)

    if bad_results:
        # change name here!!!
        bad_results_dir = SAVE_DIR / f'metric={metric}_threshold={threshold}_brightness={change_brightness}_turn={turntreshold}'
        for img_path in bad_results:
            GT, PRED, dist, _ = df.loc[img_path.stem]
            bad_results_dir_person = bad_results_dir / GT / PRED
            bad_results_dir_person.mkdir(parents=True, exist_ok=True)
            new_img_path = bad_results_dir_person / f'dist={round(dist, 4)}_GT={GT}_PRED={PRED}_{img_path.name}'
            shutil.copy(img_path, new_img_path)
        good_results = [img for img in imgs if img not in bad_results]
        assert len(imgs) - len(bad_results) == len(good_results)
        good_results_dir = bad_results_dir / f'good_metric={metric}_threshold={threshold}'
        good_results_dir.mkdir(parents=True, exist_ok=True)
        for img_path in good_results:
            GT, PRED, dist, _ = df.loc[img_path.stem]
            new_img_path = good_results_dir / f'dist={round(dist, 4)}_GT={GT}_PRED={PRED}_{img_path.name}'
            shutil.copy(img_path, new_img_path)

    good_results_dir = SAVE_DIR / f'good_metric={metric}_threshold={threshold}'

    labels = [person.label for person in persons]
    labels.append('Unknown')
    cf_matrix = confusion_matrix(df['GT'], df['PRED'], labels=labels)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # change name here!!!
    df.to_csv(
        SAVE_DIR / f'metric={metric}_threshold={threshold}_brightness={change_brightness}_turn={turntreshold}_top1={round(top1, 3)}.csv')
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt=".6g")
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    ax.tick_params(axis='both', labelsize=8)
    # change name here!!!
    ax.set_title(
        f'metric={metric} threshold={threshold} brightness={change_brightness} turn={turntreshold}\ntop1={round(top1, 3)}')
    ax.set_xlabel('PRED')
    ax.set_ylabel('GT')

    fig = ax.get_figure()
    # fig.show()
    # change name here!!!
    fig.savefig(
        SAVE_DIR / f'metric={metric}_threshold={threshold}_brightness={change_brightness}_turn={turntreshold}_top1={round(top1, 3)}.jpg')
