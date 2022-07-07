import cv2
import numpy as np
import pandas as pd
import requests
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
from insightface.utils.face_align import estimate_norm
from scipy.spatial.distance import cdist
from pathlib import Path
import matplotlib.pyplot as plt
import random
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200  # plot quality
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 1

PARENT_DIR = Path('/home/vid/hdd/projects/PycharmProjects/insightface/')
bright_etalon = 150  # constant 150
turnmetric = 20  # constant 20


class ArcFaceONNXVL(ArcFaceONNX):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, img, show=False):
        if show:
            plt.imshow(img)
            plt.show()
        embedding = self.get_feat(img).flatten()
        return np.expand_dims(embedding, axis=0)


class RetinaDetector(FaceAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_roi = False
        self.color = (0, 255, 0)
        self.thickness = 1

    def get(self, img, max_num=0, use_roi=None, min_face_size=None):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            xmin_box, ymin_box, xmax_box, ymax_box = bbox
            if min_face_size is not None:
                if (xmax_box - xmin_box < min_face_size[0]) or (ymax_box - ymin_box < min_face_size[1]):
                    continue  # skip small faces

            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)  # here make emb TODO: add brightnes

            if use_roi is not None:
                self.use_roi = True  # for correct plot_roi
                top_proc, bottom_proc, left_proc, right_proc = use_roi
                bbox_centroid_y, bbox_centroid_x = (xmin_box + xmax_box) / 2, (ymin_box + ymax_box) / 2
                orig_h, orig_w, _ = img.shape
                if top_proc + bottom_proc >= 100:
                    top_proc, bottom_proc = 0, 0
                if left_proc + right_proc >= 100:
                    left_proc, right_proc = 0, 0
                x_min_roi = max(0, int(orig_h / 100 * top_proc))  # for correct crop
                x_max_roi = min(orig_h, int(orig_h / 100 * (100 - bottom_proc)))  # for correct crop
                y_min_roi = max(0, int(orig_w / 100 * left_proc))  # for correct crop
                y_max_roi = min(orig_w, int(orig_w / 100 * (100 - right_proc)))  # for correct crop
                self.roi_points = {'x_min': x_min_roi, 'y_min': y_min_roi, 'x_max': x_max_roi, 'y_max': y_max_roi}

                if not x_min_roi <= bbox_centroid_x <= x_max_roi or not y_min_roi <= bbox_centroid_y <= y_max_roi:
                    continue  # centroid not in roi
            ret.append(face)
        return ret

    def draw_on(self, img, faces, plot_roi=False, plot_crop_face=False, plot_etalon=False, show=False):
        def _cv2_add_title(img, title, filled=True, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=0.7, thickness=2):
            img = img.copy()
            text_pos_x, text_pos_y = box[0] - 1, box[1] - 4
            if filled:
                (text_h, text_w), _ = cv2.getTextSize(title, font, font_scale, thickness)
                cv2.rectangle(img,
                              (text_pos_x, text_pos_y - text_w - 1),
                              (text_pos_x + text_h, text_pos_y + 4),
                              color, -1)
                cv2.putText(img, title, (text_pos_x, text_pos_y), font, font_scale, (255, 255, 255), thickness)
            else:
                cv2.putText(img, title, (text_pos_x, text_pos_y), font, font_scale, color, thickness)
            return img

        dimg = img.copy()
        if plot_roi and self.use_roi:
            dimg = cv2.rectangle(img.copy(),
                                 (self.roi_points['y_min'], self.roi_points['x_min']),
                                 (self.roi_points['y_max'], self.roi_points['x_max']),
                                 self.color, self.thickness)
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)  # xmin, ymin, xmax, ymax
            face.size = [box[2] - box[0], box[3] - box[1]]
            color = face['color'] if face.get('color') else (0, 255, 0)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 1)
            title = f'"{face.label}", ({round(float(face.det_score), 4)}, {round(float(face.rec_score), 4)}) size={face.size}'
            dimg = _cv2_add_title(dimg, title)
        if plot_crop_face:
            crops = [face.crop_face for face in faces]
            crops_together = self._get_coll_imgs(crops, dimg.shape)
            dimg = np.concatenate([dimg, crops_together], axis=1)

        if plot_etalon:
            etalon_pathes = [face.etalon_path for face in faces]
            etalons = []
            for etalon_path in etalon_pathes:
                if etalon_path is not None:
                    etalon = cv2.cvtColor(cv2.imread(str(etalon_path)), cv2.COLOR_BGR2RGB)
                else:
                    etalon = np.full(shape=(112, 112, 3), fill_value=(255, 0, 0), dtype=np.uint8)  # empties
                etalons.append(etalon)
            etalons_together = self._get_coll_imgs(etalons, dimg.shape)
            dimg = np.concatenate([dimg, etalons_together], axis=1)
        if show:
            plt.imshow(dimg)
            plt.show()
        return dimg

    def _get_coll_imgs(self, imgs_list, size, top=10, left=5, right=5):
        max_w = max([i.shape[1] for i in imgs_list])
        top_one, bottom_one = 1, 1
        good_size_ready = []
        for i in imgs_list:
            curr_w = i.shape[1]
            left = int((max_w - curr_w) / 2)
            right = max_w - curr_w - left
            vis_part_img = cv2.copyMakeBorder(i, top_one, bottom_one, left, right, cv2.BORDER_CONSTANT)
            good_size_ready.append(vis_part_img)
        ready_together = np.concatenate(good_size_ready, axis=0)

        bottom = size[0] - ready_together.shape[0] - top
        ready_together = cv2.copyMakeBorder(ready_together, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return ready_together


detector = RetinaDetector(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    # allowed_modules=['detection', 'recognition'],
    allowed_modules=['detection'],  # because need custom recognition module
    det_name='retinaface_mnet025_v2', rec_name='arcface_r100_v1',
)
detector.prepare(ctx_id=0, det_thresh=0.5)  # 0.5

session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)


class Person2:
    def __init__(self, path=None, full_img=None, face=None, embedding=None, label='Unknown', color=(255, 0, 0),
                 change_brightness=True, show=False):
        self.color = color
        self.label = label
        self.path = path
        if embedding is None:
            if full_img is not None:
                self.full_img = full_img
            elif self.path is not None:
                self.full_img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            if face is None:
                face = detector.get(img=self.full_img,
                                    # use_roi=(30, 10, 20, 28),  # how to change?
                                    min_face_size=(112, 112),  # how to change?
                                    )[0]
            self.crop_face, face.kps = norm_crop_self(full_img, face.kps, show=False)
            embedding = recognator.get(self.crop_face)
        self.embedding = embedding
        self.face = face

    def get_label(self, persons, threshold=0.7, metric='cosine',
                  turnmetric=turnmetric, face=None, use_nn=False,
                  show=False):
        dists = []
        for person in persons:
            dist = cdist(self.embedding, person.embedding, metric=metric)[0][0]
            dists.append(dist)
        who = np.argmin(dists)
        min_dist = round(dists[who], 5)
        self.etalon_path = persons[who].path
        if dists[who] < threshold:
            self.label = persons[who].label
            self.color = persons[who].color
            # self.etalon_path = persons[who].path
        if show:
            plt.imshow(self.crop_face)
            plt.title(f'"{self.label}": score={min_dist} (treshold={threshold})')
            plt.show()
        return min_dist


def norm_crop_self(img, landmark, image_size=112, mode='arcface', change_kpss_for_crop=True, show=False):
    # TODO: add brightness
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    if change_kpss_for_crop:
        landmark = np.array(list(map(
            lambda point: [
                M[0][0] * point[0] + M[0][1] * point[1] + M[0][2],  # change Ox
                M[1][0] * point[0] + M[1][1] * point[1] + M[1][2]  # change Oy
            ], landmark)))  # r_eye 0, l_eye 1, nose 2, r_mouth 3, l_mouth 4
    landmark = landmark.astype(int)
    if show:
        cimg = warped.copy()
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
        for idx_p, p in enumerate(landmark):
            cv2.circle(cimg, p, 1, colors[idx_p], 1)

        src = np.zeros(cimg.shape)
        cv2.line(src, landmark[0], landmark[1], (255, 255, 255), 1)
        cv2.line(src, landmark[1], landmark[4], (255, 255, 255), 1)
        cv2.line(src, landmark[3], landmark[4], (255, 255, 255), 1)
        cv2.line(src, landmark[0], landmark[3], (255, 255, 255), 1)
        cv2.circle(src, landmark[2], 1, (255, 255, 255), 1)

        concat = np.concatenate((cimg, src.astype(int)), axis=1)
        plt.imshow(concat)
        # TODO: add determiner if nose inside face
        # plt.title(f'nose inside face = {inside}')
        plt.show()
    return warped, landmark


def get_random_color():
    randomcolor = (random.randint(50, 200), random.randint(50, 200), random.randint(0, 150))
    return randomcolor


def persons_list_from_csv(df_path):
    df_persons = pd.read_csv(df_path, index_col=0)
    persons = []
    for label, line in df_persons.iterrows():
        img_path = line[0]
        emb = np.array([line.to_list()[1:]])
        person = Person2(path=img_path, label=label, color=get_random_color(), embedding=emb)
        persons.append(person)
    return persons


def get_imgs_thispersondoesnotexist(n=1, colors='RGB', show=False):
    imgs = []
    for i in range(n):
        img_str = requests.get('https://www.thispersondoesnotexist.com/image?').content
        nparr = np.frombuffer(img_str, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if colors == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if show:
            if colors != 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
        imgs.append(img)
    return imgs
