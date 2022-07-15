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
import onnxruntime as ort

mpl.rcParams['figure.dpi'] = 200  # plot quality
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 1

PARENT_DIR = Path('/home/vid/hdd/projects/PycharmProjects/insightface/')
bright_etalon = 150  # constant 150

landmarks_colors = [(0, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 0), (255, 0, 255)]


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
            title = f'"{face.label}", ({round(float(face.det_score), 4)}, {round(float(face.rec_score), 4)}) turn={face.turn}, size={face.size}'
            dimg = _cv2_add_title(dimg, title)
        if plot_crop_face:
            crops = [face.crop_face for face in faces]

            # draw landmarsks on crops
            for crop_idx, crop in enumerate(crops):
                for idx_p, p in enumerate(faces[crop_idx].kps):
                    cv2.circle(crop, p, 1, landmarks_colors[idx_p], 1)
            # /draw landmarsks on crops

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

    def _get_coll_imgs(self, imgs_list, size, top=1, left=1, right=1):  # top=10, left=5, right=5
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
detector.prepare(ctx_id=0, det_thresh=0.7)  # 0.5

session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)

frame_selector_model_path = '/home/vid/hdd/projects/PycharmProjects/insightface/models/ConvNext_selector.onnx'
frame_selector_model = ort.InferenceSession(frame_selector_model_path, providers=['CUDAExecutionProvider'])
frame_selector_model_input_name = frame_selector_model.get_inputs()[0].name


class Person2:
    def __init__(self, path=None, full_img=None, face=None, embedding=None, label='Unknown', color=(255, 0, 0),
                 change_brightness=False, show=False):
        self.color = color
        self.label = label
        self.path = path
        if full_img is not None:
            self.full_img = full_img
        if self.path is not None and full_img is None:
            self.full_img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if embedding is None:
            if face is None:
                face = detector.get(img=self.full_img,
                                    # use_roi=(30, 10, 20, 28),  # how to change?
                                    min_face_size=(50, 50),  # how to change?
                                    )[0]
            crop_face, face.kps, self.turn = norm_crop_self(full_img, face.kps, show=show)
            if change_brightness:
                self.crop_face = brightness_changer(crop_face, etalon=bright_etalon)
            else:
                self.crop_face = crop_face

            embedding = recognator.get(self.crop_face, show=show)
        self.embedding = embedding
        self.face = face

    def get_label(self, persons,
                  threshold=0.7, metric='cosine',
                  turn_bias=0, use_nn=False, limits=None,
                  show=False):
        # TODO: add nn bad img filter
        dists = []
        for person in persons:
            dist = cdist(self.embedding, person.embedding, metric=metric)[0][0]
            dists.append(dist)
        who = np.argmin(dists)
        min_dist = round(dists[who], 5)
        self.etalon_path = persons[who].path
        if limits is not None and self.turn + turn_bias >= 0:
            print(220, self.face.kps)

            '''
            get_middle = lambda p1, p2: [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
        m_right = get_middle(landmark[0], landmark[3])
        m_lelf = get_middle(landmark[1], landmark[4])
        m_eye = get_middle(landmark[0], landmark[1])
        m_mouth = get_middle(landmark[3], landmark[4])

        centroid_face = [int((m_eye[0] + m_mouth[0]) / 2), int((m_lelf[1] + m_right[1]) / 2)]

        difX = int((m_lelf[0] - m_right[0]) / 100 * borderXp / 2)
        difY = int((m_mouth[1] - m_eye[1]) / 100 * borderYp / 2)  # /2 for half
        borderXp = (centroid_face[0] - difX, centroid_face[0] + difX)
        borderYp = (centroid_face[1] - difY, centroid_face[1] + difY)
            '''

            pass
        if use_nn and self.turn + turn_bias >= 0:
            img_for_selector = preprocess_input(self.crop_face, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            selector_out = frame_selector_model.run(None, {frame_selector_model_input_name: img_for_selector})[0]
            good_frame = np.argmax(selector_out)  # bad = 0, good = 1
            if good_frame == 0:  # if "bad"
                self.turn = -99
        if dists[who] < threshold and self.turn + turn_bias >= 0:
            self.label = persons[who].label
            self.color = persons[who].color
            # self.etalon_path = persons[who].path
        if show:
            plt.imshow(self.crop_face)
            plt.title(f'"{self.label}": turn={self.turn} score={min_dist} (treshold={threshold})')
            plt.show()
        return min_dist


def norm_crop_self_old(img, landmark, image_size=112, mode='arcface', change_kpss_for_crop=True, show=False):
    def _get_nose_inside(countur, pt, bias=0):  # r_eye 0, l_eye 1, nose 2, r_mouth 3, l_mouth 4
        inside = cv2.pointPolygonTest(countur.astype(np.float32),
                                      pt.astype(np.float32),
                                      measureDist=True)
        return round(inside + bias, 2)  # positive (inside), negative (outside), or zero (on an edge) value

    # TODO: add brightness
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    if change_kpss_for_crop:
        landmark = np.array(list(map(
            lambda point: [
                M[0][0] * point[0] + M[0][1] * point[1] + M[0][2],  # change Ox
                M[1][0] * point[0] + M[1][1] * point[1] + M[1][2]  # change Oy
            ], landmark)))  # r_eye 0, l_eye 1, nose 2, r_mouth 3, l_mouth 4
    landmark = landmark.astype(np.uint8)
    without_nose = np.array([landmark[1], landmark[4], landmark[3], landmark[0]])
    nose_inside = _get_nose_inside(without_nose, landmark[2], bias=0)
    if show:
        cimg = warped.copy()
        for idx_p, p in enumerate(landmark):
            cv2.circle(cimg, p, 1, landmarks_colors[idx_p], 1)

        src = np.zeros(cimg.shape).astype(np.uint8)
        nose_color = (0, 255, 0) if nose_inside >= 0 else (255, 0, 0)

        cv2.polylines(src, np.int32([without_nose]), True, 255, 1)
        cv2.circle(src, landmark[2], 1, nose_color, 1)

        concat = np.concatenate((cimg, src.astype(int)), axis=1)
        plt.imshow(concat)
        plt.title(f'"{nose_inside}" nose_inside face')
        plt.show()
    return warped, landmark, nose_inside


def norm_crop_self(img, landmark, image_size=112, mode='arcface', change_kpss_for_crop=True, show=False):
    borderXp, borderYp = 100, 75

    def _get_nose_inside(countur, pt, bias=0):  # r_eye 0, l_eye 1, nose 2, r_mouth 3, l_mouth 4
        inside = cv2.pointPolygonTest(countur.astype(np.float32),
                                      pt.astype(np.float32),
                                      measureDist=True)
        return round(inside + bias, 2)  # positive (inside), negative (outside), or zero (on an edge) value

    # TODO: add brightness
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    if change_kpss_for_crop:
        landmark = np.array(list(map(
            lambda point: [
                M[0][0] * point[0] + M[0][1] * point[1] + M[0][2],  # change Ox
                M[1][0] * point[0] + M[1][1] * point[1] + M[1][2]  # change Oy
            ], landmark)))  # r_eye 0, l_eye 1, nose 2, r_mouth 3, l_mouth 4
    landmark = landmark.astype(np.uint8)
    eye_mouth_countur = np.array([landmark[1], landmark[4], landmark[3], landmark[0]])
    nose_inside = _get_nose_inside(eye_mouth_countur, landmark[2], bias=0)
    if show:
        cimg = warped.copy()
        for idx_p, p in enumerate(landmark):
            cv2.circle(cimg, p, 1, landmarks_colors[idx_p], 1)

        only_landmarsk = np.zeros(cimg.shape).astype(np.uint8)
        nose_color = (0, 255, 0) if nose_inside >= 0 else (255, 0, 0)

        get_middle = lambda p1, p2: [int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)]
        m_right = get_middle(landmark[0], landmark[3])
        m_lelf = get_middle(landmark[1], landmark[4])
        m_eye = get_middle(landmark[0], landmark[1])
        m_mouth = get_middle(landmark[3], landmark[4])

        centroid_face = [int((m_eye[0] + m_mouth[0]) / 2), int((m_lelf[1] + m_right[1]) / 2)]

        difX = int((m_lelf[0] - m_right[0]) / 100 * borderXp / 2)
        difY = int((m_mouth[1] - m_eye[1]) / 100 * borderYp / 2)  # /2 for half
        borderXp = (centroid_face[0] - difX, centroid_face[0] + difX)
        borderYp = (centroid_face[1] - difY, centroid_face[1] + difY)

        if nose_inside >= 0 and not (
                borderXp[0] <= landmark[2][0] <= borderXp[1] and borderYp[0] <= landmark[2][1] <= borderYp[1]):
            nose_color = (255, 255, 0)
            nose_inside = -50

        cv2.polylines(only_landmarsk, np.int32([eye_mouth_countur]), isClosed=True, color=(255, 255, 255), thickness=1)
        cv2.circle(only_landmarsk, landmark[2], 1, nose_color, 1)
        cv2.circle(only_landmarsk, centroid_face, 1, (255, 255, 255), 1)

        concat = np.concatenate((cimg, only_landmarsk.astype(int)), axis=1)
        plt.imshow(concat)
        plt.title(f'"{nose_inside}" nose_inside face')
        plt.show()
    return warped, landmark, nose_inside


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


def brightness_changer(img, etalon=None, diff=None, show=False):  # etalon=150
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
        plt.imshow(vis)
        plt.title(f'before {orig_br}:after ~{etalon}')
        plt.show()
    return final_img


def get_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return int(np.mean(v))


def preprocess_input(img, mean=None, std=None, input_space="RGB", size=(112, 112)):
    max_pixel_value = 255.0
    if input_space == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resizeimg = cv2.resize(img, size)

    img = resizeimg.astype(np.float32)
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value
        img -= mean

    if std is not None:
        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)
        img *= denominator

    img = np.moveaxis(img, -1, 0)
    img = img[np.newaxis, :, :, :]
    return img
