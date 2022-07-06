import cv2
import numpy as np
import pandas as pd
import requests
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.model_zoo import ArcFaceONNX
from insightface.model_zoo.model_zoo import PickableInferenceSession
from insightface.utils import face_align
from insightface.utils.face_align import norm_crop, estimate_norm
from scipy.spatial.distance import cdist
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math
import onnxruntime as ort

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
                self.use_roi = True
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
    def __init__(self, path=None, full_img=None, face=None, embedding=None, label='Unknown', color=(0, 0, 255),
                 change_brightness=True, show=False):
        self.color = color
        self.label = label
        if full_img is not None:
            self.full_img = full_img
        elif path is not None:
            self.full_img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        if embedding is None:
            if face is None:
                face = detector.get(img=self.full_img,
                                    # use_roi=(30, 10, 20, 28),  # how to change?
                                    min_face_size=(112, 112),  # how to change?
                                    )[0]
            self.crop_face, face.kps = norm_crop_self(full_img, face.kps, show=False)
            embedding = recognator.get(self.crop_face)
        self.embedding = embedding
        self.face = face


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
