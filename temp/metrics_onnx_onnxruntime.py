import argparse
from pathlib import Path
import cv2
import onnxruntime as ort
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm as l2norm
import time
from metrics import get_GT_bbox, find_iou_for_all_boxes
from metrics import mean_average_precision as mAP

print(f"inference with {ort.get_device()}")

defaults = {
    'output_dir': None,
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/recognition/retinaface.h5',
    'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_l/det_10g.onnx',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_l/640x640_det_10g.onnx',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/antelopev2/scrfd_10g_bnkps.onnx',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_sc/det_500m.onnx',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_m/det_2.5g.onnx',
}

parser = argparse.ArgumentParser(description='Image detection program')
parser.add_argument("--images_dir", type=str, required=True,
                    help="dir images for prediction (required)")
parser.add_argument("--output_dir", type=str, default=defaults['output_dir'],
                    help=f"dir for img predictions (default: '{defaults['output_dir']}')")
parser.add_argument("--onnx_model", type=str, default=defaults['onnx_model'],
                    help=f"*.onnx model path (default: '{defaults['onnx_model']}')")
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = f'{Path(args.onnx_model).stem}_onnx'
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

df_path = Path(args.output_dir).parent / f'{Path(args.onnx_model).stem}.csv'


class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding, ord=None)

    @property
    def normed_embedding(self):  # need for recognition (unnecessary now)
        if self.embedding is None:
            return None
        print(self.embedding)
        print(self.embedding_norm)
        return self.embedding / self.embedding_norm

    @property
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'


class ONNXRunner:
    def __init__(self, det_thresh=0.5, det_size=(640, 640)):
        self.session = ort.InferenceSession(
            args.onnx_model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.input_mean = 127.5  # from original
        self.input_std = 128.0  # from original
        self.fmc = 3  # from original
        self._feat_stride_fpn = [8, 16, 32]  # from original
        self._num_anchors = 2  # from original
        self.center_cache = dict()
        self.det_scale = 1

        self.original_h = None  # for img orig size
        self.original_w = None  # for img orig size
        # self.yolobbox = []  # for YOLO format

    def _img_preprocessing(self, img, input_size=None):
        input_size = self.det_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        self.det_scale = float(new_height) / img.shape[0]

        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        return det_img

    def _get_blob(self, img):
        blob = cv2.dnn.blobFromImage(
            image=img,
            scalefactor=1.0 / self.input_std,
            size=self.det_size,
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )  # from original
        return blob

    def _distance2bbox(self, points, distance, mode='default', max_shape=None):  # TODO: add YOLO format
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def _distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def nms(self, dets):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.det_thresh)[0]
            order = order[inds + 1]

        return keep

    def forward(self, img):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        blob = self._get_blob(img)
        net_outs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        # print('net_outs', type(net_outs), len(net_outs), [len(l) for l in net_outs], [l[0][0] for l in net_outs])

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outs[idx + self.fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = self._distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss = self._distance2kps(anchor_centers, kps_preds)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, max_num=0, metric='default'):
        det_img = onnxrunner._img_preprocessing(img)
        scores_list, bboxes_list, kpss_list = onnxrunner.forward(det_img)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / onnxrunner.det_scale

        kpss = np.vstack(kpss_list) / onnxrunner.det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = onnxrunner.nms(pre_det)
        det = pre_det[keep, :]
        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def get(self, img):
        self.yolobbox = []  # for YOLO format
        self.original_h, self.original_w, _ = img.shape
        bboxes, kpss = onnxrunner.detect(img)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            # if kpss is not None:  # TODO: add kpss
            #     kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            ret.append(face)
        return ret

    @staticmethod
    def bbox2yolobbox(img, box):  # box = (left, top, right, bottom)
        original_h, original_w, _ = img.shape
        xmin, ymin, xmax, ymax = box

        dw = 1. / original_w
        dh = 1. / original_h

        x = (xmin + xmax) / 2.0
        y = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        x = round(x * dw, 6)
        y = round(y * dh, 6)
        w = round(w * dw, 6)
        h = round(h * dh, 6)
        yolo_format = (0, x, y, w, h)  # 0 because only 1 class
        return yolo_format

    @staticmethod
    def yolobbox2bbox(img, bbox):
        cl, x, y, w, h = bbox
        original_h, original_w, _ = img.shape

        xmin = original_w * (x - w / 2)
        ymin = original_h * (y - h / 2)
        xmax = original_w * (x + w / 2)
        ymax = original_h * (y + h / 2)
        return [cl, int(xmin), int(ymin), int(xmax), int(ymax)]

    def draw_on(self, img, faces):
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int32)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int32)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img=dimg, center=(kps[l][0], kps[l][1]), radius=1, color=color, thickness=2)
            # if face.gender is not None and face.age is not None:  # TODO: rewrite for recognition
            #     cv2.putText(img=dimg,
            #                 text=f"{face.sex}, {face.age}",
            #                 org=(box[0] - 1, box[1] - 4),
            #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
            #                 fontScale=0.7,
            #                 color=(0, 255, 0),
            #                 thickness=1,
            #                 lineType=None,
            #                 )
        return dimg


onnxrunner = ONNXRunner(det_thresh=0.5)

images = []
for format in ['png', 'jpg']:
    images.extend(list(Path(args.images_dir).glob(f'*.{format}')))
p_bar = tqdm(images, colour='green', leave=True)
metrics = dict()
df = pd.DataFrame(columns=['clear_iou', 'full_iou', 'map', 'fp', 'fn', 'inference_duration'])
for idx, img_path in enumerate(p_bar):
    # img_path = Path('/home/psv/file/project/161-Росатом-Инциденты по охране труда/pseudo_labeled/13_00014.jpg')

    p_bar.set_description(f'{img_path} processing now{"." * (idx % 3 + 1)}')
    output_path = Path(args.output_dir, img_path.name)

    img = cv2.imread(str(img_path))
    # cv2.imshow(f"img", img)  # show img after preprocessing
    # cv2.waitKey()

    start = time.perf_counter_ns()  # ns (e-9)
    faces = onnxrunner.get(img)
    duration_mcs = (time.perf_counter_ns() - start) / 1000  # mcs (e-6)

    PRED_yolo = []
    for idx, face in enumerate(faces):
        yolo_format = ONNXRunner.bbox2yolobbox(img, face['bbox'])
        PRED_yolo.append(list(yolo_format))
    PRED = list(map(lambda bbox: ONNXRunner.yolobbox2bbox(img, bbox), PRED_yolo))

    txt_path = img_path.with_suffix('.txt')
    GT_yolo = get_GT_bbox(img_path.with_suffix('.txt'))
    GT = list(map(lambda bbox: ONNXRunner.yolobbox2bbox(img, bbox), GT_yolo))

    marked_img = onnxrunner.draw_on(img, faces)

    count_not_find_face, count_fp, iou_metric = find_iou_for_all_boxes(GT, PRED)
    map_metric = mAP(PRED, GT, box_format="corners")

    iou_metric = torch.Tensor(iou_metric)
    clear_iou_metric = torch.Tensor([metric for metric in iou_metric if metric])

    if PRED == GT:
        mean_full_iou_metric = 1
        mean_clear_iou_metric = 1
    elif (len(PRED) == 0 and GT) or (len(GT) == 0 and PRED):
        mean_full_iou_metric = 0
        mean_clear_iou_metric = 0
    else:
        mean_full_iou_metric = float(torch.mean(iou_metric))
        mean_clear_iou_metric = float(torch.mean(clear_iou_metric))

    cv2.imwrite(str(Path(args.output_dir, f'{img_path.name}')), marked_img)
    df.loc[img_path.stem, :] = [
        mean_clear_iou_metric,
        mean_full_iou_metric,
        map_metric,
        count_fp,
        count_not_find_face,
        duration_mcs
    ]

df.to_csv(df_path)
with open(df_path.parent / f'{Path(args.onnx_model).stem}.txt', 'w') as txt:
    txt.writelines([
        f'mean iou\t\t\t{df["full_iou"].mean()}\n',
        f'mean clean iou\t\t{df["clear_iou"].mean()}\n',
        f'mean map\t\t\t{df["map"].mean()}\n',
        f'mean inference time\t{int(df["inference_duration"].mean())}\n',
        f'watch nvidia-smi\t',
    ])
# 515MiB видеопамять
