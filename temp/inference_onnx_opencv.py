import argparse
from pathlib import Path
import cv2
import onnx
import onnxruntime
import numpy as np
from tqdm import tqdm

print(cv2.__version__)


def distance2bbox(points, distance, max_shape=None):
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


def distance2kps(points, distance, max_shape=None):
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


defaults = {
    'output_dir': 'onnx_output',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_l/new_det_10g.onnx',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_sc/det_500m.onnx',
    # 'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_l/w600k_r50.onnx',
    'onnx_model': '/home/psv/PycharmProjects/insightface/models/buffalo_l/640x640_det_10g.onnx',
}

parser = argparse.ArgumentParser(description='Image detection program')
parser.add_argument("--images_dir", type=str, required=True,
                    help="dir images for prediction (required)")
parser.add_argument("--output_dir", type=str, default=defaults['output_dir'],
                    help=f"dir for img predictions (default: '{defaults['output_dir']}')")
parser.add_argument("--onnx_model", type=str, default=defaults['onnx_model'],
                    help=f"*.onnx model path (default: '{defaults['onnx_model']}')")
args = parser.parse_args()

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Load the network
# onnx
# net = onnx.load(args.onnx_model)
# onnx.checker.check_model(net)
# exit()

# opencv
net = cv2.dnn.readNet(args.onnx_model)

# Get the output layer from model
layers = net.getLayerNames()
output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

images = list(Path(args.images_dir).glob('*.jpg'))
p_bar = tqdm(images, colour='green', leave=False)
for idx, img_path in enumerate(p_bar):
    scores_list = []
    bboxes_list = []
    kpss_list = []

    p_bar.set_description(f'{img_path} processing now {"." * (idx % 3 + 1)}')
    output_path = Path(args.output_dir, img_path.name)

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]

    # blob = cv2.dnn.blobFromImage(
    #     image=img,
    #     scalefactor=None,  # 0.00392
    # swapRB=True,
    # crop=False,
    # )
    blob = cv2.dnn.blobFromImage(image=img,
                                 scalefactor=1.0 / 128.0,
                                 size=(640, 640),
                                 mean=(127.5, 127.5, 127.5),
                                 swapRB=True,
                                 )  # from original

    # cv2.imshow(f"img from blob", cv2.dnn.imagesFromBlob(blob)[0])  # show original img
    # cv2.waitKey()
    print('blob', blob.shape, '\n')
    net.setInput(blob)
    net_outs = net.forward(output_layers)
    print('net_outs', type(net_outs), len(net_outs), [len(l) for l in net_outs], [l[0][0] for l in net_outs])

    # change form (as in original)
    # net_outs = np.asarray(net_outs, dtype=[object, np.matrix] ?])
    # net_outs = np.transpose(
    #     np.asarray(net_outs, dtype=np.matrix),
    #     axes=(0, 3, 6, 1, 4, 7, 2, 5, 8),
    # )
    net_outs = np.moveaxis(
        net_outs,
        # np.asarray(net_outs, dtype=np.matrix),
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        # [0, 3, 6, 1, 4, 7, 2, 5, 8],
    )
    print('net_outs', type(net_outs), len(net_outs), [len(l) for l in net_outs], [l[0][0] for l in net_outs])
    _, fmc, input_height, input_width = blob.shape
    _feat_stride_fpn = [8, 16, 32]  # from original ?
    _num_anchors = 2
    threshold = 0.5

    for idx, stride in enumerate(_feat_stride_fpn):
        print(idx, stride)
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        bbox_preds = bbox_preds * stride
        kps_preds = net_outs[idx + fmc * 2] * stride
        # print(scores.shape)
        # print(bbox_preds.shape)
        # print(kps_preds.shape)

        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)
        print('key', key)

        center_cache = dict()
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)

            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers] * _num_anchors, axis=1).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= threshold)[0]
        assert anchor_centers.shape[0] == bbox_preds.shape[0], (anchor_centers.shape, bbox_preds.shape)
        print('anchor_centers, bbox_preds', anchor_centers.shape, bbox_preds.shape)
        # bboxes = distance2bbox(anchor_centers, bbox_preds)
        # pos_scores = scores[pos_inds]
        # pos_bboxes = bboxes[pos_inds]
        # scores_list.append(pos_scores)
        # bboxes_list.append(pos_bboxes)
        #
        # kpss = distance2kps(anchor_centers, kps_preds)
        # kpss = kpss.reshape((kpss.shape[0], -1, 2))
        # pos_kpss = kpss[pos_inds]
        # kpss_list.append(pos_kpss)

    print(181, scores_list, bboxes_list, kpss_list)
    break

# TODO: 360x640 original
# TODO: del comments from /home/psv/.cache/pypoetry/virtualenvs/insightface-g_gNxq4w-py3.8/lib/python3.8/site-packages/insightface/model_zoo/retinaface.py
