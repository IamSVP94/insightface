from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

model_pack_name = 'buffalo_m'
affine = False
IMG_DIR = Path('/home/psv/file/project/recognition-dataset/2004_img/')
# IMG_DIR = Path('/home/psv/Downloads/archive/images')
if affine:
    NEW_DIR = IMG_DIR.parent / f'img_faces_affine_{model_pack_name}'
else:
    NEW_DIR = IMG_DIR.parent / f'img_faces_{model_pack_name}'

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   name=model_pack_name,
                   # allowed_modules=['detection', 'recognition'],  # enable detection model only
                   )
app.prepare(ctx_id=0, det_size=(640, 640))

NEW_DIR.mkdir(parents=True, exist_ok=True)
img_paths = []
for format in ['png', 'jpg', 'jpeg']:
    img_paths.extend(list(IMG_DIR.glob(f'*.{format}')))
p_bar = tqdm(img_paths)
for idx, img_path in enumerate(p_bar):
    p_bar.set_description(f'{img_path} processing now{"." * (idx % 3 + 1)}')
    img = cv2.imread(str(img_path))
    # cv2.imshow(f"img", img)  # show img after preprocessing
    # cv2.waitKey()
    if affine:
        srcTri = np.array([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]]).astype(np.float32)
        dstTri = np.array([[0, img.shape[1] * 0.33], [img.shape[1] * 0.85, img.shape[0] * 0.25],
                           [img.shape[1] * 0.15, img.shape[0] * 0.7]]).astype(np.float32)
        warp_mat = cv2.getAffineTransform(srcTri, dstTri)
        warp_dst = cv2.warpAffine(img, warp_mat, (img.shape[1], img.shape[0]))
        # Rotating the image after Warp
        # center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
        # angle = -5
        # scale = 0.9
        # rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        # warp_rotate_dst = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
        img = warp_dst

    # print(44, img.shape)
    faces = app.get(img)
    # print(46, faces)
    rimg = app.draw_on(img, faces)
    cv2.imshow(f"rimg", rimg)  # show img after preprocessing
    cv2.waitKey()
    # exit()
    new_ing_path = NEW_DIR / f'{idx}_{img_path.name}'
    cv2.imwrite(str(new_ing_path), rimg)
    if idx > 100:
        print(new_ing_path.parent)
        break
