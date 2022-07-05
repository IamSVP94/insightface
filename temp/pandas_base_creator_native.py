import random

import cv2
import pandas as pd
from insightface.model_zoo.model_zoo import PickableInferenceSession
from tqdm import tqdm
from pathlib import Path

from recognition.arcface_mxnet.common.face_align import norm_crop
from self_src.utils import PARENT_DIR, ArcFaceONNXVL, detector

session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)

cols = ['path']
cols.extend([i for i in range(0, 512)])
df = pd.DataFrame(columns=cols)

PERSONS_MAIN_DIR = Path('/home/vid/hdd/projects/PycharmProjects/insightface/temp/faces0407bylabels/')
etalons_path = list(PERSONS_MAIN_DIR.glob('**/*_best.jpg'))

p_bar = tqdm(etalons_path, colour='green', leave=False)
for img_idx, img_path in enumerate(p_bar):
    for_write = [img_path]
    label = img_path.parts[-2]
    p_bar.set_description(f'{label}')
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    emb = recognator.get(img, show=False)

    for_write.extend(emb.tolist()[0])
    df.loc[label, :] = for_write

OFFICE_PERSONS_MAIN_DIR = Path('/home/vid/hdd/file/project/recog_datasets/LABELED_FACES/LABELED_full/')
etalons_path = list(OFFICE_PERSONS_MAIN_DIR.glob('**/fas.jpg'))

p_bar = tqdm(etalons_path, colour='green', leave=False)
for img_idx, img_path in enumerate(p_bar):
    for_write = [img_path]
    label = img_path.parts[-2]
    p_bar.set_description(f'{label}')
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    faces = detector.get(img=img, max_num=1)

    for face in faces:
        crop_face = norm_crop(img, face.kps, image_size=112, mode='arcface')
        emb = recognator.get(crop_face, show=False)

    for_write.extend(emb.tolist()[0])
    df.loc[label, :] = for_write

df.to_csv(f'n={len(df)}_native.csv')
