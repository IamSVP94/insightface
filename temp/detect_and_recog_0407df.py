import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from insightface.model_zoo.model_zoo import PickableInferenceSession
from insightface.utils.face_align import norm_crop
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path
from self_src.utils import ArcFaceONNXVL
from self_src.utils_new import detector, PARENT_DIR

session = PickableInferenceSession(
    model_path=str(PARENT_DIR / 'models/IResNet100l.onnx'),
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
recognator = ArcFaceONNXVL(model_file=PARENT_DIR / 'models/IResNet100l.onnx', session=session)
recognator.prepare(ctx_id=0)

# df = pd.read_csv(PARENT_DIR / 'temp/n=19_office2.csv', index_col=0)
df = pd.read_csv(PARENT_DIR / 'temp/n=849_native.csv', index_col=0)

etalon_embs = []
labels = []
for label, line in tqdm(df.iterrows(), total=len(df), colour='red', leave=False):
    emb = np.array([line.to_list()[1:]])
    etalon_embs.append(emb)
    labels.append(label)

if __name__ == '__main__':
    DATASET_DIRS = [
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames/',
        '/home/vid/Pictures/mpv/office_cooler_persons/',
    ]
    imgs = []
    for dir in DATASET_DIRS:
        imgs.extend(Path(dir).glob('**/*.jpg'))
    # random.seed(2)
    random.shuffle(imgs)

    p_bar = tqdm(imgs, colour='green', leave=False)
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        label = img_path.parts[-2]
        faces = detector.get(img=img,
                             # change_kpss_for_crop=False,
                             # use_roi=(30, 10, 20, 28),
                             # min_face_size=(112, 112),
                             )

        for face in faces:
            crop_face = norm_crop(img, face.kps, image_size=112, mode='arcface')
            emb = recognator.get(crop_face)

            dists_emb = []
            for idx, etalon_emb in enumerate(etalon_embs):
                dist = cdist(emb.reshape(1, -1), etalon_emb, metric='cosine')[0][0]
                dists_emb.append(dist)

            who = np.argmin(dists_emb)

            print(labels[who], dists_emb[who])

            plt.imshow(crop_face)
            plt.title(f'who={labels[who]}, score={dists_emb[who]}')
            plt.show()

        # if faces:
        #     for face_idx, face in enumerate(faces):
        #         emb = np.expand_dims(face['embedding'], axis=0)
        if img_idx > 20:
            exit()
