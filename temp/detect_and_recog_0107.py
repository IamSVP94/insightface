import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from insightface.utils.face_align import norm_crop
from tqdm import tqdm
from pathlib import Path
# from self_src.utils import Person, detector, PARENT_DIR, turnmetric, bright_etalon, persons_list_from_csv
from self_src.utils_new import detector

persons_info = dict()
if __name__ == '__main__':

    DATASET_DIRS = [
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames/',
    ]
    imgs = []
    for dir in DATASET_DIRS:
        imgs.extend(Path(dir).glob('**/*.jpg'))
    random.seed(2)
    random.shuffle(imgs)

    p_bar = tqdm(imgs, colour='green')
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        print(img.shape)

        faces = detector.get(img=img,
                             change_kpss_for_crop=False,
                             # use_roi=(30, 10, 20, 28),
                             # min_face_size=(112, 112),
                             )

        # plt.imshow(img)
        # plt.show()

        print('len(faces)', len(faces))
        for face in faces:
            crop_face = norm_crop(img, face.kps, image_size=112, mode='arcface')

            # box = face.bbox.astype(np.int32)
            # crop_face1 = img[box[1]:box[3], box[0]:box[2]]

            plt.imshow(crop_face)
            plt.show()

        if faces:
            for face_idx, face in enumerate(faces):
                emb = np.expand_dims(face['embedding'], axis=0)