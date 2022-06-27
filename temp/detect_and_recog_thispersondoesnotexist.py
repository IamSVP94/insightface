import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from self_src.utils import Person, detector, PARENT_DIR, turnmetric, persons_list_from_csv, \
    get_imgs_thispersondoesnotexist

recog_tresh = 0.6
n_persons = 9999999

out_dir = Path('/home/vid/hdd/projects/PycharmProjects/insightface/temp/faces_crops/fromgoodtobadfaces/')
out_dir.mkdir(parents=True, exist_ok=True)

new_img_dir_path = PARENT_DIR / 'temp' / f'2706_newnew_turn={turnmetric}_recog_tresh={recog_tresh}'
print(f'save to {new_img_dir_path}')

# TODO: add new class for img (orig, face, norm_face, with_bbox, ...)

df_path = f'/home/vid/hdd/projects/PycharmProjects/insightface/temp/officeonly.csv'
df_persons = pd.read_csv(df_path, index_col=0)
all_persons = persons_list_from_csv(df_path)

persons_info = dict()
if __name__ == '__main__':
    p_bar = tqdm(range(n_persons), colour='green')
    for i in p_bar:
        img = get_imgs_thispersondoesnotexist(colors='RGB', show=False)[0]

        faces = detector.get(img)
        whoes = []
        for face_idx, face in enumerate(faces):
            emb = np.expand_dims(face['embedding'], axis=0)
            box = face.bbox.astype(np.int32)
            crop_face = img[box[1]:box[3], box[0]:box[2]]

            try:
                unknown = Person(img=crop_face, change_brightness=True, align=True, kps=face.kps, show=False)
            except cv2.error:
                continue

            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, show=False)
            whoes.append((unknown.label, round(near_dist, 4), unknown.color))

            face.brightness = unknown.brightness
            face.label = unknown.label
            face.rec_score = near_dist

            if len(faces) == len(whoes):
                for face in faces:
                    if face.label in persons_info:
                        persons_info[face.label].append(face.rec_score)
                    else:
                        persons_info[face.label] = [face.rec_score]

                    new_ready_path = new_img_dir_path / face.label / f'{i}.jpg'
                    new_ready_path.parent.mkdir(exist_ok=True, parents=True)

                    dimg = detector.draw_on(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        faces,
                        whoes=whoes,
                        show_kps=True,
                        show=False
                    )

                    imgslist = [dimg]

                    if face.label != 'Unknown':
                        etalon_path = df_persons.loc[unknown.label, 'path']
                        etalon = cv2.imread(str(etalon_path))
                        imgslist.append(etalon)

                    max_h = max([i.shape[0] for i in imgslist])
                    left, right = 1, 1
                    for idx, i in enumerate(imgslist):
                        curr_h = i.shape[0]
                        top_curr = int((max_h - curr_h) / 2)
                        bottom_curr = max_h - curr_h - top_curr
                        imgslist[idx] = cv2.copyMakeBorder(i, top_curr, bottom_curr, left, right, cv2.BORDER_CONSTANT)

                    vis = np.concatenate(imgslist, axis=1)
                    cv2.imwrite(str(new_ready_path), vis)

                    if face.label != 'Unknown':
                        print(new_ready_path)
                        exit()
