import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from self_src.utils import Person, detector, PARENT_DIR, turnmetric, bright_etalon, persons_list_from_csv

recog_tresh = 0.6

new_img_dir_path = PARENT_DIR / 'temp' / f'office3_turn={turnmetric}_recog_tresh={recog_tresh}'
new_img_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_img_dir_path}')

# TODO: add new class for img (orig, face, norm_face, with_bbox, ...)

df_path = f'/home/psv/PycharmProjects/insightface/temp/all_persons=5723_bright_etalon=150_embeddings.csv'
df_persons = pd.read_csv(df_path, index_col=0)
all_persons = persons_list_from_csv(df_path)

persons_info = dict()
if __name__ == '__main__':
    # DATASET_DIR = Path('/home/psv/PycharmProjects/insightface/temp/out/frames/')
    DATASET_DIR = Path('/home/psv/Pictures/mpv/office_cooler_persons/')
    imgs = list(DATASET_DIR.glob('**/*.jpg'))

    p_bar = tqdm(imgs, colour='green')
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.imread(str(img_path))

        faces = detector.get(img)
        whoes = []
        for face_idx, face in enumerate(faces):
            emb = np.expand_dims(face['embedding'], axis=0)
            box = face.bbox.astype(np.int32)
            crop_face = img[box[1]:box[3], box[0]:box[2]]
            try:
                crop_face = Person.change_brightness(crop_face, etalon=bright_etalon)  # bug
            except cv2.error:
                continue

            unknown = Person(img=crop_face, change_brightness=True, align=True, kps=face.kps, show=False)
            face.brightness = unknown.brightness

            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, face=face, show=False)
            whoes.append((unknown.label, round(near_dist, 4), unknown.color))

            face.label = unknown.label
            face.rec_score = near_dist

        if len(faces) == len(whoes):
            for face in faces:
                if face.label != 'Unknown':
                    if face.label in persons_info:
                        persons_info[face.label].append(face.rec_score)
                    else:
                        persons_info[face.label] = [face.rec_score]
                    new_ready_path = new_img_dir_path / face.label / f'{img_path.name}'
                    new_ready_path.parent.mkdir(exist_ok=True, parents=True)
                    dimg = detector.draw_on(img, faces, whoes=whoes, show_kps=True, show=False)

                    etalon_img = cv2.imread(str(df_persons.loc[face.label].path))
                    etalon_h, etalon_w, _ = etalon_img.shape
                    curr_h, curr_w, _ = dimg.shape
                    max_h, max_w = max(etalon_h, curr_h), max(etalon_w, curr_w)

                    top_curr = int((max_h - curr_h) / 2)
                    bottom_curr = max_h - curr_h - top_curr
                    top_etal = int((max_h - etalon_h) / 2)
                    bottom_etal = max_h - etalon_h - top_etal
                    left, right = 10, 10

                    etalon_img = cv2.copyMakeBorder(etalon_img, top_etal, bottom_etal, left, right, cv2.BORDER_CONSTANT)
                    vis = np.concatenate((dimg, etalon_img), axis=1)

                    cv2.imwrite(str(new_ready_path), vis)
        # if img_idx > 100:
        # exit()
        # break
# '''

df = pd.DataFrame(columns=['mean', 'disp', 'n'])
for person, rec_scores in persons_info.items():
    mean = np.mean(rec_scores)
    disp = np.var(rec_scores)
    df.loc[person, :] = [mean, disp, len(rec_scores)]

df.to_csv(new_img_dir_path / f'mean_disp_office3.csv')
