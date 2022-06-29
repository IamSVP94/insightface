import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from self_src.utils import Person, detector, PARENT_DIR, turnmetric, persons_list_from_csv, \
    get_imgs_thispersondoesnotexist

recog_tresh = 0.6
n_persons = 9999999

new_img_dir_path = PARENT_DIR / 'temp' / f'2806_newnew_turn={turnmetric}_recog_tresh={recog_tresh}'
print(f'save to {new_img_dir_path}')

# TODO: add new class for img (orig, face, norm_face, with_bbox, ...)

# df_path = f'/home/vid/hdd/projects/PycharmProjects/insightface/temp/officeonly.csv'
df_path = f'/home/vid/hdd/projects/PycharmProjects/insightface/temp/full2106new_persons=5732_bright_etalon=150_embeddings.csv'
df_persons = pd.read_csv(df_path, index_col=0)
all_persons = persons_list_from_csv(df_path)

persons_info = dict()
if __name__ == '__main__':
    p_bar = tqdm(range(n_persons), colour='green')
    for i in p_bar:
        try:
            img = get_imgs_thispersondoesnotexist(colors='RGB', show=False)[0]
        except cv2.error:
            continue

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

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    dimg = detector.draw_on(
                        img,
                        faces,
                        whoes=whoes,
                        show_kps=True,
                        show=False
                    )

                    imgslist = [dimg]

                    if face.label != 'Unknown':

                        new_ready_path = new_img_dir_path / face.label / f'{i}.jpg'
                        new_ready_path.parent.mkdir(exist_ok=True, parents=True)
                        new_ready_orig_path = new_img_dir_path / 'original' / f'{i}.jpg'
                        new_ready_orig_path.parent.mkdir(exist_ok=True, parents=True)
                        cv2.imwrite(str(new_ready_orig_path), img)

                        etalon_path = df_persons.loc[unknown.label, 'path']
                        etalon = cv2.imread(str(etalon_path))
                        imgslist.append(etalon)

                        max_h = max([i.shape[0] for i in imgslist])
                        left, right = 1, 1
                        for idx, i in enumerate(imgslist):
                            curr_h = i.shape[0]
                            top_curr = int((max_h - curr_h) / 2)
                            bottom_curr = max_h - curr_h - top_curr
                            imgslist[idx] = cv2.copyMakeBorder(i, top_curr, bottom_curr, left, right,
                                                               cv2.BORDER_CONSTANT)

                        vis = np.concatenate(imgslist, axis=1)
                        cv2.imwrite(str(new_ready_path), vis)

                        print('\n', new_ready_path, face.rec_score)
