import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from self_src.utils import Person, detector, PARENT_DIR, turnmetric, bright_etalon, persons_list_from_csv

recog_tresh = 0.6

new_img_dir_path = PARENT_DIR / 'temp' / f'1548_turn={turnmetric}_recog_tresh={recog_tresh}'
new_img_dir_path.mkdir(exist_ok=True, parents=True)
print(f'\nsave to {new_img_dir_path}')

df_path = f'/home/psv/PycharmProjects/insightface/temp/full_persons=5728_bright_etalon=150_embeddings.csv'
df_persons = pd.read_csv(df_path, index_col=0)
all_persons = persons_list_from_csv(df_path)

# show_persons = True
# DIR = Path('/home/psv/file/project/recog_datasets/LABELED_FACES/LABELED_full/')
# person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', color=(0, 255, 0), makemask=True, show=show_persons)
# person2 = Person(path=DIR / 'Vladislav/fas.jpg', label='Vladislav', color=(255, 0, 0), makemask=True, show=show_persons)
# person3 = Person(path=DIR / 'Farid/fas.jpg', label='Farid', color=(180, 237, 140), makemask=True, show=show_persons)
# person4 = Person(path=DIR / 'Denis/fas.jpg', label='Denis', color=(80, 127, 200), makemask=True, show=show_persons)
# person5 = Person(path=DIR / 'Anton/fas.jpg', label='Anton', color=(213, 147, 138), makemask=True, show=show_persons)
# person6 = Person(path=DIR / 'Alexandr/fas.jpg', label='Alexandr', color=(131, 158, 101), makemask=True,
#                  show=show_persons)
# person7 = Person(path=DIR / 'Putin/fas.jpg', label='Putin', color=(113, 47, 38), makemask=True, show=show_persons)
# person8 = Person(path=DIR / 'Irina/fas.jpg', label='Irina', color=(31, 58, 61), makemask=True, show=show_persons)
# person9 = Person(path=DIR / 'Korzh/fas.jpg', label='Korzh', color=(137, 40, 80), makemask=True, show=show_persons)
# person10 = Person(path=DIR / 'Bruce/fas.jpg', label='Bruce', color=(180, 40, 137), makemask=True, show=show_persons)
# person11 = Person(path=DIR / 'Alena/fas.jpg', label='Alena', color=(108, 84, 173), makemask=True, show=show_persons)
# person12 = Person(path=DIR / 'Dmitryi/fas.jpg', label='Dmitryi', color=(0, 100, 0), makemask=True, show=show_persons)
# person13 = Person(path=DIR / 'VladislavV/fas.jpg', label='VladislavV', color=(100, 0, 100), makemask=True,
#                   show=show_persons)
# all_persons.extend([person1, person2, person3, person4, person5, person6, person7, person8, person9, person10,
#                     person11, person12, person13,
#                     ])

if __name__ == '__main__':
    DATASET_DIR = Path('/home/psv/Pictures/mpv/goods_cooler/')
    imgs = list(DATASET_DIR.glob('*.jpg'))

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
                crop_face = Person.change_brightness(crop_face, etalon=bright_etalon)
            except cv2.error:
                continue
            unknown = Person(img=crop_face, change_brightness=True, align=True, kps=face.kps, show=False)

            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, face=face, show=False)

            face.brightness = unknown.brightness
            face.label = unknown.label  # need this?
            face.rec_score = near_dist  # need this?

            if unknown.label != 'Unknown':
                print('\n', img_path, unknown.label)
            whoes.append((unknown.label, round(near_dist, 4), unknown.color))

        if len(faces) == len(whoes):
            dimg = detector.draw_on(img, faces, whoes=whoes, show_kps=False, show=False)
            cv2.imwrite(str(new_img_dir_path / f'{img_path.name}'), dimg)
        # if img_idx > 20:
        #     exit()
# '''
