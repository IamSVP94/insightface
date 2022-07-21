import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
from self_src.utils_new import detector, PARENT_DIR, Person2, persons_list_from_csv, brightness_changer, recognator, \
    get_random_color

recog_tresh = 0.6
det_size = 1920  # 640  # 1280 баланс?
detector.prepare(ctx_id=0, det_thresh=0.7, det_size=(det_size, det_size))  # 0.5

new_output_dir_path = PARENT_DIR / 'temp' / f'frames1907_2107_recog_tresh={recog_tresh}_{det_size}'

new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

# all_persons = persons_list_from_csv(PARENT_DIR / 'temp/n=10414_native.csv')
emb0 = np.ones(shape=(1, 512))
img0 = np.ones(shape=(112, 112, 3))
all_persons = [Person2(full_img=img0, label='zero', color=get_random_color(), embedding=emb0)]

person_idx = 0
if __name__ == '__main__':
    DATASET_DIRS = [
        # '/home/vid/hdd/projects/PycharmProjects/GFPGAN/o_result100/cmp/',
        # '/home/vid/hdd/projects/PycharmProjects/GFPGAN/o_result4000/cmp/',
        # '/home/vid/Downloads/datasets/zavod/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames4/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames2/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames3/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames7/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames8/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames1807/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames1907/',
        # '/home/vid/Downloads/datasets/office_cooler_persons/',
        # '/home/vid/Downloads/datasets/office_cooler_persons/',
    ]
    imgs = []
    for dir in DATASET_DIRS:
        for format in ['jpg', 'png', 'jpeg']:
            imgs.extend(Path(dir).glob(f'**/*.{format}'))
    # random.seed(2)
    # random.shuffle(imgs)
    # imgs = imgs[:200]

    p_bar = tqdm(sorted(imgs), colour='green', leave=False)
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        frame = cv2.imread(str(img_path))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detector.get(img=img,
                             # max_num=9,  # del this!
                             # use_roi=(15, 0, 5, 5),  # top, bottom, left, right
                             # min_face_size=(45, 45),
                             )
        if len(faces) == 0:  # need work only with face
            continue
        # cv2.imwrite(str(f'/home/vid/Downloads/zavod2/faceimg/{img_path.name}'), frame)  # save face frame
        for face in faces:
            unknown = Person2(full_img=img, face=face, show=False)
            near_dist = unknown.get_label(all_persons, threshold=recog_tresh,
                                          turn_bias=3, limits=(100, 75), use_nn=False,
                                          show=False,
                                          )

            # face.brightness = unknown.brightness
            face.turn = round(unknown.turn, 1)
            face.crop_face = unknown.crop_face

            face.label = unknown.label
            face.color = unknown.color
            face.rec_score = near_dist

            face.etalon_path = unknown.etalon_path
            face.etalon_crop = unknown.etalon_crop

            if face.label == 'Unknown' and face.turn >= 0:  # for adding to etalons (for reidentification)
                unknown.label = f'person_{person_idx}'
                unknown.color = get_random_color()
                person_idx += 1
                all_persons.append(unknown)
                print(f'appended {person_idx}')

        dimg = detector.draw_on(img, faces, plot_roi=True, plot_crop_face=True, plot_etalon=True, show=False)
        new_suffix = f'{[(face.label, face.rec_score) for face in faces]}.jpg'
        new_path = new_output_dir_path / f'{img_path.stem}_{det_size}_{new_suffix}'
        cv2.imwrite(str(new_path), cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB))

        # if img_idx > 20:
        #     exit()
