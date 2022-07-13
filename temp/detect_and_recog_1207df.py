import cv2
import random
from tqdm import tqdm
from pathlib import Path
from self_src.utils_new import detector, PARENT_DIR, Person2, persons_list_from_csv, recognator

recog_tresh = 0.6
new_output_dir_path = PARENT_DIR / 'temp' / f'cooler_1307_frames2_recog_tresh={recog_tresh}'

new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

all_persons = persons_list_from_csv(PARENT_DIR / 'temp/n=5465_native.csv')

# all_persons = []
# thispersondoesnotexist_etalons_dir = PARENT_DIR / 'temp/1207_site_recog_tresh=0.65/thispersondoesnotexist'
# tpdne_imgs = list(thispersondoesnotexist_etalons_dir.glob('*.jpg'))
# tpdne_pbar = tqdm(tpdne_imgs, leave=False, desc=f'tpdne processing', colour='yellow')
# for tpdne_path in tpdne_pbar:
#     tpdne_img = cv2.cvtColor(cv2.imread(str(tpdne_path)), cv2.COLOR_BGR2RGB)
#     tpdne_embedding = recognator.get(tpdne_img)
#     tpdne_etalon = Person2(path=tpdne_path, full_img=tpdne_img, label=tpdne_path.stem, embedding=tpdne_embedding)
#     all_persons.append(tpdne_etalon)

min_score = 999
min_score_img_name = ''
if __name__ == '__main__':
    DATASET_DIRS = [
        # '/home/vid/Downloads/datasets/lfw/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames8/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames7/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames6/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames5/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames4/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames3/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames2/',
        # '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames/',
        '/home/vid/Pictures/mpv/office_cooler_persons/',
    ]
    imgs = []
    for dir in DATASET_DIRS:
        for format in ['jpg', 'png', 'jpeg']:
            imgs.extend(Path(dir).glob(f'**/*.{format}'))
    # random.seed(2)
    # random.shuffle(imgs)

    p_bar = tqdm(imgs, colour='green', leave=False)
    for img_idx, img_path in enumerate(p_bar):
        p_bar.set_description(f'{img_path}')
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        label = img_path.parts[-2]
        faces = detector.get(img=img, max_num=2,
                             # use_roi=(30, 10, 15, 15),
                             min_face_size=(50, 50),
                             )
        if len(faces) == 0:  # need work only with face
            continue
        for face in faces:
            unknown = Person2(full_img=img, face=face)
            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, turn_bias=0, show=False)

            # face.brightness = unknown.brightness
            face.turn = unknown.turn
            face.crop_face = unknown.crop_face
            face.label = unknown.label
            face.color = unknown.color
            face.etalon_path = unknown.etalon_path
            face.rec_score = near_dist
            if face.rec_score < min_score:
                min_score = face.rec_score
                min_score_img_name = img_path

        dimg = detector.draw_on(img, faces, plot_roi=True, plot_crop_face=True, plot_etalon=True, show=False)

        save = True
        # for face in faces:
        #     if face.label not in ['Alena', 'VladislavV', 'Sergey_M', 'Evgeniy', 'Andrey',
        #                           'Dmitryi', 'Vladislav', 'Denis', 'Anton', 'Farid',
        #                           'Dmitryi_Base', 'Victor', 'Sergei', 'Alexandr', 'Anna',
        #                           'Unknown',
        #                           ]:
        #         save = False
        #         break

        if save:
            new_suffix = f'{[(face.label, face.rec_score) for face in faces]}.jpg'
            new_path = new_output_dir_path / f'{img_path.stem}_{new_suffix}'
            cv2.imwrite(str(new_path), cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB))

        # if img_idx > 10:
        #     exit()

    print(min_score, min_score_img_name)
