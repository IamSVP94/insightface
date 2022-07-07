import random

import cv2
from tqdm import tqdm
from pathlib import Path
from self_src.utils_new import detector, PARENT_DIR, Person2, persons_list_from_csv

recog_tresh = 10.4

# new_output_dir_path = PARENT_DIR / 'temp' / f'without_0707_office_recog_tresh={recog_tresh}'
new_output_dir_path = PARENT_DIR / 'temp' / f'all_0707_recog_tresh={recog_tresh}'
new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

# all_persons = persons_list_from_csv(PARENT_DIR / 'temp/n=834_native_without_employee.csv')
all_persons = persons_list_from_csv(PARENT_DIR / 'temp/n=849_native.csv')

min_score = 999
min_score_img_name = ''
if __name__ == '__main__':
    DATASET_DIRS = [
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames8/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames7/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames6/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames5/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames4/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames3/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames2/',
        '/home/vid/hdd/projects/PycharmProjects/insightface/temp/out/frames/',
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
        faces = detector.get(img=img,
                             use_roi=(30, 10, 20, 28),
                             min_face_size=(112, 112),
                             )
        if len(faces) == 0:  # need work only with face
            continue
        for face in faces:
            unknown = Person2(full_img=img, face=face)
            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, show=False)

            # face.brightness = unknown.brightness
            # face.turn = unknown.turn
            face.label = unknown.label
            face.color = unknown.color
            face.rec_score = near_dist
            if face.rec_score < min_score:
                min_score = face.rec_score
                min_score_img_name = img_path

        dimg = detector.draw_on(img, faces, plot_roi=True, show=False)
        # cv2.imwrite(str(new_output_dir_path / f'{img_path.name}'), cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR))

        new_suffix = f'{[face.label for face in faces]}.jpg'
        cv2.imwrite(str(new_output_dir_path / f'{img_path.stem}_{new_suffix}'), cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR))

    print(min_score, min_score_img_name)
