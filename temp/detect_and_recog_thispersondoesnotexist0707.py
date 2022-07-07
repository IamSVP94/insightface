import cv2
from datetime import datetime
from self_src.utils_new import Person2, detector, PARENT_DIR, persons_list_from_csv, get_imgs_thispersondoesnotexist

recog_tresh = 0.6

new_output_dir_path = PARENT_DIR / 'temp' / f'0707_site_recog_tresh={recog_tresh}'
new_output_dir_path.mkdir(exist_ok=True, parents=True)
print(f'save to {new_output_dir_path}')

all_persons = persons_list_from_csv(PARENT_DIR / 'temp/n=913_native.csv')

persons_info = dict()
if __name__ == '__main__':
    img_idx = 0
    while True:
        img = get_imgs_thispersondoesnotexist(colors='RGB', show=False)[0]
        faces = detector.get(img=img)
        if len(faces) == 0:  # need work only with face
            continue
        for face in faces:
            unknown = Person2(full_img=img, face=face)
            near_dist = unknown.get_label(all_persons, threshold=recog_tresh, show=False)

            face.crop_face = unknown.crop_face
            face.label = unknown.label
            face.color = unknown.color
            face.etalon_path = unknown.etalon_path
            face.rec_score = near_dist

        img_idx += 1
        if faces[0].label != 'Unknown':
            dimg = detector.draw_on(img, faces, plot_roi=True, plot_crop_face=True, plot_etalon=True, show=False)
            cv2.imwrite(str(new_output_dir_path / f'{datetime.now()}.jpg'), cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR))
        print(img_idx)
