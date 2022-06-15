import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from self_src.utils import Person, get_random_color

align = True
bright_etalon = 150  # 150
n_persons = 20000

cols = ['path']
cols.extend([i for i in range(0, 512)])
df = pd.DataFrame(columns=cols)

all_persons = []
'''
PERSON_DIR = Path('/home/psv/Downloads/datasets/lfw1img/')
person_dirs = list(PERSON_DIR.glob('*'))
PERSON_DIR2 = Path('/home/psv/Downloads/datasets/lfw/')
person_dirs.extend(list(PERSON_DIR2.glob('*')))
# PERSON_DIR = Path('/home/psv/Downloads/datasets/ready_100/')
p_bar = tqdm(person_dirs[:n_persons], colour='yellow')
error_counter = 0
for dir in p_bar:
    p_bar.set_description(f'{str(dir)}')
    img_path = list(dir.glob('*.jpg'))[0]
    label = img_path.parts[-2]
    try:
        person = Person(path=img_path, label=label, makemask=True, change_brightness=True)
        for_write = [person.path]
        for_write.extend(person.embedding.tolist()[0])
        df.loc[person.label, :] = for_write
        all_persons.append(person)
    except (IndexError, cv2.error, AttributeError) as e:
        error_counter += 1
        print('\n', f'error #{error_counter}', dir, e)
        continue

# '''
show_p = False
DIR = Path('/home/psv/file/project/recog_datasets/LABELED_FACES/LABELED_full/')
person1 = Person(path=DIR / 'Dmitryi_Base/fas.jpg', label='Dmitryi_Base', color=get_random_color(), makemask=True, show_kps=True)
person2 = Person(path=DIR / 'Andrey/fas.jpg', label='Andrey', color=get_random_color(), makemask=True, show_kps=True)
person3 = Person(path=DIR / 'Sergey_M/fas.jpg', label='Sergey_M', color=get_random_color(), makemask=True, show_kps=True)
person4 = Person(path=DIR / 'Victor/fas.jpg', label='Victor', color=get_random_color(), makemask=True, show_kps=True)
# person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', color=(0, 255, 0), makemask=True, show=show_p)
# person1 = Person(path=DIR / 'Sergei/fas.jpg', label='Sergei', color=(0, 255, 0), makemask=True, show=show_p)
# person2 = Person(path=DIR / 'Vladislav/fas.jpg', label='Vladislav', color=(255, 0, 0), makemask=True, show=show_p)
# person3 = Person(path=DIR / 'Farid/fas.jpg', label='Farid', color=(180, 237, 140), makemask=True, show=show_p)
# person4 = Person(path=DIR / 'Denis/fas.jpg', label='Denis', color=(80, 127, 200), makemask=True, show=show_p)
# person5 = Person(path=DIR / 'Anton/fas.jpg', label='Anton', color=(213, 147, 138), makemask=True, show=show_p)
# person6 = Person(path=DIR / 'Alexandr/fas.jpg', label='Alexandr', color=(131, 158, 101), makemask=True, show=show_p)
# person7 = Person(path=DIR / 'Putin/fas.jpg', label='Putin', color=(113, 47, 38), makemask=True, show=show_p)
# person8 = Person(path=DIR / 'Irina/fas.jpg', label='Irina', color=(31, 58, 61), makemask=True, show=show_p)
# person9 = Person(path=DIR / 'Korzh/fas.jpg', label='Korzh', color=(137, 40, 80), makemask=True, show=show_p)
# person10 = Person(path=DIR / 'Bruce/fas.jpg', label='Bruce', color=(180, 40, 137), makemask=True, show=show_p)
# person11 = Person(path=DIR / 'Alena/fas.jpg', label='Alena', color=(108, 84, 173), makemask=True, show=show_p)
# person12 = Person(path=DIR / 'Dmitryi/fas.jpg', label='Dmitryi', color=(0, 100, 0), makemask=True, show=show_p)
# person13 = Person(path=DIR / 'VladislavV/fas.jpg', label='VladislavV', color=(100, 0, 100), makemask=True, show=show_p)
# person14 = Person(path=DIR / 'Anna/fas.jpg', label='Anna', color=get_random_color(), makemask=True, show=show_p)
# person15 = Person(path=DIR / 'Evgeniy/fas.jpg', label='Evgeniy', color=get_random_color(), makemask=True, show=show_p)
all_persons.extend(
    [person1, person2, person3, person4])
# '''

for person in all_persons:
    for_write = [person.path]
    for_write.extend(person.embedding.tolist()[0])
    df.loc[person.label, :] = for_write

df.to_csv(f'123123_all_persons={len(df)}_bright_etalon={bright_etalon}_embeddings.csv')
# TODO: save all crop face in special dir
