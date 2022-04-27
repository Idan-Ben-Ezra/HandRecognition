import shutil
import os
from tqdm import tqdm
DATASET_ROOT_DIR_PATH = "D:\magshimim\project\dataset"
DATASET_PATH = os.path.join(DATASET_ROOT_DIR_PATH, "Dataset")
FILE_ENDING = ".jpg"
get_id_from_name = (lambda name: int(name.split(".")[0]))
get_name_from_id = (lambda id: str(id) + FILE_ENDING)

AUGMENTED_DATA_DIR = os.path.join(DATASET_ROOT_DIR_PATH, "rgb")  # fill in, similar to the line below
FINAL_DIR = os.path.join(DATASET_ROOT_DIR_PATH, "Dataset")

file_num = 32560
dir = sorted(os.listdir(AUGMENTED_DATA_DIR), key=get_id_from_name)
# print(len(os.listdir(ADDITIONAL_DATA_PATH)), ADDITIONAL_DATA_PATH)
dir = dir[file_num:]
print(dir[0])  # should be 32560.jpg
new_filename_num = 35971

for image in tqdm(dir):
    old_name = os.path.join(AUGMENTED_DATA_DIR, image)
    new_name = os.path.join(FINAL_DIR, get_name_from_id(new_filename_num))

    shutil.copy(old_name, new_name)

    new_filename_num += 1

print(new_filename_num)