# Importing Libraries
import os
import torch
import argparse
from dataLoaders import ImSituVerbGender


# Constants
BASE_DIR = "C:/Users/btokas/Projects/Datasets/imSitu/"
CONCEPT_DIR = BASE_DIR + "concepts/"
GENDER_DIR = CONCEPT_DIR + "gender/"
TASK_DIR = CONCEPT_DIR + "task/"


# Helper Functions
def getDirName(verb_ann: torch.tensor, gender_ann: torch.tensor) -> tuple[str, str]:
    verb_pos = torch.where(verb_ann == 1)[0]
    if len(verb_pos) > 0:
        verb_pos = verb_pos.item()
        verb_dir_name = TASK_DIR + "t_" + str(verb_pos).zfill(3) + "/"
    else:
        verb_dir_name = None
    gender = "male" if (gender_ann[0].item() == 1) else "female"
    gender_dir_name = GENDER_DIR + gender + "/"
    return (verb_dir_name, gender_dir_name)


# Data Loader
parser = argparse.ArgumentParser()
parser.add_argument("--balanced", default=False)
parser.add_argument("--ratio", default=3)  # Set 1 for balanced
parser.add_argument("--blackout_box", default=False)
parser.add_argument("--gender_balanced", default=False)  # Set True for balanced
args = parser.parse_args()

img_dir = BASE_DIR + "of500_images_resized/"
ann_dir = BASE_DIR


data_obj = ImSituVerbGender(args, ann_dir, img_dir, transform=None)

for num, (img, verb_ann, gender_ann) in enumerate(data_obj, start=1):
    print(f"\rWorking on Image : {num}/{len(data_obj)}", end="")
    verb_dir, gender_dir = getDirName(verb_ann, gender_ann)
    if verb_dir != None:
        if not (os.path.isdir(verb_dir)):
            os.makedirs(verb_dir)
        img.save(verb_dir + f"{num}.jpg")
    if not (os.path.isdir(gender_dir)):
        os.makedirs(gender_dir)
    img.save(gender_dir + f"{num}.jpg")
