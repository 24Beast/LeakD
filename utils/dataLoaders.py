import os
import json
import random
import pickle
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

BASE_DIR = "C:/Users/btokas/Projects/Datasets/imSitu/"

BALANCED_CLASSES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
]


class ImSituVerbGender(data.Dataset):
    def __init__(
        self,
        args,
        annotation_dir,
        image_dir,
        split="train",
        transform=None,
        balanced_val=False,
        balanced_test=False,
    ):
        print("ImSituVerbGender dataloader")

        self.split = split
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.args = args

        verb_id_map = pickle.load(open(BASE_DIR + "verb_id.map", "rb"))
        self.verb2id = verb_id_map["verb2id"]
        self.id2verb = verb_id_map["id2verb"]

        print("loading %s annotations.........." % self.split)
        self.ann_data = pickle.load(
            open(os.path.join(annotation_dir, split + ".data"), "rb")
        )

        if args.balanced and split == "train":
            balanced_subset = pickle.load(
                open(BASE_DIR + "{}_ratio_{}.ids".format(split, args.ratio), "rb")
            )
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_val and split == "val":
            balanced_subset = pickle.load(
                open(BASE_DIR + "{}_ratio_{}.ids".format(split, args.ratio), "rb")
            )
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        if balanced_test and split == "test":
            balanced_subset = pickle.load(
                open(BASE_DIR + "{}_ratio_{}.ids".format(split, args.ratio), "rb")
            )
            self.ann_data = [self.ann_data[i] for i in balanced_subset]

        print("dataset size: %d" % len(self.ann_data))
        self.verb_ann = np.zeros((len(self.ann_data), len(self.verb2id)))
        self.gender_ann = np.zeros((len(self.ann_data), 2), dtype=int)

        for index, ann in enumerate(self.ann_data):
            self.verb_ann[index][ann["verb"]] = 1
            self.gender_ann[index][ann["gender"]] = 1

        if args.gender_balanced:
            man_idxs = np.nonzero(self.gender_ann[:, 0])[0]
            woman_idxs = np.nonzero(self.gender_ann[:, 1])[0]
            # random.shuffle(man_idxs)  # only blackout box is available for imSitu
            # random.shuffle(woman_idxs)
            min_len = 7300 if self.split == "train" else 3000
            selected_idxs = list(man_idxs[:min_len]) + list(woman_idxs[:min_len])

            self.ann_data = [self.ann_data[idx] for idx in selected_idxs]
            self.verb_ann = np.take(self.verb_ann, selected_idxs, axis=0)
            self.gender_ann = np.take(self.gender_ann, selected_idxs, axis=0)

        self.image_ids = range(len(self.ann_data))
        self.verb_ann = self.verb_ann[:, BALANCED_CLASSES]

        print(
            "man size : {} and woman size: {}".format(
                len(np.nonzero(self.gender_ann[:, 0])[0]),
                len(np.nonzero(self.gender_ann[:, 1])[0]),
            )
        )

        if args.blackout_box:
            self.masks_ann = json.load(
                open(os.path.join(annotation_dir, "masks/masks/" + split + ".json"))
            )

    def __getitem__(self, index):
        img = self.ann_data[index]
        image_name = img["image_name"]
        image_path_ = os.path.join(self.image_dir, image_name)

        img_ = Image.open(image_path_).convert("RGB")
        if self.args.blackout_box:  # only blackout box is available for imSitu

            img_ = self.blackout_img(image_name, img_)

        if self.transform is not None:
            img_ = self.transform(img_)

        return (
            img_,
            torch.Tensor(self.verb_ann[index]),
            torch.LongTensor(self.gender_ann[index]),
        )

    #        return img_, torch.Tensor(self.verb_ann[index]), \
    #                torch.LongTensor(self.gender_ann[index]), torch.LongTensor([self.image_ids[index]])

    def getGenderWeights(self):
        return (self.gender_ann == 0).sum(axis=0) / (
            1e-15 + (self.gender_ann.sum(axis=0) + (self.gender_ann == 0).sum(axis=0))
        )

    def getVerbWeights(self):
        return (self.verb_ann == 0).sum(axis=0) / (1e-15 + self.verb_ann.sum(axis=0))

    def blackout_img(self, img_name, img):
        if "agent" not in self.masks_ann[img_name]["bb"]:
            return img  # if mask is not available, return the original img

        bb = self.masks_ann[img_name]["bb"]["agent"]
        if -1 in bb:
            return img  # if mask if not available, return the original img
        else:
            xmin, ymin, xmax, ymax = self.masks_ann[img_name]["bb"]["agent"]
            width = self.masks_ann[img_name]["width"]
            height = self.masks_ann[img_name]["height"]
            black_img = Image.fromarray(np.zeros((img.size[1], img.size[0])))
            mask = np.zeros((width, height))
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    mask[j][i] = 1
            img_mask = Image.fromarray(255 * (mask > 0).astype("uint8")).resize(
                (img.size[0], img.size[1]), Image.ANTIALIAS
            )
            return Image.composite(black_img, img, img_mask)

    def __len__(self):
        return len(self.ann_data)


class ImSituVerbGenderFeature(data.Dataset):
    def __init__(self, feature_dir, split="train"):
        print("ImSituVerbGenderFeature dataloader")

        self.split = split

        print("loading %s annotations.........." % self.split)

        self.targets = torch.load(
            os.path.join(feature_dir, "{}_targets.pth".format(split))
        )
        self.genders = torch.load(
            os.path.join(feature_dir, "{}_genders.pth".format(split))
        )
        self.image_ids = torch.load(
            os.path.join(feature_dir, "{}_image_ids.pth".format(split))
        )
        self.potentials = torch.load(
            os.path.join(feature_dir, "{}_potentials.pth".format(split))
        )

        print(
            "man size : {} and woman size: {}".format(
                len(self.genders[:, 0].nonzero().squeeze()),
                len(self.genders[:, 1].nonzero().squeeze()),
            )
        )

    def __getitem__(self, index):
        return (
            self.targets[index],
            self.genders[index],
            self.image_ids[index],
            self.potentials[index],
        )

    def __len__(self):
        return len(self.targets)


class ImSituConceptDataset(data.Dataset):

    def __init__(self, gender: bool = True, task_num: int = 0):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", default=True)
    parser.add_argument("--ratio", default=1)  # Set 1 for balanced
    parser.add_argument("--blackout_box", default=False)
    parser.add_argument("--gender_balanced", default=1)  # Set True for balanced
    args = parser.parse_args()

    img_dir = BASE_DIR + "of500_images_resized/"
    ann_dir = BASE_DIR

    # https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
    rescale = lambda x: x / 255
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformation = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.functional.pil_to_tensor,
            rescale,
            normalize,
        ]
    )

    data_obj = ImSituVerbGender(args, ann_dir, img_dir, transform=transformation)
