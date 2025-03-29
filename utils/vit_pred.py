import os
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from dataLoaders import ImSituVerbGender
import torchvision.transforms as transforms

# Params
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 152
BASE_DIR = "C:/Users/btokas/Projects/Datasets/imSitu/"

# ARG_PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--balanced", default=1, type=int)
parser.add_argument("--ratio", default=1, type=int)  # Set 1 for balanced
parser.add_argument("--gender_balanced", default=1, type=int)  # Set 1 for balanced
parser.add_argument("--blackout_box", default=False)
parser.add_argument("--img_dir", default=BASE_DIR + "of500_images_resized/")
parser.add_argument("--ann_dir", default=BASE_DIR)
parser.add_argument("--batch_size", default=128, type=int)
args = parser.parse_args()
print(args)

# Load Pretrained ViT and modify
model = models.vit_b_16(
    pretrained=False
)  # Set pretrained=False to avoid re-downloading weights
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model = model.to(DEVICE)

# Load the trained weights
out_dir = f"../models/vit_ratio_{args.ratio}_gender_bal_{args.gender_balanced}/"
model_path = out_dir + "best_vit_model.pth"
if not (os.path.isdir(out_dir)):
    os.makedirs(out_dir)

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

print("Model loaded successfully!")


# Data transformations
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


# Load datasets
train_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="train"
)
test_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="test"
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Getting Predictions
model.eval()


def save_predictions(dataloader, filename):
    gts = torch.zeros((len(dataloader.dataset), num_classes))
    preds = torch.zeros((len(dataloader.dataset), num_classes))
    curr = 0
    with torch.no_grad():
        for images, labels, gender in tqdm(
            dataloader, desc=f"Saving Predictions {filename}"
        ):
            labels = torch.hstack([labels, gender])
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            d = len(labels)
            gts[curr : curr + d] = labels
            preds[curr : curr + d] = probs
            curr = curr + d
    torch.save(gts, filename + "_gts.pth")
    torch.save(preds, filename + "_preds.pth")
    print(f"Predictions saved to {filename}")
    return gts, preds


# Save predictions
gt_train, pred_train = save_predictions(train_loader, out_dir + "train")
gt_test, pred_test = save_predictions(test_loader, out_dir + "test")
