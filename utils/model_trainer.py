# Importing Libraries
import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from dataLoaders import ImSituVerbGender
from torchvision import transforms, models

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "C:/Users/btokas/Projects/Datasets/imSitu/"

# ARG_PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--balanced", default=1, type=int)
parser.add_argument("--ratio", default=1, type=int)  # Set 1 for balanced
parser.add_argument("--gender_balanced", default=1, type=int)  # Set 1 for balanced
parser.add_argument("--blackout_box", default=False)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_epochs", default=10, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--img_dir", default=BASE_DIR + "of500_images_resized/")
parser.add_argument("--ann_dir", default=BASE_DIR)
parser.add_argument("--model", default="vit")
args = parser.parse_args()
print(args)

# Data definitions
num_classes = 207

# Data transformations
# https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
model_name = args.model
if model_name == "swin":
    resize = transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC)
else:
    resize = transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR)
rescale = lambda x: x / 255
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformation = transforms.Compose(
    [
        resize,
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
val_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="val"
)
test_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="test"
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load Pretrained model and modify
if model_name == "vit":
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
elif model_name == "swin":
    model = models.swin_t(pretrained=True)
    model.heads.head = nn.Linear(model.head.in_features, num_classes)
elif model_name == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif model_name == "vgg16":
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(DEVICE)


# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# Training and validation loop
best_loss = float("inf")
best_model_dir = f"../models/{model_name}_ratio_{args.ratio}_genderbal_{args.gender_balanced}_bal_{args.balanced}/"
best_model_path = best_model_dir + "best_{model_name}_model.pth"
if not (os.path.isdir(best_model_dir)):
    os.makedirs(best_model_dir)

for epoch in range(args.num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels, gender in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}"
    ):
        labels = torch.hstack([labels, gender])
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels, gender in val_loader:
            labels = torch.hstack([labels, gender])
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(
        f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val loss: {val_loss:.4f}"
    )

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("Best model saved!")

# Load best model
model.load_state_dict(torch.load(best_model_path))
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
gt_train, pred_train = save_predictions(train_loader, best_model_dir + "train")
gt_test, pred_test = save_predictions(test_loader, best_model_dir + "test")
