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
args = parser.parse_args()
print(args)

# Data definitions
num_classes = 152

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
val_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="val"
)
test_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="test"
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load Pretrained ViT and modify
model = models.vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# Training and validation loop
best_loss = float("inf")
best_model_dir = f"../models/vit_ratio_{args.ratio}_gender_bal_{args.gender_balanced}/"
best_model_path = best_model_dir + "best_vit_model.pth"
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


def save_predictions(dataloader, dataset, filename):
    predictions = []
    with torch.no_grad():
        for images, labels, gender in tqdm(
            dataloader, desc=f"Saving Predictions {filename}"
        ):
            labels = torch.hstack([labels, gender])
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(zip(dataset.samples, preds.cpu().numpy()))
    df = pd.DataFrame(predictions, columns=["ImagePath", "Prediction"])
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


# Save predictions
save_predictions(train_loader, train_dataset, "train_predictions.csv")
save_predictions(test_loader, test_dataset, "test_predictions.csv")
