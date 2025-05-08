# Importing Libraries
import os
import glob
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from dataLoaders import ImSituVerbGender
from torch.utils.data import DataLoader
from torchvision import transforms, models
from captum.attr import LayerIntegratedGradients
from interpret import ModifiedTCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import (
    dataset_to_dataloader,
    CustomIterableDataset,
)

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "C:/Users/btokas/Projects/Datasets/imSitu/"


# ARG_PARSER
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobile_v2")
parser.add_argument("--balanced", default=1, type=int)
parser.add_argument("--ratio", default=1, type=int)  # Set 1 for balanced
parser.add_argument("--gender_balanced", default=1, type=int)  # Set 1 for balanced
parser.add_argument("--blackout_box", default=False)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--task_cav", action="store_true")
parser.add_argument("--img_dir", default=BASE_DIR + "of500_images_resized/")
parser.add_argument("--ann_dir", default=BASE_DIR)
args = parser.parse_args()
print(args)

# Data definitions
num_classes = 207

# Data transformations
# https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
model_name = args.model
if (model_name == "swin") or (model_name == "mobile_v3"):
    resize = transforms.Resize(232, interpolation=transforms.InterpolationMode.BICUBIC)
elif model_name == "swin_s":
    resize = transforms.Resize(246, interpolation=transforms.InterpolationMode.BICUBIC)
elif model_name == "maxvit":
    resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
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

# Initialize dataset
test_dataset = ImSituVerbGender(
    args, args.ann_dir, args.img_dir, transform=transformation, split="test"
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


# Helper Functions
def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return transformation(img).to(DEVICE)


def load_image_tensors(class_name, root_path, transform=True):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + "/*.jpg")
    tensors = []
    for filename in filenames:
        img = Image.open(filename).convert("RGB")
        tensors.append(transformation(img) if transform else img)
    return tensors


def assemble_concept(name, id, concepts_path):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=name, data_iter=concept_iter)


# Concept Definition
concepts = []
CONCEPT_DIR = BASE_DIR + "/concepts_balanced/"
relevant_ind = []
if args.task_cav:
    CONCEPT_DIR = CONCEPT_DIR + "task/"
    for i in range(num_classes - 2):
        concept_name = "t_" + str(i).zfill(3)
        curr_concept = assemble_concept(concept_name, i, CONCEPT_DIR)
        concepts.append(curr_concept)
    relevant_ind = [205, 206]
else:
    CONCEPT_DIR = CONCEPT_DIR + "gender/"
    male_concept = assemble_concept("male", 0, CONCEPT_DIR)
    female_concept = assemble_concept("female", 1, CONCEPT_DIR)
    concepts = [male_concept, female_concept]
    relevant_ind = [i for i in range(205)]

# Load Pretrained model and modify
layers = None
if model_name == "vit":
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    layers = "encoder.ln"
elif model_name == "swin":
    model = models.swin_t(pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    layers = "flatten"
elif model_name == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif model_name == "vgg16":
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    layers = "avgpool"
elif model_name == "mobile_v3":
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    layers = "classifier.0"
elif model_name == "mobile_v2":
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    layers = "features.18.2"
elif model_name == "squeezenet_1_1":
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    layers = "classifier.0"
elif model_name == "wide_resnet50":
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    layers = "avgpool"
elif model_name == "wide_resnet101":
    model = models.wide_resnet101_2(
        weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    layers = "avgpool"
elif model_name == "vit_b_32":
    model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    layers = "encoder.ln"
elif model_name == "swin_s":
    model = models.swin_s(pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    layers = "flatten"
elif model_name == "squeezenet_1_0":
    model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    layers = "classifier.0"
elif model_name == "maxvit":
    model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
    model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)
    layers = "classifier.3"

model_dir = f"../models/{model_name}_ratio_{args.ratio}_genderbal_{args.gender_balanced}_bal_{args.balanced}/"
model_path = model_dir + f"best_{model_name}_model.pth"

model = model.to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# Initializing TCAV
mytcav = ModifiedTCAV(
    model=model,
    layers=layers,
    layer_attr_method=LayerIntegratedGradients(model, None, multiply_by_inputs=False),
)

# Getting TCAV scores
tcav_scores = []
for index in relevant_ind[:5]:
    ind_scores = []
    for batch in test_loader:
        imgs = batch[0].to(DEVICE)
        curr_scores = mytcav.interpret(
            inputs=imgs,
            experimental_sets=[concepts],
            target = index,
            n_steps=5,
        )
        ind_scores.append(curr_scores['0-1'][layers]['abs_magnitude'].cpu())
    tcav_scores.append(np.array(ind_scores))

with open(model_dir + "tcav.npy", "wb") as f:
    np.save(f, np.array(tcav_scores))
