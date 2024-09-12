# %%
import glob
import io
import os
import pdb
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import auc, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import binarize
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# %% [markdown]
# # Misc. Setup


# %%
# Set up device and random seed

timestamp = "20240830205516"


@dataclass
class CFG:
    model_name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    scheduler = ("CosineAnnealingLR",)
    lr = 1e-4
    lr_decay_rate = 1.0
    weight_decay = 1e-3
    warmup_ratio = 0.05
    T_max = 500
    min_lr = 1e-6
    batch_size = 32
    batch_size_val = 512
    img_size = 224  # 224 or 384 or 336(eva02)
    early_stop_count = 100  # 改善しなかったときに止めるepoch数
    num_epochs = 30
    num_folds = 5
    # train_folds = [2]
    train_folds = [0, 1, 2, 3, 4]
    ratio_int = 1
    ratio_int_val = 10
    ratio_upsampling = 2
    OUTPUT_DIR = Path(f"output_{timestamp}")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    print(f"=== Output to {OUTPUT_DIR} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    random_seed = 42

    # train entire model vs. just the classifier
    freeze_base_model = False  # didn't get good results

    # if this is set to true - full model is only generated as part of scoring (quick_train_record_count used)
    # this saves GPU quota - but saved model won't reflect what was scored...
    full_train_only_when_scoring = False  # must be False to save full model!
    quick_train_record_count = 50000  # need to get at least some positive cases even for test run


def seed_everything(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith("__"))


cfg = CFG()

seed_everything(cfg.random_seed)

# %% [markdown]
# # Load meta - and split folds

# %%
df_train = pd.read_csv("./input/isic-2024-challenge/train-metadata.csv")
df_fold = pd.read_csv("./df_fold.csv")

df_train = df_train.merge(df_fold, left_on="isic_id", right_on="isic_id", how="left")

# Add summary
fold_summary = df_train.groupby("fold")["patient_id"].nunique().to_dict()
total_patients = df_train["patient_id"].nunique()

print(f"Fold Summary (patients per fold):")
for fold, count in fold_summary.items():
    if fold != -1:  # Exclude the initialization value
        print(f"Fold {fold}: {count} patients")
print(f"Total patients: {total_patients}")

# %%
# Additional Filter
print("-" * 20)
print("Indeterminate の数")
print(df_train[df_train["iddx_1"] == "Indeterminate"].groupby("fold")["target"].value_counts())

print("-" * 20)
print("iddx_2が存在する数")
print(df_train[df_train["iddx_2"].notna()].groupby("fold")["target"].value_counts())

exclude_isic_ids = []
# # 1
# filter = (df_train["target"] == 0) & (df_train["iddx_1"] == "Indeterminate")
# exclude_isic_ids.extend(df_train[filter]["isic_id"].values.tolist())
# # 2
# filter = (df_train["target"] == 0) & (df_train["iddx_2"].notna())
# exclude_isic_ids.extend(df_train[filter]["isic_id"].values.tolist())
# # 3
# filter = (df_train["target"] == 0) & (df_train["lesion_id"].notna())
# exclude_isic_ids.extend(df_train[filter]["isic_id"].values.tolist())

# exclude_isic_ids = list(set(exclude_isic_ids))


# %% [markdown]
# # Load meta data / review

# %%
# Set the HDF5 file path
TRAIN_HDF5_FILE_PATH = "./input/isic-2024-challenge/train-image.hdf5"

# are we scoring?
scoring = False
# check length of test data to see if we are scoring....
test_length = len(pd.read_csv("./input/isic-2024-challenge/test-metadata.csv"))
if test_length > 3:
    scoring = True

if not scoring:
    if cfg.full_train_only_when_scoring:
        df_train = df_train.head(cfg.quick_train_record_count)

print("\nOriginal Dataset Summary:")
print(f"Total number of samples: {len(df_train)}")
print(f"Number of unique patients: {df_train['patient_id'].nunique()}")

original_positive_cases = df_train["target"].sum()
original_total_cases = len(df_train)
original_positive_ratio = original_positive_cases / original_total_cases

print(f"Number of positive cases: {original_positive_cases}")
print(f"Number of negative cases: {original_total_cases - original_positive_cases}")
print(
    f"Ratio of negative to positive cases: {(original_total_cases - original_positive_cases) / original_positive_cases:.2f}:1"
)

train_steps = int(
    (cfg.ratio_int + 1)
    * original_positive_cases
    * cfg.ratio_upsampling
    * cfg.num_epochs
    * ((cfg.num_folds - 1) / cfg.num_folds)
    / cfg.batch_size
)
print(f"Train steps: {train_steps}")
cfg.T_max = train_steps


# %%
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        if "eva02" in self.model_name:
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()
            self.linear = nn.Linear(in_features, num_classes)
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        elif "efficientnetv2" in self.model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            self.pooling = GeM()
            if self.pooling:  # My custom pooling
                self.model.global_pool = nn.Identity()
            self.linear = nn.Linear(in_features, num_classes)
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        elif "convnextv2" in self.model_name:
            in_features = self.model.head.fc.in_features
            self.model.head = nn.Identity()
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
                nn.Flatten(),  # フラット化
                nn.Linear(in_features, 256),  # 新しい全結合層1
                nn.ReLU(),  # 活性化関数
                nn.Dropout(0.5),  # ドロップアウト
                nn.Linear(256, num_classes),  # 出力層（2クラス分類）
            )
        elif "swinv2" in self.model_name:
            in_features = self.model.head.fc.in_features
            self.model.head = nn.Identity()
            self.feature_extractor = nn.Sequential(
                # nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
                GeM(),
                nn.Flatten(),  # フラット化
                nn.Linear(in_features, 256),  # 新しい全結合層1
                nn.ReLU(),  # 活性化関数
            )
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])  # 5つのDropout
            self.classifier = nn.Linear(256, num_classes)  # 出力層（2クラス分類）
        elif "vit_tiny" in self.model_name:
            in_features = self.model.head.in_features
            self.model.norm = nn.Identity()
            self.model.fc_norm = nn.Identity()
            self.model.head_drop = nn.Identity()
            self.model.head = nn.Identity()
            self.feature_extractor = nn.Sequential(
                nn.Flatten(),  # フラット化
                nn.Linear(in_features, 64),  # 新しい全結合層1
                nn.ReLU(),  # 活性化関数
            )
            self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(1)])  # 5つのDropout
            self.classifier = nn.Linear(64, num_classes)  # 出力層（2クラス分類）

    def forward(self, images):
        features = self.model(images)

        if any(
            [
                "efficientnetv2" in self.model_name,
                "eva02" in self.model_name,
            ]
        ):
            # Custom poolingがある場合
            if self.pooling:
                features = self.pooling(features).flatten(1)
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    output = self.linear(dropout(features))
                else:
                    output += self.linear(dropout(features))
            output /= len(self.dropouts)

        if "convnextv2" in self.model_name:
            output = self.head(features)

        if "swinv2" in self.model_name:
            features = self.feature_extractor(features)
            output = torch.mean(torch.stack([dropout(features) for dropout in self.dropouts]), dim=0)
            output = self.classifier(output)

        if "vit_tiny" in self.model_name:
            features = self.feature_extractor(features)
            output = torch.mean(torch.stack([dropout(features) for dropout in self.dropouts]), dim=0)
            output = self.classifier(output)

        return output.squeeze()


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


# %% [markdown]
# # Data Loading / Image Augmentation


# %%
class ISICDataset(Dataset):
    def __init__(self, hdf5_file, isic_ids, targets=None, transform=None, ratio_int=2):
        self.hdf5_file = hdf5_file
        self.isic_ids = isic_ids
        self.targets = targets
        self.transform = transform
        self.ratio_int = ratio_int  # 例えば2の場合、pos:neg = 1:2となる
        self.positive_list = [ii for ii, tt in zip(self.isic_ids, self.targets) if tt == 1]
        random.shuffle(self.positive_list)
        self.negative_list = [ii for ii, tt in zip(self.isic_ids, self.targets) if tt == 0]
        random.shuffle(self.negative_list)
        self.balanced_list = self.create_balanced_list()

    def create_balanced_list(self):
        balanced_list = []
        pos_count = 0
        neg_count = 0
        # PositiveリストとNegativeリストを比率に従って繰り返し並べる
        while pos_count < len(self.positive_list) or neg_count < len(self.negative_list):
            if pos_count < len(self.positive_list):
                balanced_list.append(self.positive_list[pos_count])
                pos_count += 1

            for _ in range(self.ratio_int):
                if neg_count < len(self.negative_list):
                    balanced_list.append(self.negative_list[neg_count])
                    neg_count += 1
        return balanced_list

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):

        isic_id = self.balanced_list[idx]
        org_idx = np.where(self.isic_ids == isic_id)[0][0]

        with h5py.File(self.hdf5_file, "r") as f:
            # img_bytes = f[self.isic_ids[idx]][()]
            img_bytes = f[isic_id][()]

        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)  # Convert PIL Image to numpy array

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.targets is not None:
            target = self.targets[org_idx]
        else:
            target = torch.tensor(-1)  # Dummy target for test set

        return img, target


class ISICDatasetVal(Dataset):
    def __init__(self, hdf5_file, isic_ids, targets=None, transform=None):
        self.hdf5_file = hdf5_file
        self.isic_ids = isic_ids
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, "r") as f:
            img_bytes = f[self.isic_ids[idx]][()]

        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)  # Convert PIL Image to numpy array

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = torch.tensor(-1)  # Dummy target for test set

        return img, target


# Prepare augmentation
# aug_transform = A.Compose(
#     [
#         A.RandomRotate90(),
#         A.Flip(),
#         A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
#         A.Resize(cfg.img_size, cfg.img_size),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2(),
#     ]
# )

aug_transform = A.Compose(
    [
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.75),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ],
            p=0.7,
        ),
        A.OneOf(
            [
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.0),
                A.ElasticTransform(alpha=3),
            ],
            p=0.7,
        ),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(cfg.img_size, cfg.img_size),
        A.CoarseDropout(max_height=int(cfg.img_size * 0.375), max_width=int(cfg.img_size * 0.375), min_holes=1, p=0.7),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

base_transform = A.Compose(
    [
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

# %% [markdown]
# # Visualize image augmentation

# %%
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torchvision.utils import make_grid


def visualize_augmentations_positive(dataset, num_samples=3, num_augmentations=5, figsize=(20, 10)):
    # Find positive samples
    positive_samples = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 1:  # Assuming 1 is the positive class
            positive_samples.append(i)

        if len(positive_samples) == num_samples:
            break

    if len(positive_samples) < num_samples:
        print(f"Warning: Only found {len(positive_samples)} positive samples.")

    fig, axes = plt.subplots(num_samples, num_augmentations + 1, figsize=figsize)
    fig.suptitle("Original and Augmented Versions of Positive Samples", fontsize=16)

    for sample_num, sample_idx in enumerate(positive_samples):
        # Get a single sample
        original_image, label = dataset[sample_idx]

        # If the image is already a tensor (due to ToTensorV2 in the transform), convert it back to numpy
        if isinstance(original_image, torch.Tensor):
            original_image = original_image.permute(1, 2, 0).numpy()

        # Reverse the normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_image = (original_image * std + mean) * 255
        original_image = original_image.astype(np.uint8)

        # Display original image
        axes[sample_num, 0].imshow(original_image)
        axes[sample_num, 0].axis("off")
        axes[sample_num, 0].set_title("Original", fontsize=10)

        # Apply and display augmentations
        for aug_num in range(num_augmentations):
            augmented = dataset.transform(image=original_image)["image"]
            # If the result is a tensor, convert it back to numpy
            if isinstance(augmented, torch.Tensor):
                augmented = augmented.permute(1, 2, 0).numpy()
            # Reverse the normalization
            augmented = (augmented * std + mean) * 255
            augmented = augmented.astype(np.uint8)

            axes[sample_num, aug_num + 1].imshow(augmented)
            axes[sample_num, aug_num + 1].axis("off")
            axes[sample_num, aug_num + 1].set_title(f"Augmented {aug_num + 1}", fontsize=10)

    plt.tight_layout()
    plt.show()


augtest_dataset = ISICDataset(
    hdf5_file=TRAIN_HDF5_FILE_PATH,
    isic_ids=df_train["isic_id"].values,
    targets=df_train["target"].values,
    transform=aug_transform,
)

# visualize_augmentations_positive(augtest_dataset)

# %% [markdown]
# Scoring code from https://www.kaggle.com/code/metric/isic-pauc-abovetpr


# %%
def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float = 0.80) -> float:

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution.values) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(submission.values)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc


def custom_metric(y_hat, y_true):
    # y_hat = estimator.predict_proba(X)[:, 1]
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])

    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


# %% [markdown]
# # Train / CV

# %%
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, fold, epoch, device, global_step):
    scaler = GradScaler()

    # Training phase
    model.train()
    train_loss = 0.0  # 総損失を計算
    for inputs, targets in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1}  Training"):
        inputs, targets = inputs.to(device), targets.to(device, dtype=torch.float)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

    # 平均トレーニング損失をログ
    train_loss /= len(train_loader)

    # Evaluation phase
    model.eval()
    val_targets, val_outputs = [], []
    val_loss = 0.0  # 評価損失を計算
    with torch.no_grad(), autocast():
        for inputs, targets in tqdm(val_loader, desc=f"Fold {fold} - Epoch {epoch+1} Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device, dtype=torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_targets.append(targets.cpu())
            # val_outputs.append(outputs.softmax(dim=1)[:, 1].cpu())
            val_outputs.append(outputs.cpu())

    # 平均評価損失をログ
    val_loss /= len(val_loader)
    val_targets = torch.cat(val_targets).numpy()
    val_outputs = torch.cat(val_outputs).numpy()
    val_score = custom_metric(val_outputs, val_targets)
    print(f"Fold {fold} - Epoch {epoch+1} Score: {val_score:.7f}")

    return val_targets, val_outputs, val_score, global_step


def cross_validation_train(
    df_train, num_folds, train_folds, num_epochs, hdf5_file_path, aug_transform, base_transform, device
):
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    all_val_targets, all_val_outputs = [], []

    for fold in train_folds:
        print(f"\nFold {fold}/{len(train_folds)}")

        # Initialize model, optimizer, and scheduler
        # model = setup_model().to(device)
        model = ISICModel(cfg.model_name)
        model.to(device)

        # ============================
        # Optimizer
        # ============================
        # Set optimizer parameters
        def get_optimizer_params(model, lr_ini, lr_decay_rate=1.0, weight_decay=0.0):
            no_decay = ["bias", "norm"]
            # initialize lr for task specific layer (head部分, fc層とか）
            optimizer_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if "model" not in n],
                    "weight_decay": weight_decay,
                    "lr": lr_ini,
                },
            ]
            # モデルに応じてモジュール名に注意
            if "convnextv2" in model.model_name:
                layers = [model.model.stem] + list(model.model.stages)
            elif "swinv2" in model.model_name:
                layers = [model.model.patch_embed] + list(model.model.stages)
            elif "vit_tiny" in model.model_name:
                layers = [model.model.patch_embed] + list(model.model.blocks)

            layers.reverse()
            lr = lr_ini * lr_decay_rate
            for layer in layers:
                optimizer_parameters += [
                    # no_decayリストに該当しないものは、weight_decayする
                    {
                        "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": weight_decay,
                        "lr": lr,
                    },
                    # no_decayリストに該当する者は、weight_decayを適用しない
                    {
                        "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr": lr,
                    },
                ]
                lr *= lr_decay_rate

            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(model, cfg.lr, cfg.lr_decay_rate, weight_decay=cfg.weight_decay)
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=cfg.lr)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _step: _step / int(cfg.T_max * cfg.warmup_ratio)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[int(cfg.T_max * cfg.warmup_ratio)]
        )

        global_step = 1
        no_improve_count = 0
        best_score = -np.inf
        # prev_score = -np.inf
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Create datasets and data loaders
            # Split data for current fold
            train_df = df_train[df_train["fold"] != fold]
            train_df_1 = train_df[train_df["target"] == 1]
            train_df_0 = train_df[train_df["target"] == 0]
            # --- Custom filtering
            train_df_0 = train_df_0[train_df_0["tbp_lv_dnn_lesion_confidence"] > 80]
            train_df_0 = train_df[~train_df["isic_id"].isin(exclude_isic_ids)]
            # ------
            train_df_0_sampled = train_df_0.sample(
                len(train_df_1) * cfg.ratio_upsampling * cfg.ratio_int, random_state=epoch
            )
            train_df_1 = pd.concat([train_df_1 for _ in range(cfg.ratio_upsampling)])  # upsampling
            train_df = pd.concat([train_df_1, train_df_0_sampled]).reset_index(drop=True)

            val_df = df_train[df_train["fold"] == fold]
            val_df_1 = val_df[val_df["target"] == 1]
            val_df_0 = val_df[val_df["target"] == 0]
            # --- Custom filtering
            val_df_0 = val_df_0[val_df_0["tbp_lv_dnn_lesion_confidence"] > 80]
            val_df_0 = val_df_0[~val_df_0["isic_id"].isin(exclude_isic_ids)]
            # ------
            val_df_0_sampled = val_df_0.sample(len(val_df_1) * cfg.ratio_int_val, random_state=42)
            val_df = pd.concat([val_df_1, val_df_0_sampled]).reset_index(drop=True)

            train_dataset = ISICDataset(
                hdf5_file_path,
                train_df["isic_id"].values,
                train_df["target"].values,
                aug_transform,
                ratio_int=cfg.ratio_int,
            )
            val_dataset = ISICDatasetVal(
                hdf5_file_path, val_df["isic_id"].values, val_df["target"].values, base_transform
            )
            train_loader = DataLoader(
                train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=cfg.batch_size_val, shuffle=False, num_workers=4, pin_memory=True
            )
            print(
                f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
                f"Train Pos Ratio: {train_df['target'].mean():.2%}, Val Pos Ratio: {val_df['target'].mean():.2%}"
            )

            # Train and evaluate
            val_targets, val_outputs, val_score, global_step = train_evaluate(
                model, train_loader, val_loader, criterion, optimizer, scheduler, fold, epoch, device, global_step
            )
            if val_score > best_score:
                print(f"Score improved! : {best_score:.7f}->{val_score:.7f}")
                best_score = val_score
                # if val_score > prev_score:
                #     print(f"Better than previous epoch! : {prev_score:.7f}->{val_score:.7f}")
                torch.save(
                    model.state_dict(), cfg.OUTPUT_DIR / f"fold_{fold}_epoch_{epoch + 1}_score_{val_score:.4f}.pth"
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > cfg.early_stop_count:
                    print("しばらく改善しなかったので、学習を止めます")
                    break
            prev_score = val_score

        del model
        torch.cuda.empty_cache()

        all_val_targets.extend(val_targets)
        all_val_outputs.extend(val_outputs)
    all_val_outputs = np.array(all_val_outputs)
    all_val_targets = np.array(all_val_targets)

    # Create DataFrames with row_id for scoring
    cv_score = custom_metric(all_val_outputs, all_val_targets)
    print(f"CV pAUC Score: {cv_score:.7f}")

    return all_val_targets, all_val_outputs


# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Perform cross-validation training
all_val_targets, all_val_outputs = cross_validation_train(
    df_train,
    cfg.num_folds,
    cfg.train_folds,
    cfg.num_epochs,
    TRAIN_HDF5_FILE_PATH,
    aug_transform,
    base_transform,
    device,
)


# %% [markdown]
# # ↑ Train finish!

# %%
# Final overall evaluation
print("\nFinal Overall Evaluation:")

# Calculate the official pAUC score
solution_df = pd.DataFrame({"target": all_val_targets, "row_id": range(len(all_val_targets))})
submission_df = pd.DataFrame({"prediction": all_val_outputs, "row_id": range(len(all_val_outputs))})
official_score = score(solution_df, submission_df, "row_id")
print(f"Overall pAUC Score: {official_score:.4f}")

# Generate and print classification report
binary_predictions = binarize(np.array(all_val_outputs).reshape(-1, 1), threshold=0.5).reshape(-1)
report = classification_report(all_val_targets, binary_predictions, target_names=["Class 0", "Class 1"])
print("\nOverall Classification Report:")
print(report)

# Print specific metrics for Class 1
report_dict = classification_report(
    all_val_targets, binary_predictions, target_names=["Class 0", "Class 1"], output_dict=True
)
print(f"\nClass 1 Metrics:")
print(f"Precision: {report_dict['Class 1']['precision']:.4f}")
print(f"Recall: {report_dict['Class 1']['recall']:.4f}")
print(f"F1-score: {report_dict['Class 1']['f1-score']:.4f}")


# %% [markdown]
# # Inference Code
# * There are some duplicate definitions / includes here to make copying to other notebooks easier

import io

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import timm

# %%
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ISICDataset(Dataset):
    def __init__(self, hdf5_file, isic_ids, targets=None, transform=None):
        self.hdf5_file = h5py.File(hdf5_file, "r")  # Keep file open
        self.isic_ids = isic_ids
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        img_bytes = self.hdf5_file[self.isic_ids[idx]][()]
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        target = self.targets[idx] if self.targets is not None else torch.tensor(-1)
        return img, target

    def __del__(self):
        self.hdf5_file.close()  # Ensure file is closed when object is destroyed


# Define the albumentations transformation
base_transform = A.Compose(
    [
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def get_latest_epoch_file(folder_path, target_fold):
    # 正規表現パターン：fold_X_epoch_Y_score_Z.pth の形式に一致
    pattern = re.compile(r"fold_(\d+)_epoch_(\d+)_score_(\d+\.\d+)\.pth")

    max_epoch = -1
    latest_file = None

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            fold, epoch, score = match.groups()
            fold = int(fold)
            epoch = int(epoch)

            if fold == target_fold and epoch > max_epoch:
                max_epoch = epoch
                latest_file = filename

    if latest_file:
        return os.path.join(folder_path, latest_file)
    else:
        return None


def get_max_score_file(folder_path, target_fold):
    # 正規表現パターン：fold_X_epoch_Y_score_Z.pth の形式に一致
    pattern = re.compile(r"fold_(\d+)_epoch_(\d+)_score_(\d+\.\d+)\.pth")

    max_score = -1
    latest_file = None

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            fold, epoch, score = match.groups()
            fold = int(fold)
            epoch = int(epoch)
            score = float(score)

            if fold == target_fold and score > max_score:
                max_score = score
                latest_file = filename

    if latest_file:
        return os.path.join(folder_path, latest_file)
    else:
        return None


def load_model(fold, device):
    model = ISICModel(cfg.model_name)
    model.to(device)
    model_w_path = get_latest_epoch_file(cfg.OUTPUT_DIR, fold)
    model.load_state_dict(torch.load(model_w_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()  # Apply no_grad to the entire function
def ensemble_predict(models, test_loader, device):
    all_predictions = []
    for inputs, _ in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(device)
        # fold_predictions = torch.stack([model(inputs).softmax(dim=1)[:, 1] for model in models])
        fold_predictions = torch.stack([model(inputs) for model in models])
        avg_predictions = fold_predictions.mean(dim=0)
        all_predictions.extend(avg_predictions.cpu().numpy())
    return all_predictions


# %% [markdown]
# # Generate out-of-fold predictions for Train
# * Only done if not being submitted


# %%
def generate_oof_predictions(df_train, folds, hdf5_file_path, transform):
    oof_predictions = np.zeros(len(df_train))
    model_filenames = [""] * len(df_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in folds:
        print(f"Generating predictions for fold {fold}/{cfg.num_folds}")

        model = load_model(fold, device)
        val_df = df_train[df_train["fold"] == fold].copy()
        val_dataset = ISICDataset(hdf5_file_path, val_df["isic_id"].values, val_df["target"].values, transform)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size_val, shuffle=False, num_workers=4, pin_memory=True
        )

        fold_predictions = ensemble_predict([model], val_loader, device)

        oof_predictions[val_df.index] = fold_predictions
        model_filename = f"model_fold_{fold}_epoch_{cfg.num_epochs}.pth"
        for idx in val_df.index:
            model_filenames[idx] = model_filename

        del model
        torch.cuda.empty_cache()

    return oof_predictions, model_filenames


# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the number of folds

# Generate out-of-fold predictions
oof_predictions, model_filenames = generate_oof_predictions(
    df_train, cfg.train_folds, TRAIN_HDF5_FILE_PATH, base_transform
)

# Create DataFrame for OOF predictions
oof_df = pd.DataFrame(
    {
        "isic_id": df_train["isic_id"],
        "target": df_train["target"],
        "fold": df_train["fold"],
        "oof_prediction": oof_predictions,
        "model_filename": model_filenames,
    }
)

# Save OOF predictions to CSV
oof_df.to_csv(cfg.OUTPUT_DIR / "oof_predictions.csv", index=False)
print("Out-of-fold predictions saved to oof_predictions.csv")
print(oof_df.head())


# %%
# OOF pAUC calc
official_scores = []
folds = cfg.train_folds
# folds = [0, 1, 3]
for fold in folds:
    solution_df = oof_df[oof_df["fold"] == fold][["target", "isic_id"]]
    submission_df = oof_df[oof_df["fold"] == fold][["oof_prediction", "isic_id"]]
    official_score = score(solution_df, submission_df, "isic_id")
    official_scores.append(official_score)
    print(f"OOF Score (fold={fold}): {official_score:.7f}")

print(f"OOF Score folds mean: {np.mean(official_scores):.7}")
solution_df = oof_df[oof_df["fold"].isin(folds)][["target", "isic_id"]]
submission_df = oof_df[oof_df["fold"].isin(folds)][["oof_prediction", "isic_id"]]
official_score = score(solution_df, submission_df, "isic_id")
print(f"OOF Score (fold=[{folds}]): {official_score:.7f}")

print(f"{cfg.OUTPUT_DIR} was finished!")
