import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from imblearn.over_sampling import RandomOverSampler
import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

import torch
from torch.utils.data import DataLoader
from torchvision.transformers import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize, 
    ToTensor
)

from pathlib import Path
from tqdm import tqdm
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dict = {}

file_names = []
labels  = []

for file in sorted ((Path('data') / 'train').glob('*.jpg')):
    label = str(file).split('/')[-2]
    labels.append(label)
    file_names.append(str(file))

print(len(file_names), len(labels))

df = pd.DataFrame({'image': file_names, 'label': labels})
print(df.shape)