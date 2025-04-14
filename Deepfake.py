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
from torchvision.transforms import (
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

df['label'].unique()

y = df[['label']]

df = df.drop('label', axis=1)
ros = RandomOverSampler(random_state=83)
df, y_resampled = ros.fit_resample(df, y)
del y
df['label'] = y_resampled
del y_resampled

gc.collect()
print(df.shape)

dataset = Dataset.from_pandas(df).cast_column("image", Image())
labels_list = ['Real', 'Fake']

label2id, id2label = dict(), dict()

for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

ClassLabels = ClassLabel(num_classes = len(labels_list), names = labels_list)

def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched = True)

dataset = dataset.cast_colum('label', ClassLabels)

dataset = dataset.train_test_split(test_size = 0.4, shuffle = True, stratify_by_column = 'label')

train_data = dataset['train']
test_data = dataset['test']

model_str = "dima806/deepfake_vs_real_image_detection"
processor = ViTImageProcessor.from_pretrained(model_str)
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size['height']

normalize = Normalize(mean=image_mean, std=image_std)

_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(90),
    RandomAdjustSharpness(sharpness_factor=2),
    ToTensor(),
    normalize
])

_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    normalize,
])

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {'pixel_values': pixel_values, 'labels': labels}

model = ViTForImageClassification.from_pretrained(
    model_str,
    num_labels=len(labels_list),
)

model.config.label2id = label2id
model.config.id2label = id2label   

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids =  eval_pred.label_ids
    predicted_labels = predictions.argmax(axis = 1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']

    return {
        "accuracy" : acc_score
    }

metric_name = "accuracy"
model_name = "deepfake_vs_real_image_detection"
num_train_epochs = 2

args = TrainingArguments(
    output_dir = model_name,
    logging_dir = './logs',
    evaluation_strategy = "epcoh",
    learning_rate = 1e-6,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=32,
    num_train_epochs=num_train_epochs,
    weight_decay=0.02,
    warmup_steps=50,
    remove_unused_columns=False,
    save_strategy = "epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor
)

trainer.train()
trainer.evaluate()
outputs = trainer.predict(test_data)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(axis=1)

def plot_confusion_matrix(cm, classes, title = 'confusion Matrix', cmap = plt.cm.Blues, figsize = (10, 8)):
    plt.figure(figsize = figsize)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    fmt = '.0f'
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

if len(labels_list) < 150:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes = labels_list, figsize=(8,6))

print()
print("Classification Report:")
print()
print(classification_report(y_true, y_pred, target_names = labels_list, digits = 4))
