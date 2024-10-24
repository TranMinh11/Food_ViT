from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import DefaultDataCollator
import evaluate
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoImageProcessor, MobileViTForImageClassification


food_train = load_dataset("food101", split="train")
food_val = load_dataset("food101", split="validation")

labels = food_train.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


checkpoint = "apple/mobilevit-small"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

# normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
# _transforms = Compose([RandomResizedCrop(size), ToTensor()])
_transforms = Compose([RandomResizedCrop(size), ToTensor(), Normalize(mean=mean, std=std)])

##
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples
##

food_train_T = food_train.with_transform(transforms)
food_val_T = food_val.with_transform(transforms)

data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")

##
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
##

model = MobileViTForImageClassification.from_pretrained(
    checkpoint, num_labels=101, 
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="MobileViT_Food",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=25,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food_train_T,
    eval_dataset=food_val_T,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()











