# -*- coding: utf-8 -*-
"""
ICASSP 2026

Original fine-tuning script is located at: https://colab.research.google.com/drive/19RsowsOKlgBThERZ9eQ-fiiveLTpEkQn
"""

# 1. Install system libraries like FFmpeg
!apt-get update && apt-get install -y ffmpeg

# 2. Install primary Python packages, pinning fsspec and gcsfs to compatible versions
!pip install "fsspec<=2023.5.0" "gcsfs<=2023.5.0"
!pip install datasets transformers tensorboard jiwer wandb evaluate accelerate loguru

# 3. Install compatible PyTorch, torchvision, and torchcodec libraries
!pip install torch torchaudio torchvision torchcodec --extra-index-url https://download.pytorch.org/whl/cu121

import sys
import time
from loguru import logger
import os

### Setting up logging ###########################################
# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Remove default handlers and add stdout with INFO level
logger.remove()
logger.add(sys.stdout, level="INFO")

from pathlib import Path

# Define a custom rotation policy for file logging
def rotation_policy(message, file):
    file_path = Path(file.name)  # Extract the file path
    if not file_path.exists():
        return False  # If the file doesn't exist, no rotation needed

    # Rotate if file is older than 1 day (86400 seconds) or if size exceeds 10 MB
    if time.time() - file_path.stat().st_mtime > 86400:
        return True
    if file_path.stat().st_size > 10 * 1024 * 1024:
        return True
    return False

# Add a file sink with the custom rotation policy
logger.add("logs/script.log", rotation=rotation_policy, level="INFO")
logger.info("### Script started")

### Settings #####################################################
model_name = "/content/drive/MyDrive/Phd/Paper2/converted_wav2vec2_models/xlsr2_300m_1000h_faroese/checkpoint_best"
repo_name = "wav2vec2-xls-r-300m-cpt-1000h-FO-cp-best-faroese-100h-30-epochs"
asr_data_set = "carlosdanielhernandezmena/ravnursson_asr"

num_train_epochs = 30
save_steps = 5_000
eval_steps = 1_000
warmup_steps = 5_000
learning_rate = 1e-4

#### Print of settings ###################################
logger.info(f"### Model name: {model_name}")
logger.info(f"### Repo name: {repo_name}")
logger.info(f"### Number of epochs: {num_train_epochs}")
logger.info(f"### Learning rate: {learning_rate}")
logger.info(f"### ASR dataset: {asr_data_set}")
##################################################################

## Log-in to Hugging Face and Wand ###############################
from huggingface_hub import login
huggingface_token = ""
login(token=huggingface_token)

import wandb
wandb.login(key="")
##################################################################

## Prepare data, tokenizer, feature extractor
from datasets import load_dataset, Audio

data_train = load_dataset(asr_data_set, split="train")
data_validation = load_dataset(asr_data_set, split="validation")

logger.info(data_train)
logger.info(data_validation)

logger.info("### Removing columns...")
data_train = data_train.remove_columns(["audio_id", "speaker_id", "gender", "age", "duration", "dialect"])
data_validation = data_validation.remove_columns(["audio_id", "speaker_id", "gender", "age", "duration", "dialect"])
logger.info("### Renaming column...")
data_train = data_train.rename_column("normalized_text", "text")
data_validation = data_validation.rename_column("normalized_text", "text")

logger.info(data_train)
logger.info(data_validation)

import re
chars_to_ignore_regex = r'[,?!_;:"“%‘”�><-]'
logger.info(f"Removing 'Characters to ignore' ({chars_to_ignore_regex})...")

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

data_train = data_train.map(remove_special_characters)
data_validation = data_validation.map(remove_special_characters)

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = data_train.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=data_train.column_names
)
vocab_test = data_validation.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=data_validation.column_names
)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
logger.info(f"Vocab_dict: {vocab_dict}")
len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
tokenizer.push_to_hub(repo_name) # Push to Huggingface

## XLSR-Wav2Vec2 Feature Extractor
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

## Preprocess data
logger.info("Preprocessing data...")
logger.info("First sample in train dataset:")
logger.info(data_train[0])

# Resample to 16khz
data_train = data_train.cast_column("audio", Audio(sampling_rate=16_000))
data_validation = data_validation.cast_column("audio", Audio(sampling_rate=16_000))

def prepare_dataset(batch):
    audio = batch["audio"]

    # Process the audio inputs
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    # Process the text labels in the same call using the `text` argument
    processed = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["text"]
    )
    batch["labels"] = processed["labels"]

    return batch


data_train = data_train.map(prepare_dataset, remove_columns=data_train.column_names)
data_validation = data_validation.map(prepare_dataset, remove_columns=data_validation.column_names)

## Set up trainer
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

import numpy as np
import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor()

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    # per_device_eval_batch_size=1,
    fp16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=25,
    hub_strategy="every_save",
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    push_to_hub=True,
    report_to=["tensorboard", "wandb"],
    greater_is_better=False,
    dataloader_num_workers=16,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data_train,
    eval_dataset=data_validation,
    processing_class=processor.feature_extractor,
)

## Training
logger.info("### Training started")
trainer.train()

logger.info("### Push model to hub")
trainer.push_to_hub()

logger.info("### Script ended")