import os
import re
import pickle
import random
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")

# Change this automatically if your captions file has another name
CAPTION_FILE = os.path.join(DATASET_DIR, "captions.txt")

FEATURE_FILE = os.path.join(BASE_DIR, "image_features.pkl")
VOCAB_FILE = os.path.join(BASE_DIR, "vocab.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "cnn_lstm_model.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print(" AI IMAGE CAPTION GENERATOR")
print(" CNN + VGG16 + LSTM")
print(" PyTorch Version")
print("======================================")
print()
print("Device:", DEVICE)


# ============================================================
# SETTINGS
# ============================================================

EMBED_SIZE = 256
HIDDEN_SIZE = 512

BATCH_SIZE = 32

# For quick testing use 2.
# For better training use 10-20.
EPOCHS = 2

LEARNING_RATE = 0.001

MIN_WORD_FREQ = 2

MAX_LENGTH = 34


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_caption(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z ]",
        "",
        text
    )

    words = text.split()

    words = [
        word for word in words
        if len(word) > 1
    ]

    return " ".join(words)


# ============================================================
# LOAD FLICKR8K CAPTIONS
# ============================================================

def load_captions():

    print("\nLoading captions...")

    if not os.path.exists(CAPTION_FILE):

        print(
            "\nERROR: captions.txt not found."
        )

        print(
            "Expected:"
        )

        print(
            CAPTION_FILE
        )

        exit()

    df = pd.read_csv(
        CAPTION_FILE,
        sep=",",
        header=None,
        names=[
            "image",
            "caption"
        ],
        engine="python"
    )

    # Remove header if present
    first_value = str(
        df.iloc[0]["image"]
    ).lower()

    if "image" in first_value:

        df = df.iloc[1:]

    captions = {}

    for _, row in df.iterrows():

        image_name = str(
            row["image"]
        ).strip()

        caption = str(
            row["caption"]
        ).strip()

        # Flickr8k sometimes uses:
        # image.jpg#0
        image_name = image_name.split("#")[0]

        caption = clean_caption(
            caption
        )

        caption = (
            "<start> "
            + caption
            + " <end>"
        )

        if image_name not in captions:

            captions[image_name] = []

        captions[image_name].append(
            caption
        )

    print(
        "Images:",
        len(captions)
    )

    print(
        "Captions:",
        sum(
            len(v)
            for v in captions.values()
        )
    )

    return captions


# ============================================================
# VOCABULARY
# ============================================================

class Vocabulary:

    def __init__(
        self,
        min_freq=2
    ):

        self.word2idx = {

            "<pad>": 0,

            "<start>": 1,

            "<end>": 2,

            "<unk>": 3
        }

        self.idx2word = {

            0: "<pad>",

            1: "<start>",

            2: "<end>",

            3: "<unk>"
        }

        self.min_freq = min_freq

    def build(
        self,
        captions
    ):

        counter = Counter()

        for image_name in captions:

            for caption in captions[
                image_name
            ]:

                words = caption.split()

                counter.update(words)

        for word, count in counter.items():

            if count >= self.min_freq:

                index = len(
                    self.word2idx
                )

                self.word2idx[word] = index

                self.idx2word[index] = word

        print(
            "Vocabulary size:",
            len(self.word2idx)
        )

    def encode(
        self,
        caption
    ):

        result = []

        for word in caption.split():

            if word in self.word2idx:

                result.append(
                    self.word2idx[word]
                )

            else:

                result.append(
                    self.word2idx["<unk>"]
                )

        return result

    def __len__(self):

        return len(
            self.word2idx
        )


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# VGG16
# ============================================================

def create_vgg():

    print("\nLoading VGG16...")

    weights = (
        models.VGG16_Weights.DEFAULT
    )

    vgg = models.vgg16(
        weights=weights
    )

    # Remove final classification layer
    vgg.classifier = nn.Sequential(
        *list(
            vgg.classifier.children()
        )[:-1]
    )

    vgg = vgg.to(DEVICE)

    vgg.eval()

    return vgg


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    captions
):

    if os.path.exists(
        FEATURE_FILE
    ):

        print(
            "\nExisting image features found."
        )

        print(
            "Loading:",
            FEATURE_FILE
        )

        with open(
            FEATURE_FILE,
            "rb"
        ) as f:

            return pickle.load(f)

    vgg = create_vgg()

    features = {}

    print(
        "\nExtracting VGG16 features..."
    )

    with torch.no_grad():

        for image_name in tqdm(
            captions
        ):

            image_path = os.path.join(
                IMAGE_DIR,
                image_name
            )

            if not os.path.exists(
                image_path
            ):

                continue

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                image = transform(
                    image
                )

                image = image.unsqueeze(
                    0
                )

                image = image.to(
                    DEVICE
                )

                feature = vgg(
                    image
                )

                feature = feature.squeeze(
                    0
                )

                features[
                    image_name
                ] = feature.cpu().numpy()

            except Exception as e:

                print(
                    "\nError:",
                    image_name,
                    e
                )

    with open(
        FEATURE_FILE,
        "wb"
    ) as f:

        pickle.dump(
            features,
            f
        )

    print(
        "\nFeatures saved:"
    )

    print(
        FEATURE_FILE
    )

    print(
        "Features:",
        len(features)
    )

    return features


# ============================================================
# DATASET
# ============================================================

class CaptionDataset(
    Dataset
):

    def __init__(
        self,
        captions,
        features,
        vocab
    ):

        self.data = []

        self.features = features

        self.vocab = vocab

        for image_name in captions:

            if image_name not in features:

                continue

            for caption in captions[
                image_name
            ]:

                tokens = vocab.encode(
                    caption
                )

                if len(tokens) > MAX_LENGTH:

                    tokens = tokens[
                        :MAX_LENGTH
                    ]

                self.data.append(
                    (
                        image_name,
                        tokens
                    )
                )

    def __len__(self):

        return len(
            self.data
        )

    def __getitem__(
        self,
        index
    ):

        image_name, tokens = (
            self.data[index]
        )

        feature = torch.tensor(

            self.features[
                image_name
            ],

            dtype=torch.float32
        )

        caption = torch.tensor(

            tokens,

            dtype=torch.long
        )

        return (
            feature,
            caption
        )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(
    batch
):

    features = []

    captions = []

    for feature, caption in batch:

        features.append(
            feature
        )

        captions.append(
            caption
        )

    features = torch.stack(
        features
    )

    max_length = max(
        len(caption)
        for caption in captions
    )

    padded_captions = []

    for caption in captions:

        padding = torch.zeros(

            max_length
            - len(caption),

            dtype=torch.long
        )

        padded = torch.cat(

            [
                caption,
                padding
            ]
        )

        padded_captions.append(
            padded
        )

    captions = torch.stack(
        padded_captions
    )

    return (
        features,
        captions
    )


# ============================================================
# CNN + LSTM MODEL
# ============================================================

class CNN_LSTM(
    nn.Module
):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        # VGG16 gives 4096 features

        self.image_projection = nn.Linear(

            4096,

            EMBED_SIZE
        )

        self.embedding = nn.Embedding(

            vocab_size,

            EMBED_SIZE
        )

        self.lstm = nn.LSTM(

            EMBED_SIZE,

            HIDDEN_SIZE,

            batch_first=True
        )

        self.fc = nn.Linear(

            HIDDEN_SIZE,

            vocab_size
        )

    def forward(
        self,
        image_features,
        captions
    ):

        image_features = (
            self.image_projection(
                image_features
            )
        )

        image_features = (
            image_features.unsqueeze(1)
        )

        word_embeddings = (
            self.embedding(
                captions
            )
        )

        inputs = torch.cat(

            [
                image_features,
                word_embeddings
            ],

            dim=1
        )

        outputs, _ = self.lstm(
            inputs
        )

        outputs = self.fc(
            outputs
        )

        return outputs


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    dataloader
):

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    print(
        "\n======================================"
    )

    print(
        "STARTING TRAINING"
    )

    print(
        "======================================"
    )

    model.train()

    for epoch in range(
        EPOCHS
    ):

        total_loss = 0

        progress = tqdm(

            dataloader,

            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            )
        )

        for features, captions in progress:

            features = features.to(
                DEVICE
            )

            captions = captions.to(
                DEVICE
            )

            # Input
            inputs = captions[:, :-1]

            # Expected output
            targets = captions[:, 1:]

            optimizer.zero_grad()

            outputs = model(

                features,

                inputs
            )

            # Remove image token
            outputs = outputs[:, 1:, :]

            outputs = outputs.reshape(

                -1,

                outputs.size(-1)
            )

            targets = targets.reshape(
                -1
            )

            loss = criterion(

                outputs,

                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            progress.set_postfix(

                loss=round(
                    loss.item(),
                    4
                )
            )

        average_loss = (

            total_loss
            / len(dataloader)
        )

        print(

            f"\nEpoch "
            f"{epoch + 1} "
            f"Loss: "
            f"{average_loss:.4f}"
        )

    torch.save(

        model.state_dict(),

        MODEL_FILE
    )

    print(
        "\nModel saved:"
    )

    print(
        MODEL_FILE
    )


# ============================================================
# LOAD IMAGE FOR PREDICTION
# ============================================================

def get_image_feature(
    image_path
):

    vgg = create_vgg()

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(
        image
    )

    image = image.unsqueeze(
        0
    )

    image = image.to(
        DEVICE
    )

    with torch.no_grad():

        feature = vgg(
            image
        )

    return feature


# ============================================================
# GENERATE CAPTION
# ============================================================

def generate_caption(

    model,

    image_path,

    vocab
):

    model.eval()

    feature = get_image_feature(
        image_path
    )

    start_token = (
        vocab.word2idx[
            "<start>"
        ]
    )

    end_token = (
        vocab.word2idx[
            "<end>"
        ]
    )

    current = torch.tensor(

        [[start_token]],

        dtype=torch.long

    ).to(DEVICE)

    result = []

    for _ in range(
        MAX_LENGTH
    ):

        with torch.no_grad():

            output = model(

                feature,

                current
            )

        output = output[:, -1, :]

        predicted = output.argmax(
            dim=1
        ).item()

        if predicted == end_token:

            break

        word = vocab.idx2word.get(

            predicted,

            "<unk>"
        )

        if word not in [

            "<start>",

            "<pad>",

            "<unk>"
        ]:

            result.append(
                word
            )

        next_token = torch.tensor(

            [[predicted]],

            dtype=torch.long

        ).to(DEVICE)

        current = torch.cat(

            [
                current,
                next_token
            ],

            dim=1
        )

    return " ".join(
        result
    )


# ============================================================
# MAIN
# ============================================================

def main():

    captions = load_captions()

    # ---------------------------------
    # Vocabulary
    # ---------------------------------

    vocab = Vocabulary(
        MIN_WORD_FREQ
    )

    vocab.build(
        captions
    )

    with open(
        VOCAB_FILE,
        "wb"
    ) as f:

        pickle.dump(
            vocab,
            f
        )

    # ---------------------------------
    # VGG16 Features
    # ---------------------------------

    features = extract_features(
        captions
    )

    # ---------------------------------
    # Dataset
    # ---------------------------------

    dataset = CaptionDataset(

        captions,

        features,

        vocab
    )

    print(
        "\nTraining samples:",
        len(dataset)
    )

    dataloader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        collate_fn=collate_fn,

        num_workers=0
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = CNN_LSTM(
        len(vocab)
    )

    model = model.to(
        DEVICE
    )

    # ---------------------------------
    # Train
    # ---------------------------------

    train_model(

        model,

        dataloader
    )

    # ---------------------------------
    # Prediction
    # ---------------------------------

    print(
        "\n======================================"
    )

    print(
        "CAPTION GENERATION"
    )

    print(
        "======================================"
    )

    image_path = input(

        "\nEnter image path: "
    ).strip()

    image_path = image_path.strip(
        '"'
    )

    image_path = image_path.strip(
        "'"
    )

    if not os.path.exists(
        image_path
    ):

        print(
            "\nImage not found."
        )

        return

    caption = generate_caption(

        model,

        image_path,

        vocab
    )

    print(
        "\nGenerated Caption:"
    )

    print(
        caption
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()