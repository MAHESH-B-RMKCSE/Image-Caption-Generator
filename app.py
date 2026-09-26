import os
import pickle

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #08080c;
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #a855f7;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #c4b5fd;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #17131f;
        border: 1px solid #7c3aed;
        border-radius: 15px;
        padding: 25px;
        margin-top: 25px;
    }

    .caption-text {
        text-align: center;
        color: #e9d5ff;
        font-size: 23px;
        font-weight: 600;
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🖼️ AI Image Caption Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">VGG16 + LSTM Image Captioning</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "cnn_lstm_model.pth"
)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "vocab.pkl"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error("cnn_lstm_model.pth was not found.")

    st.code(MODEL_PATH)

    st.stop()


if not os.path.exists(VOCAB_PATH):

    st.error("vocab.pkl was not found.")

    st.code(VOCAB_PATH)

    st.stop()


# ============================================================
# VOCABULARY CLASS
# ============================================================

class Vocabulary:
    pass


# ============================================================
# LOAD VOCABULARY
# ============================================================

@st.cache_resource
def load_vocabulary():

    with open(
        VOCAB_PATH,
        "rb"
    ) as file:

        vocab = pickle.load(file)

    return vocab


try:

    vocab = load_vocabulary()

except Exception as error:

    st.error("Could not load vocabulary.")

    st.code(str(error))

    st.stop()


# ============================================================
# GET WORD TO INDEX
# ============================================================

def get_word_to_index(vocab):

    possible_names = [
        "word2idx",
        "word_to_idx",
        "stoi",
        "token_to_idx"
    ]

    for name in possible_names:

        if hasattr(vocab, name):

            value = getattr(
                vocab,
                name
            )

            if isinstance(value, dict):

                return value

    if isinstance(vocab, dict):

        for name in possible_names:

            if name in vocab:

                value = vocab[name]

                if isinstance(value, dict):

                    return value

    raise ValueError(
        "Could not find word-to-index vocabulary."
    )


# ============================================================
# GET INDEX TO WORD
# ============================================================

def get_index_to_word(vocab):

    possible_names = [
        "idx2word",
        "idx_to_word",
        "itos",
        "index_to_word"
    ]

    for name in possible_names:

        if hasattr(vocab, name):

            value = getattr(
                vocab,
                name
            )

            if isinstance(value, dict):

                return value

            if isinstance(value, list):

                return {
                    i: word
                    for i, word in enumerate(value)
                }

    if isinstance(vocab, dict):

        for name in possible_names:

            if name in vocab:

                value = vocab[name]

                if isinstance(value, dict):

                    return value

                if isinstance(value, list):

                    return {
                        i: word
                        for i, word in enumerate(value)
                    }

    raise ValueError(
        "Could not find index-to-word vocabulary."
    )


# ============================================================
# LOAD VOCABULARY MAPPINGS
# ============================================================

try:

    word_to_idx = get_word_to_index(
        vocab
    )

    idx_to_word = get_index_to_word(
        vocab
    )

    vocab_size = len(
        word_to_idx
    )

except Exception as error:

    st.error("Invalid vocabulary format.")

    st.code(str(error))

    st.stop()


# ============================================================
# SPECIAL TOKENS
# ============================================================

def find_token(
    possible_tokens,
    default_value
):

    for token in possible_tokens:

        if token in word_to_idx:

            return word_to_idx[token]

    return default_value


START_TOKEN = find_token(
    [
        "<start>",
        "<START>",
        "start"
    ],
    0
)

END_TOKEN = find_token(
    [
        "<end>",
        "<END>",
        "end"
    ],
    1
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class ImageCaptionModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        hidden_size=512
    ):

        super().__init__()

        self.image_projection = nn.Linear(
            4096,
            embed_size
        )

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(
        self,
        features,
        captions
    ):

        features = self.image_projection(
            features
        )

        embeddings = self.embedding(
            captions
        )

        features = features.unsqueeze(1)

        embeddings = torch.cat(
            (
                features,
                embeddings
            ),
            dim=1
        )

        outputs, _hidden_state = self.lstm(
            embeddings
        )

        outputs = self.fc(
            outputs
        )

        return outputs


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_caption_model():

    model = ImageCaptionModel(
        vocab_size=vocab_size,
        embed_size=256,
        hidden_size=512
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    return model


try:

    model = load_caption_model()

except Exception as error:

    st.error(
        "The trained caption model could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# LOAD VGG16 FEATURE EXTRACTOR
# ============================================================

@st.cache_resource
def load_vgg16():

    weights = models.VGG16_Weights.DEFAULT

    vgg = models.vgg16(
        weights=weights
    )

    class VGG16FeatureExtractor(nn.Module):

        def __init__(
            self,
            vgg_model
        ):

            super().__init__()

            self.features = (
                vgg_model.features
            )

            self.avgpool = (
                vgg_model.avgpool
            )

            self.classifier = nn.Sequential(
                *list(
                    vgg_model.classifier.children()
                )[:-1]
            )

        def forward(
            self,
            x
        ):

            x = self.features(
                x
            )

            x = self.avgpool(
                x
            )

            x = torch.flatten(
                x,
                start_dim=1
            )

            x = self.classifier(
                x
            )

            return x

    feature_extractor = VGG16FeatureExtractor(
        vgg
    )

    feature_extractor.to(
        device
    )

    feature_extractor.eval()

    return feature_extractor


try:

    vgg16 = load_vgg16()

except Exception as error:

    st.error(
        "VGG16 could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose(
    [
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
    ]
)


# ============================================================
# EXTRACT VGG16 FEATURES
# ============================================================

def extract_features(image):

    image = image.convert(
        "RGB"
    )

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        features = vgg16(
            image_tensor
        )

    return features


# ============================================================
# GENERATE CAPTION
# ============================================================

def generate_caption(
    image,
    max_length=34
):

    # --------------------------------------------------------
    # 1. Extract VGG16 features
    # Output: [1, 4096]
    # --------------------------------------------------------

    features = extract_features(
        image
    )

    # --------------------------------------------------------
    # 2. Project 4096 -> 256
    # --------------------------------------------------------

    image_features = model.image_projection(
        features
    )

    # --------------------------------------------------------
    # 3. Give image feature to LSTM
    # --------------------------------------------------------

    image_input = image_features.unsqueeze(
        1
    )

    with torch.no_grad():

        lstm_output, hidden_state = model.lstm(
            image_input
        )

    # --------------------------------------------------------
    # 4. Get hidden and cell states
    # --------------------------------------------------------

    hidden = hidden_state[0]

    cell = hidden_state[1]

    # --------------------------------------------------------
    # 5. Start caption with <start>
    # --------------------------------------------------------

    current_word = torch.tensor(
        [START_TOKEN],
        dtype=torch.long,
        device=device
    )

    generated_words = []

    # --------------------------------------------------------
    # 6. Generate caption word by word
    # --------------------------------------------------------

    for step in range(
        max_length
    ):

        with torch.no_grad():

            word_embedding = model.embedding(
                current_word
            )

            word_embedding = word_embedding.unsqueeze(
                1
            )

            output, hidden_state = model.lstm(
                word_embedding,
                (
                    hidden,
                    cell
                )
            )

            hidden = hidden_state[0]

            cell = hidden_state[1]

            output = model.fc(
                output.squeeze(1)
            )

            predicted_id = output.argmax(
                dim=1
            ).item()

        # ----------------------------------------------------
        # Stop when <end> is predicted
        # ----------------------------------------------------

        if predicted_id == END_TOKEN:

            break

        # ----------------------------------------------------
        # Convert ID to word
        # ----------------------------------------------------

        word = idx_to_word.get(
            predicted_id,
            "<unk>"
        )

        # ----------------------------------------------------
        # Ignore special tokens
        # ----------------------------------------------------

        if word not in [
            "<pad>",
            "<start>",
            "<end>",
            "<unk>"
        ]:

            generated_words.append(
                word
            )

        # ----------------------------------------------------
        # Next input word
        # ----------------------------------------------------

        current_word = torch.tensor(
            [predicted_id],
            dtype=torch.long,
            device=device
        )

    return " ".join(
        generated_words
    )


# ============================================================
# FIND DATASET
# ============================================================

if not os.path.exists(
    DATASET_DIR
):

    st.warning(
        "Dataset folder was not found."
    )

    st.code(
        DATASET_DIR
    )

    st.stop()


# ============================================================
# SEARCH DATASET FOR IMAGES
# ============================================================

image_files = []

for root, folders, files in os.walk(
    DATASET_DIR
):

    for file in files:

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):

            image_files.append(
                os.path.join(
                    root,
                    file
                )
            )


# ============================================================
# CHECK IMAGES
# ============================================================

if len(image_files) == 0:

    st.warning(
        "No images were found inside the dataset folder."
    )

    st.code(
        DATASET_DIR
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Model Information"
    )

    st.write(
        "**Model:** VGG16 + LSTM"
    )

    st.write(
        "**Dataset:** Flickr8k"
    )

    st.write(
        "**VGG16 Features:** 4096"
    )

    st.write(
        "**Projection:** 4096 → 256"
    )

    st.write(
        "**Embedding:** 256"
    )

    st.write(
        "**LSTM Hidden:** 512"
    )

    st.write(
        "**Vocabulary:** "
        + str(vocab_size)
    )

    st.write(
        "**Images Found:** "
        + str(len(image_files))
    )

    st.write(
        "**Device:** "
        + str(device)
    )

    st.markdown("---")

    st.info(
        "For best results, use images from the Flickr8k "
        "dataset because the model was trained on Flickr8k."
    )


# ============================================================
# IMAGE SELECTION
# ============================================================

st.markdown(
    "### 📂 Select a Flickr8k Image"
)

image_names = sorted(
    [
        os.path.basename(path)
        for path in image_files
    ]
)

selected_name = st.selectbox(
    "Choose an image",
    image_names
)


# ============================================================
# FIND SELECTED IMAGE
# ============================================================

selected_path = None

for path in image_files:

    if os.path.basename(path) == selected_name:

        selected_path = path

        break


if selected_path is None:

    st.error(
        "Selected image was not found."
    )

    st.stop()


# ============================================================
# OPEN IMAGE
# ============================================================

try:

    image = Image.open(
        selected_path
    ).convert(
        "RGB"
    )

except Exception as error:

    st.error(
        "Could not open the selected image."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# DISPLAY IMAGE
# ============================================================

st.markdown(
    "### 🖼️ Selected Image"
)

st.image(
    image,
    use_container_width=True
)


# ============================================================
# GENERATE CAPTION
# ============================================================

if st.button(
    "✨ Generate Caption",
    use_container_width=True
):

    with st.spinner(
        "Generating caption..."
    ):

        try:

            caption = generate_caption(
                image
            )

            if caption.strip() == "":

                st.warning(
                    "The model generated an empty caption."
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box">

                        <h3 style="text-align:center;">
                            🤖 Generated Caption
                        </h3>

                        <div class="caption-text">
                            {caption}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as error:

            st.error(
                "Caption generation failed."
            )

            st.code(
                str(error)
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>

    <p style="
        text-align:center;
        color:#71717a;
        font-size:14px;
    ">

        AI Image Caption Generator
        <br>
        VGG16 + LSTM • Flickr8k

    </p>
    """,
    unsafe_allow_html=True
)