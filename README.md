# AI Image Caption Generator using Deep Learning

An AI-powered Image Caption Generator that automatically generates meaningful natural-language descriptions for images using a **CNN-LSTM deep learning architecture**.

The system uses **VGG16** as the CNN-based image feature extractor and **LSTM** for generating captions word by word. The model is implemented completely using **PyTorch** and trained using the **Flickr8k image-caption dataset**.

---

## 📌 Project Overview

Image captioning is a computer vision and natural language processing task where an AI system understands the visual content of an image and generates a corresponding textual description.

This project combines:

- **Computer Vision** → Extract visual features from images
- **Deep Learning** → Learn relationships between image features and words
- **Natural Language Processing** → Generate meaningful captions

### Example

**Input Image:**

A picture of a dog playing in a grassy field.

**Generated Caption:**

> "A dog is playing in the grass."

---

## 🚀 Features

- Image feature extraction using **VGG16**
- Caption generation using **LSTM**
- CNN-LSTM based architecture
- Flickr8k dataset support
- Caption preprocessing and vocabulary creation
- Start and end token handling
- Unknown word handling
- Automatic image preprocessing
- Model training using PyTorch
- Trained model saving
- Vocabulary saving
- Caption generation for new images
- CPU/GPU device detection
- Feature caching to reduce repeated computation

---

## 🧠 System Architecture

```text
                    Input Image
                         │
                         ▼
                  Image Preprocessing
                         │
                         ▼
                    VGG16 CNN
                         │
                         ▼
                Image Feature Vector
                         │
                         ▼
                 Feature Projection
                         │
                         ▼
              ┌─────────────────────┐
              │        LSTM         │
              │ Caption Generator   │
              └─────────────────────┘
                         │
                         ▼
                 Word Prediction
                         │
                         ▼
              Next Word Generation
                         │
                         ▼
                  Generated Caption


🔍 How It Works
1. Image Input

The user provides an image to the system.

The image is resized and normalized according to the requirements of the VGG16 network.

2. Feature Extraction

The pretrained VGG16 CNN processes the image and extracts high-level visual features.

The final classification layer is removed because the project requires image features rather than image-classification results.

3. Feature Projection

The extracted VGG16 feature vector is converted into a lower-dimensional representation suitable for the LSTM network.

4. Caption Processing

The Flickr8k captions are cleaned and converted into tokens.

Special tokens are used:

<pad>
<start>
<end>
<unk>

Example:

<start> a dog is running <end>
5. Word Embedding

Each word is converted into a numerical vector using an embedding layer.

6. LSTM Caption Generation

The LSTM receives the image representation and caption word embeddings.

It learns the relationship between the image and the sequence of words.

7. Caption Prediction

The model predicts one word at a time until the <end> token is generated.

Example:

<start>
       ↓
A
       ↓
A dog
       ↓
A dog is
       ↓
A dog is running
       ↓
<end>
📊 Dataset

This project uses the Flickr8k Dataset.

The dataset contains:

8,091 images
40,455 captions
Multiple captions for each image

The dataset is used for training the image caption generation model.

Dataset Structure
dataset/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│
└── captions.txt

The dataset images are not included in this GitHub repository because of their large size.

🛠️ Technologies Used
Technology	Purpose
Python	Programming language
PyTorch	Deep learning framework
Torchvision	VGG16 and image processing
VGG16	CNN feature extraction
LSTM	Caption sequence generation
NumPy	Numerical operations
Pandas	Data processing
Pillow	Image processing
tqdm	Progress monitoring
📁 Project Structure
Image-Caption-Generator/
│
├── dataset/
│   ├── images/
│   └── captions.txt
│
├── image_caption_pytorch.py
│
├── README.md
│
└── .gitignore
⚙️ Installation
1. Clone the repository
git clone https://github.com/MAHESH-B-RMKCSE/Image-Caption-Generator.git
2. Open the project
cd Image-Caption-Generator
3. Create a virtual environment
python -m venv ai_env
4. Activate the environment
Windows PowerShell
ai_env\Scripts\Activate.ps1
5. Install dependencies
pip install torch torchvision pandas numpy pillow tqdm
📂 Dataset Setup

Download the Flickr8k dataset and place the files in the following structure:

Image-Caption-Generator/
│
├── dataset/
│   ├── images/
│   └── captions.txt
│
└── image_caption_pytorch.py

The images folder should contain the Flickr8k images.

▶️ Running the Project

Run the main Python file:

python image_caption_pytorch.py

The program performs the following steps:

Load Dataset
     ↓
Clean Captions
     ↓
Build Vocabulary
     ↓
Load VGG16
     ↓
Extract Image Features
     ↓
Prepare Training Data
     ↓
Train CNN-LSTM Model
     ↓
Save Model
     ↓
Generate Caption
💾 Generated Files

During execution, the program may create files such as:

image_features.pkl
vocab.pkl
cnn_lstm_model.pth
image_features.pkl

Stores extracted VGG16 image features so that the features do not need to be extracted again every time.

vocab.pkl

Stores the vocabulary used by the caption-generation model.

cnn_lstm_model.pth

Stores the trained PyTorch model parameters.

🧪 Model Configuration

The project can be configured using parameters such as:

EMBED_SIZE = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 2
MIN_WORD_FREQ = 2
MAX_LENGTH = 34

The number of epochs can be increased for longer training and potentially better caption generation.

🧩 Model Components
VGG16

VGG16 is used as the CNN encoder.

Its responsibility is:

Image → Visual Features
Embedding Layer

Converts words into numerical vector representations:

Word → Vector
LSTM

The LSTM processes the sequence of words and learns the relationship between image features and caption sequences.

Image Features + Previous Words
              ↓
             LSTM
              ↓
        Next Word
Linear Layer

The final layer predicts the probability of each word in the vocabulary.

LSTM Output
     ↓
Linear Layer
     ↓
Vocabulary Probabilities
     ↓
Predicted Word
📈 Training Process

The training process follows:

Flickr8k Dataset
       ↓
Caption Cleaning
       ↓
Vocabulary Creation
       ↓
Image Preprocessing
       ↓
VGG16 Feature Extraction
       ↓
Training Dataset Creation
       ↓
CNN-LSTM Training
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Model Update

The trained model is saved as a PyTorch .pth file.

🎯 Applications

This project can be useful for:

Automatic image description
Accessibility applications
Assistive technology
Image search systems
Content management systems
Visual question-answering extensions
AI-powered multimedia applications
Computer vision and NLP research
🔮 Future Improvements

Possible improvements include:

Attention mechanism
Transformer-based caption generation
Beam search decoding
BLEU score evaluation
ROUGE evaluation
Larger image-caption datasets
Better pretrained vision models
Web-based user interface
Real-time image captioning
Multilingual caption generation
Image caption confidence scoring
📚 Concepts Covered

This project demonstrates practical knowledge of:

Convolutional Neural Networks
VGG16
Transfer Learning
Recurrent Neural Networks
LSTM
Word Embeddings
Natural Language Processing
Tokenization
Vocabulary Generation
Sequence Modeling
Deep Learning
PyTorch
Computer Vision
Image-to-Text Generation
👨‍💻 Author

Mahesh Bala Dath

B.E. Computer Science and Engineering
R.M.K. Engineering College

GitHub:
https://github.com/MAHESH-B-RMKCSE
