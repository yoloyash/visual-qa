# Visual Question Answering

This repository features the implementation of a straightforward Visual Question Answering (VQA) model and benchmarks its performance against state-of-the-art models such as ViLT. It was created as a part of the ***MSAI 337: Natural Language Processing course*** at ***Northwestern University.***


## Dataset

This project uses the [VizWiz Visual Question Answering dataset](https://vizwiz.org/tasks-and-datasets/vqa/). 

A sample visual question looks like this - 

```javascript
"answerable": 0,
"image": "VizWiz_val_00028000.jpg",
"question": "What is this?"
"answer_type": "unanswerable",
"answers": [
    {"answer": "unanswerable", "answer_confidence": "yes"},
    {"answer": "chair", "answer_confidence": "yes"},
    {"answer": "unanswerable", "answer_confidence": "yes"},
    {"answer": "unanswerable", "answer_confidence": "no"},
    {"answer": "unanswerable", "answer_confidence": "yes"},
    {"answer": "text", "answer_confidence": "maybe"},
    {"answer": "unanswerable", "answer_confidence": "yes"},
    {"answer": "bottle", "answer_confidence": "yes"},
    {"answer": "unanswerable", "answer_confidence": "yes"},
    {"answer": "unanswerable", "answer_confidence": "yes"}
]
```

![VizWiz Dataset](https://github.com/yoloyash/visual-qa/blob/main/assets/dataset.png)

## Trained Models

Download weights for trained models [here](https://drive.google.com/drive/folders/1HIXI_O-wuNhxLduvE7efWb6Nt-yur8cN?usp=drive_link)
(Blip missing)

## Installation

To set up the project environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yoloyash/visual-qa
cd visual-qa

# Create a virtual environment (optional)
conda create -n vqa python=3.9
conda activate vqa

# Install the required dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install chardet
pip install -r requirements.txt
```


### To-do
- ~~dataset analysis~~
- ~~feature extraction~~
- ~~training code for feature-based model~~
- ~~performance evaluation~~
- ~~finetune ViLT for VQA~~
- ~~finetune BLIP for VQA~~ (need to train for longer)
- ~~tensorboard support~~
- ~~link to trained models and encoders~~
- performance comparison
- streamlit webui 