import torch 
from torch.utils.data import Dataset
import os 
import pandas as pd
from loguru import logger 
from PIL import Image


class VizWizDatasetCLIP(Dataset):
    def __init__(self, 
                df_path, 
                filter_answerable=True):
        
        assert os.path.exists(df_path), f"{df_path} does not exists"
        self.df = pd.read_json(df_path)
        # using only Answerable Instances
        if filter_answerable:
            logger.info("Filtering Answerable Question Only")
            self.df = self.df[self.df.answerable == 1]

        self.img_feats = self.df.img_feat
        self.text_feats = self.df.text_feat
        self.answers = self.df.final_answer
        self.anserable = self.df

    def __len__(self):
        return len(self.df)
                
    def __getitem__(self, index):
        img_feat = torch.load(self.df.img_feat.iloc[index])
        text_feat = torch.load(self.df.text_feat.iloc[index])
        # Concatenate Image and Text Features
        feat = torch.cat((img_feat, text_feat), 1).to(torch.float32)

        answer = self.df.final_answer.iloc[index]
        answerability = self.df.answerable.iloc[index]       
        return index, feat, answer, answerability
    

class VizWizDatasetViLT(Dataset):
    def __init__(self, 
                 data, 
                 processor, 
                 label2id):
        
        print(f"Actual Data Size : {data.shape[0]}")
        #self.data = data[data.answerable == 1]
        self.data = data 
        #self.data = self.data.iloc[:128]
        print(f"Filtered Data Size : {self.data.shape[0]}")
        self.image_paths = self.data.image_path.tolist()
        self.labels = self.data.labels.tolist()
        self.scores = self.data.scores.tolist()
        self.questions = self.data.question.tolist()
        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.image_paths)
    
    

    def __getitem__(self, idx):
        # get image + text
        image = Image.open(self.image_paths[idx])
        #augmenter = RandAugment(n=2, m=9)
        #image = augmenter(image)
        text = self.questions[idx]

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        labels = self.labels[idx]
        scores = self.scores[idx]
        # # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(self.label2id))
        for label, score in zip(labels, scores):
              targets[label] = score
        encoding["labels"] = targets
        return encoding
    
    def collate_fn(self, batch):

        input_ids = [item['input_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # create padded pixel values and corresponding pixel mask
        encoding = self.processor.image_processor.pad(pixel_values, return_tensors="pt")

        # create new batch
        batch = {}
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['token_type_ids'] = torch.stack(token_type_ids)
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = torch.stack(labels)

        return batch
