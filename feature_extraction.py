import os
import pandas as pd
import clip
import numpy as np
from loguru import logger
from PIL import Image
import torch
import argparse
from tqdm import tqdm


class ExtractFeatures():

    def __init__(self, vision_model='RN50', device='cpu'):
        self.device = device
        self.vision_model = vision_model
        self.clip_model, self.preprocesser = self.load_clip_model(vision_model=self.vision_model, device=self.device)

    def load_clip_model(self, vision_model='RN50', device='cpu'):
        clip_model, preprocesser = clip.load(vision_model, device=device)
        print("CLIP model loaded successfully!")
        print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
        print(f"Input resolution: {clip_model.visual.input_resolution}")
        print(f"Context length: {clip_model.context_length}")
        print(f"Vocab size: {clip_model.vocab_size}")
        return clip_model, preprocesser

    def extract_image_and_text_features(self, img_path, text, device='cpu'):
        image = Image.open(img_path)
        if self.preprocesser is not None:
            image = self.preprocesser(image).unsqueeze(0).to(device)
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
        return image_features.detach().cpu(), text_features.detach().cpu()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract Image and Text Features")
    parser.add_argument("--json_data_path", type=str, help="Path to the data directory")
    parser.add_argument("--feature_save_dir", type=str, help="Path to save the features")
    parser.add_argument("--vision_model", type=str, default='RN50', help="CLIP vision model to use")
    parser.add_argument("--device", type=str, default='cpu', help="Device to run the model on")
    parser.add_argument("--json_save_path", type=str, help="Path to save the data with features")
    args = parser.parse_args()

    assert os.path.exists(args.json_data_path), "Data directory does not exist!"
    os.makedirs(args.feature_save_dir, exist_ok=True)

    feature_extractor = ExtractFeatures(vision_model=args.vision_model, device=args.device)

    data = pd.read_json(args.json_data_path)
    data = data.iloc[:100]
    # print(data)

    img_feat_list = []
    text_feat_list = []

    total = len(data)
    done = 0
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Extracting Features"):
        image_path = row['image_path']
        question = row['question']
        text_feat_name = row['image'].split('.')[0] + '_text.pt'
        img_feat_name = row['image'].split('.')[0] + '_img.pt'
        
        img_feat, text_feat = feature_extractor.extract_image_and_text_features(img_path=image_path,
                                        text=question,
                                        device=args.device
                                    )
        img_feat_path = os.path.join(args.feature_save_dir,img_feat_name)
        text_feat_path = os.path.join(args.feature_save_dir,text_feat_name)
        torch.save(img_feat, img_feat_path)
        torch.save(text_feat, text_feat_path)
        img_feat_list.append(img_feat_path)
        text_feat_list.append(text_feat_path)
        done+=1
        if done % 500 == 0:
            logger.info(f"Total={total} Done={done}")
    data['img_feat'] = img_feat_list
    data['text_feat'] = text_feat_list
    os.makedirs(os.path.dirname(args.json_save_path), exist_ok=True)
    data.to_json(args.json_save_path)
    logger.info(f"Saved to {args.json_save_path}")