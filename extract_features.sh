python feature_extraction.py \
        --json_data_path /home/ec2-user/vqa_project/data/train_df.json \
        --feature_save_dir /home/ec2-user/vqa_project/data/features_rn50/train/features \
        --vision_model RN50 \
        --device cuda:0 \
        --json_save_path /home/ec2-user/vqa_project/data/features_rn50/train/features.json

python feature_extraction.py \
        --json_data_path /home/ec2-user/vqa_project/data/test_df.json \
        --feature_save_dir /home/ec2-user/vqa_project/data/features_rn50/test/features \
        --vision_model RN50 \
        --device cuda:0 \
        --json_save_path /home/ec2-user/vqa_project/data/features_rn50/test/features.json

python feature_extraction.py \
        --json_data_path /home/ec2-user/vqa_project/data/val_df.json \
        --feature_save_dir /home/ec2-user/vqa_project/data/features_rn50/val/features \
        --vision_model RN50 \
        --device cuda:0 \
        --json_save_path /home/ec2-user/vqa_project/data/features_rn50/val/features.json