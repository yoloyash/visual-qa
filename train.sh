# 
python train_clip.py \
        --train_json_path /home/ec2-user/vqa-project/data/features_rn50/train/features.json \
        --val_json_path /home/ec2-user/vqa-project/data/features_rn50/val/features.json \
        --model_save_dir /home/ec2-user/vqa-project/runs \
        --device cuda:0 \
        --model_name clip-rn50-1 \
        --epochs 100 \
        --batch_size 256 \
        --lr 0.001 \
        --input_dim 2048 \
        --hidden_dim 2048 \
        --optimizer adam

# 
python train_vilt.py \
        --class_mapping_file_path /home/ec2-user/vqa-project/data/class_mapping.csv \
        --train_json_path /home/ec2-user/vqa-project/data/features_rn50/train/features.json \
        --val_json_path /home/ec2-user/vqa-project/data/features_rn50/val/features.json \
        --model_save_dir /home/ec2-user/vqa-project/runs \
        --device cuda:0 \
        --model_name vilt-final \
        --batch_size 64 \
        --lr 0.005 \
        --epochs 15
