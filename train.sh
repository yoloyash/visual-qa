# parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
#     parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
#     parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
#     parser.add_argument("--input_dim", type=int, default=2048, help="Input dimension for the model")
#     parser.add_argument("--hidden_dim", type=int, default=2048, help="Hidden dimension for the model")
#     parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training")
#     parser.add_argument("--model_name", type=str, default="clip-rn50-1", help="Name of the model file")
#     parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
#     # data related arguments
#     parser.add_argument("--train_json_path", type=str, help="Path to the training data")
#     parser.add_argument("--val_json_path", type=str, help="Path to the validation data")
#     parser.add_argument("--model_save_dir", type=str, default="/home/ec2-user/vqa_project/runs", help="Path to save the model")


python train.py \
        --train_json_path /home/ec2-user/vqa_project/data/features_rn50/train/features.json \
        --val_json_path /home/ec2-user/vqa_project/data/features_rn50/val/features.json \
        --model_save_dir /home/ec2-user/vqa_project/runs \
        --device cuda:0 \
        --model_name clip-rn50-1 \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.001 \
        --input_dim 2048 \
        --hidden_dim 2048 \
        --optimizer adam