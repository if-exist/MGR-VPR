
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=120,
                        help="Number of places in a batch. Each place consists of 4 images")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_step_epochs", type=int, default=3)
    parser.add_argument("--lr_mult", type=float, default=0.5)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--epochs_num", type=int, default=20,
                        help="number of epochs to train for")
    # Inference parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, required=True,
                        help="Path to load foundation model checkpoint.")
    parser.add_argument("--token_length", type=int, default=64)
    parser.add_argument("--topk", type=float, default=0.5)
    parser.add_argument("--num_query", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--channel_dim", type=int, default=256)
    parser.add_argument("--row_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_register", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=64, help="num_workers for all dataloaders")
    parser.add_argument('--train_resize', type=int, default=[224, 224], nargs=2, help="Training image size (HxW).")
    parser.add_argument('--test_resize', type=int, default=[336, 336], nargs=2, help="Evaluation image size (HxW).")
    parser.add_argument("--positive_dist_threshold", type=int, default=25)
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Paths parameters
    parser.add_argument("--train_dataset_folder", type=str, default=None, help="Path to GSV-Cities dataset")
    parser.add_argument("--eval_datasets_folder", type=str, default=None, help="Path with all evaluation datasets")
    parser.add_argument("--eval_dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")

    parser.add_argument("--save_dir", type=str, required=True,
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    args.features_dim = args.channel_dim * args.row_dim
    
    if args.eval_datasets_folder == None:
        raise ValueError("Please specify --eval_datasets_folder.")
    
    return args
