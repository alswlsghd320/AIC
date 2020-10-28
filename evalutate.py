import argparse
import os
from model import VideoLearner
from dataset import VideoDataset

def get_parser():
    parser = argparse.ArgumentParser(description="Action recognition model for AI Championship")
    parser.add_argument(
        "--model_dir",
        default='./',
        help="Path to model config file",
    )
    parser.add_argument(
        "--video_dir",
        default='data',
        help="Path to video file",
    )
    parser.add_argument(
        "--model_name",
        metavar="FILE",
        help="Path to model config file, model name must follow this format; <model_name>_<epochs>.pt",
    )
    parser.add_argument(
        "--file_dir",
        default='./'
    )
    parser.add_argument(
        "--file_name",
        default='prediction.txt'
    )
    parser.add_argument(
        "--test_path",
        help="Text file to test the model.",
    )
    parser.add_argument(
        "--num_classes",
        default=56
    )
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args() #create_predict_txt

    DATA_ROOT = os.path.join(args.model_dir)
    VIDEO_DIR = os.path.join(DATA_ROOT, args.video_dir)
    Test_file = os.path.join(args.test_path)
    data = VideoDataset(
        VIDEO_DIR,
        train_split_file=Test_file,
        test_split_file=Test_file,
        batch_size=4,
        sample_length=32,
        # video_ext="avi"
    )

    learner = VideoLearner(data, num_classes=args.num_classes)
    learner.load(model_name=args.model_name, model_dir=args.model_dir)

    learner.create_predict_txt(file_path=args.file_path,
                               file_name=args.file_name,
                               )
