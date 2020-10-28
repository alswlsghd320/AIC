import os
import warnings
import argparse

from model import VideoLearner
from dataset import VideoDataset

warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser(description="Action recognition model for AI Championship")
    parser.add_argument(
        "--video_dir",
        default='data',
        help="Path to video file",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default='train.txt',
        help="Train file to train the model. format: label/videos1 "
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default='test.txt',
        help="Text file to test the model. format: label/videos1 ",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=32
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )
    parser.add_argument(
        "--save_model",
        default=True
    )
    parser.add_argument(
        "--model_dir",
        default='./',
        help="Path to model config file",
    )
    parser.add_argument(
        "--model_name",
        default='model',
        help="Path to model config file, model name must follow this format; <model_name>_<epochs>.pt",
    )
    return parser

args = get_parser().parse_args()
# Number of consecutive frames used as input to the DNN. Recommended: 32 for high accuracy, 8 for inference speed.
MODEL_INPUT_SIZE = args.num_frames

# Batch size. Reduce if running out of memory.
BATCH_SIZE = args.batch_size

# Number of training epochs
EPOCHS = args.epochs

# Learning rate
LR = args.learning_rate

# Path to video data, train, test respectively
VIDEO_DIR = os.path.join(args.video_dir)
TRAIN_SPLIT = os.path.join(args.train_path)
TEST_SPLIT = os.path.join(args.test_path)

data = VideoDataset(
    VIDEO_DIR,
    train_split_file=TRAIN_SPLIT,
    test_split_file=TEST_SPLIT,
    batch_size=BATCH_SIZE,
    sample_length=MODEL_INPUT_SIZE,
    #video_ext="avi"
)

learner = VideoLearner(data, num_classes=56)

learner.fit(lr=LR, epochs=EPOCHS)


if args.save_model:
    sa = os.path.join(args.model_dir, "{model_name}_{epoch}.pt".format(model_name=args.model_name, epoch=EPOCHS))

    learner.save(sa)