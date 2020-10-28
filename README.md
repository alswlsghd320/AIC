# AI Championship : Action recogtion on MBN dataset

## Installation

```
git clone https://github.com/alswlsghd320/AIC.git 
cd AIC 
pip install -r requirement.txt

conda install ffmpeg

# Do NOT command 'pip install apex'
git clone https://www.github.com/nvidia/apex \
cd apex \
python3 setup.py install \
```

## Training
If you want to train this model, Enter this code.
```
python train.py --video_dir <vidio_path> --train_path <train.txt> test_path <test.txt>

'''
args:
    --num_frames <8 or 32> : Model Input Size (must be followed by 8 or 32)
    --epochs <Epochs> : Epochs to train
    --batch_size <Batch size> 
    --lr <Learning Rate> 
    --save_model : If true, save the trained model. Need to enter model_dir, model_name  
    --model_dir <model_dir> : you can enter the model_dir if you can save it where you want it. 
    --model_name <model_name> 
'''
```

## Evaluate
If you want to create prediction.txt to compare with the answer file, Enter this code.
```
python evaulate.py --video_dir <vidio_path> 
                   --model_dir <model_dir> 
                   --test_path <test.txt>
                   --model_name <model_name> : This format is followed by {model_epochs.pt}
                   --num_classes <num_classes=56> : In AIC task, the number of classes is 57.
                   --file_dir <file_dir> : You can save it where you want it.
                   --file_name <file_name> : default=prediction.txt
```

## Demo
The following example notebooks are provided: click [here](demo.ipynb) 
