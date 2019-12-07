### Paraphrase Question Generator using Shared Discriminator

PyTorch code for Paraphrase Question Generator

#### Requirements

To create python environment for this code using [conda](https://docs.anaconda.com/anaconda/install/) it is recommended to use `env.txt` file.
```
$ conda create --name <env> --file env.txt
```
After that for logging you need to install [tensorboardX](https://github.com/lanpa/tensorboardX).
```
pip install tensorboardX
```
#### Dataset

You can directly use following files downloading them into `data` folder or by following the process shown below it.
###### Data Files
Download all the data files from here.
- [quora_data_prepro.h5](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_data_prepro.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_train.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_val.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_test.json](https://figshare.com/s/5463afb24cba05629cdf)


###### Download Dataset
We have referred  [neuraltalk2](https://github.com/karpathy/neuraltalk2) and [Text-to-Image Synthesis ](https://github.com/reedscot/icml2016) to prepare our code base. The first thing you need to do is to download the Quora Question Pairs dataset from the [Quora Question Pair website](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) and put the same in the `data` folder.

If you want to train from scratch continue reading or if you just want to evaluate using a pretrained model then head over to `Datafiles` section and download the data files (put all the data files in the `data` folder) and pretrained model( put this in the `pretrained` folder) and run `eval.lua`

Now we need to do some preprocessing, head over to the `prepro` folder and run

```
$ cd prepro
$ python quora_prepro.py
```

**Note** The above command generates json files for 100K question pairs for train, 5k question pairs for validation and 30K question pairs for Test set.
If you want to change this and instead use only 50K question pairs for training and rest remaining the same, then you need to make some minor changes in the above file. After this step, it will generate the files under the `data` folder. `quora_raw_train.json`, `quora_raw_val.json` and `quora_raw_test.json`

###### Preprocess Paraphrase Question

```
$ python prepro_quora.py --input_train_json ../data/quora_raw_train.json --input_test_json ../data/quora_raw_test.json
```
This will generate two files in `data/` folder, `quora_data_prepro.h5` and `quora_data_prepro.json`.

#### Training

```
$ python train.py --model <name of model> --n_epoch <number of epochs>
```

You can change training data set and validation data set lengths by adding arguments `--train_dataset_len` and `--val_dataset_len` which are default to `100000` and `30000` which is maximum.

There are other arguments also for you to experiment like `--batch_size`, `--learning_rate`, `--drop_prob_lm`, etc.

You can resume training using `--start_from` argument in which you have to give path of saved model.
#### Save and log

First you have to make empty directories `save`, `samples`, `logs` and `result`.  
For each training there will be a directory having unique name in `save`. Saved model will be a `.tar` file. Each model will be saved as `<epoch number>_-1.tar` in that directory.

In `samples` directory with same unique name as above the directory contains a `.txt` file for each epoch as `<epoch number>.txt` having generated paraphrases by model at the end of that epoch on validation data set.

Logs for training and evaluation is stored in `logs` directory which you can see using `tensorboard` by running following command.
```
tensorboard --logdir <path of logs directory>
```
This command will tell you where you can see your logs on browser, commonly it is `localhost:6006` but you can change it using `--port` argument in above command.
#### Evaluation

After finishing the training run the following command to get scores and generated paraphrases for models saved from `--start_from_file_-1.tar` to `--end_to_file_-1.tar`.
```
$ python score.py --start_from <unique folder name without trailing backslash> --start_from_file <epoch number to start evaluation with> --end_to_file <epoch number to end evaluation with>
```
This will be saved in `result` directory and file `<epoch number>.txt` contains generated paraphrases and `<epoch number>-score.txt` contains Bleu, METEOR, ROUGE_L and CIDEr scores.
