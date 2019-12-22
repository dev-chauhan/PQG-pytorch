## Paraphrase Question Generator using Shared Discriminator

PyTorch code for Paraphrase Question Generator.  This code-based is built upon this paper [Learning Semantic Sentence Embeddings using Pair-wise Discriminator](https://www.aclweb.org/anthology/C18-1230.pdf).

### Requirements and Setup

##### Use Anaconda or Miniconda

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site](https://conda.io/docs/user-guide/install/download.html).
2. Clone this repository and create an environment:

```
git clone https://www.github.com/dev-chauhan/PQG-pytorch
cd PQG-pytorch
conda create -n <env name> --file env.txt

# activate the environment
conda activate <env name>
```
3. After that for logging you need to install [tensorboardX](https://github.com/lanpa/tensorboardX).
```
pip install tensorboardX
```
### Dataset

You can directly use following files downloading them into `data` folder or by following the process shown below it.
##### Data Files
Download all the data files from here.
- [quora_data_prepro.h5](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_data_prepro.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_train.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_val.json](https://figshare.com/s/5463afb24cba05629cdf)
- [quora_raw_test.json](https://figshare.com/s/5463afb24cba05629cdf)


##### Download Dataset
We have referred  [neuraltalk2](https://github.com/karpathy/neuraltalk2) and [Text-to-Image Synthesis ](https://github.com/reedscot/icml2016) to prepare our code base. The first thing you need to do is to download the Quora Question Pairs dataset from the [Quora Question Pair website](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) and put the same in the `data` folder.

If you want to train from scratch continue reading or if you just want to evaluate using a pretrained model then head over to `Datafiles` section and download the data files (put all the data files in the `data` folder) and run `score.py` to evaluate pretrained model.

Now we need to do some preprocessing, head over to the `prepro` folder and run

```
$ cd prepro
$ python quora_prepro.py
```

**Note** The above command generates json files for 100K question pairs for train, 5k question pairs for validation and 30K question pairs for Test set.
If you want to change this and instead use only 50K question pairs for training and rest remaining the same, then you need to make some minor changes in the above file. After this step, it will generate the files under the `data` folder. `quora_raw_train.json`, `quora_raw_val.json` and `quora_raw_test.json`

##### Preprocess Paraphrase Question

```
$ python prepro_quora.py --input_train_json ../data/quora_raw_train.json --input_test_json ../data/quora_raw_test.json
```
This will generate two files in `data/` folder, `quora_data_prepro.h5` and `quora_data_prepro.json`.

### Training

```
$ python train.py --model <name of model> --n_epoch <number of epochs>
```

You can change training data set and validation data set lengths by adding arguments `--train_dataset_len` and `--val_dataset_len` which are default to `100000` and `30000` which is maximum.

There are other arguments also for you to experiment like `--batch_size`, `--learning_rate`, `--drop_prob_lm`, etc.

You can resume training using `--start_from` argument in which you have to give path of saved model.
### Save and log

First you have to make empty directories `save`, `samples`, `logs` and `result`.  
For each training there will be a directory having unique name in `save`. Saved model will be a `.tar` file. Each model will be saved as `<epoch number>_-1.tar` in that directory.

In `samples` directory with same unique name as above the directory contains a `.txt` file for each epoch as `<epoch number>.txt` having generated paraphrases by model at the end of that epoch on validation data set.

Logs for training and evaluation is stored in `logs` directory which you can see using `tensorboard` by running following command.
```
tensorboard --logdir <path of logs directory>
```
This command will tell you where you can see your logs on browser, commonly it is `localhost:6006` but you can change it using `--port` argument in above command.
### Evaluation

After finishing the training run the following command to get scores and generated paraphrases for models saved from `--start_from_file_-1.tar` to `--end_to_file_-1.tar`.
```
$ python score.py --start_from <unique folder name without trailing backslash> --start_from_file <epoch number to start evaluation with> --end_to_file <epoch number to end evaluation with>
```
This will be saved in `result` directory and file `<epoch number>.txt` contains generated paraphrases and `<epoch number>-score.txt` contains Bleu, METEOR, ROUGE_L and CIDEr scores.

### Results
Following are the results for 100k quora question pairs dataset for some models.

Name of model | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | ROUGE_L | METEOR | CIDEr |
---|--|--|--|--|--|--|--|
EDL|0.4162|0.2578|0.1724|0.1219|0.4191|0.3244|0.6189|
EDLPS|0.4754|0.3160|0.2249|0.1672|0.4781|0.3488|1.0949|

Following are the results for 50k quora question pairs dataset for some models.

Name of model | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | ROUGE_L | METEOR | CIDEr |
---|--|--|--|--|--|--|--|
EDL|0.3877|0.2336|0.1532|0.1067|0.3913|0.3133|0.4550|
EDLPS|0.4553|0.2981 |0.2105|0.1560|0.4583|0.3421|0.9690|


### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@inproceedings{patro2018learning,
  title={Learning Semantic Sentence Embeddings using Sequential Pair-wise Discriminator},
  author={Patro, Badri Narayana and Kurmi, Vinod Kumar and Kumar, Sandeep and Namboodiri, Vinay},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics},
  pages={2715--2729},
  year={2018}
}
```

## Contributors
* [Dev  Chauhan][1] (devgiri@iitk.ac.in)
* [Badri N. Patro][2] (badri@iitk.ac.in)
* [Vinod K. Kurmi][2] (vinodkk@iitk.ac.in)

[1]: https://github.com/dev-chauhan
[2]: https://github.com/badripatro
[3]: https://github.com/vinodkkurmi



