import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from misc import net_utils, utils
from misc.dataloader import Dataloader
from misc.train_util import dump_samples, evaluate_scores, save_model
from models.enc_dec_dis import ParaphraseGenerator


def main():

    # get arguments ---

    parser = utils.make_parser()
    args = parser.parse_args()

    # build model

    # # get data
    data = Dataloader(args.input_json, args.input_ques_h5)

    # # make op
    op = {
        "vocab_sz": data.getVocabSize(),
        "max_seq_len": data.getSeqLength(),
        "emb_hid_dim": args.emb_hid_dim,
        "emb_dim": args.emb_dim,
        "enc_dim": args.enc_dim,
        "enc_dropout": args.enc_dropout,
        "enc_rnn_dim": args.enc_rnn_dim,
        "gen_rnn_dim": args.gen_rnn_dim,
        "gen_dropout": args.gen_dropout,
        "lr": args.learning_rate,
        "epochs": args.n_epoch
    }

    # # instantiate paraphrase generator
    pgen = ParaphraseGenerator(op)

    # setup logging
    logger = SummaryWriter(os.path.join(LOG_DIR, TIME + args.name))
    # subprocess.run(['mkdir', os.path.join(GEN_DIR, TIME)], check=False)
    os.makedirs(os.path.join(GEN_DIR, TIME), exist_ok=True)
    # ready model for training

    train_loader = Data.DataLoader(
        Data.Subset(data, range(args.train_dataset_len)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = Data.DataLoader(Data.Subset(
        data,
        range(args.train_dataset_len,
              args.train_dataset_len + args.val_dataset_len)),
                                  batch_size=args.batch_size,
                                  shuffle=True)

    pgen_optim = optim.RMSprop(pgen.parameters(), lr=op["lr"])
    pgen.train()

    # train model
    pgen = pgen.to(DEVICE)
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

    for epoch in range(op["epochs"]):

        epoch_l1 = 0
        epoch_l2 = 0
        itr = 0
        ph = []
        pph = []
        gpph = []
        pgen.train()

        for phrase, phrase_len, para_phrase, para_phrase_len, _ in tqdm(
                train_loader, ascii=True, desc="epoch" + str(epoch)):

            phrase = phrase.to(DEVICE)
            para_phrase = para_phrase.to(DEVICE)

            out, enc_out, enc_sim_phrase = pgen(
                phrase.t(),
                sim_phrase=para_phrase.t(),
                train=True,
            )

            loss_1 = cross_entropy_loss(out.permute(1, 2, 0), para_phrase)
            loss_2 = net_utils.JointEmbeddingLoss(enc_out, enc_sim_phrase)

            pgen_optim.zero_grad()
            (loss_1 + loss_2).backward()

            pgen_optim.step()

            # accumulate results

            epoch_l1 += loss_1.item()
            epoch_l2 += loss_2.item()
            ph += net_utils.decode_sequence(data.ix_to_word, phrase)
            pph += net_utils.decode_sequence(data.ix_to_word, para_phrase)
            gpph += net_utils.decode_sequence(data.ix_to_word,
                                              torch.argmax(out, dim=-1).t())

            itr += 1
            torch.cuda.empty_cache()

        # log results

        logger.add_scalar("l2_train", epoch_l2 / itr, epoch)
        logger.add_scalar("l1_train", epoch_l1 / itr, epoch)

        scores = evaluate_scores(gpph, pph)

        for key in scores:
            logger.add_scalar(key + "_train", scores[key], epoch)

        dump_samples(ph, pph, gpph,
                     os.path.join(GEN_DIR, TIME,
                                  str(epoch) + "_train.txt"))

        # start validation

        epoch_l1 = 0
        epoch_l2 = 0
        itr = 0
        ph = []
        pph = []
        gpph = []
        pgen.eval()

        with torch.no_grad():
            for phrase, phrase_len, para_phrase, para_phrase_len, _ in tqdm(
                    test_loader, ascii=True, desc="val" + str(epoch)):

                phrase = phrase.to(DEVICE)
                para_phrase = para_phrase.to(DEVICE)

                out, enc_out, enc_sim_phrase = pgen(phrase.t(),
                                                    sim_phrase=para_phrase.t())

                loss_1 = cross_entropy_loss(out.permute(1, 2, 0), para_phrase)
                loss_2 = net_utils.JointEmbeddingLoss(enc_out, enc_sim_phrase)

                epoch_l1 += loss_1.item()
                epoch_l2 += loss_2.item()
                ph += net_utils.decode_sequence(data.ix_to_word, phrase)
                pph += net_utils.decode_sequence(data.ix_to_word, para_phrase)
                gpph += net_utils.decode_sequence(
                    data.ix_to_word,
                    torch.argmax(out, dim=-1).t())

                itr += 1
                torch.cuda.empty_cache()

            logger.add_scalar("l2_val", epoch_l2 / itr, epoch)
            logger.add_scalar("l1_val", epoch_l1 / itr, epoch)

            scores = evaluate_scores(gpph, pph)

            for key in scores:
                logger.add_scalar(key + "_val", scores[key], epoch)

            dump_samples(ph, pph, gpph,
                         os.path.join(GEN_DIR, TIME,
                                      str(epoch) + "_val.txt"))
        save_model(pgen, pgen_optim, epoch, os.path.join(SAVE_DIR, TIME, str(epoch)))

    # wrap ups
    logger.close()
    print("Done !!")


if __name__ == "__main__":

    LOG_DIR = 'logs'
    SAVE_DIR = 'save'
    GEN_DIR = 'samples'
    HOME = './'
    TIME = time.strftime("%Y%m%d_%H%M%S")
    DEVICE = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    main()
