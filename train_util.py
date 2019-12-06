import torch
import misc.utils as utils
import misc.net_utils as net_utils
import torch.nn as nn
from pycocoevalcap.eval import COCOEvalCap
import os


def getObjsForScores(real_sents, pred_sents):
    class coco:
        def __init__(self, sents):
            self.sents = sents
            self.imgToAnns = [[{'caption': sents[i]}] for i in range(len(sents))]

        def getImgIds(self):
            return [i for i in range(len(self.sents))]

    return coco(real_sents), coco(pred_sents)


def save_model(encoder, generator, model_optim, epoch, it, local_loss, global_loss, save_folder, folder, discriminator=None, discriminatorg=None):

    PATH = os.path.join(save_folder, folder, str(epoch) + '_' + str(it) + '.tar')

    checkpoint = {
        'epoch': epoch,
        'iter': it,
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'optimizer_state_dict': model_optim.state_dict(),
        'local_loss': local_loss,
        'global_loss': global_loss
    }
    if discriminator is not None:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    if discriminatorg is not None:
        checkpoint['discriminatorg_state_dict'] = discriminatorg.state_dict()

    torch.save(checkpoint, PATH)


def eval_batch(encoder, generator, data, test_loader, writer_val, file_sample, device, log_idx):

    encoder.eval()
    generator.eval()

    with torch.no_grad():
        acc_local_loss = 0
        acc_global_loss = 0
        acc_seq = []
        acc_real = []
        acc_out = []
        for input_sentence, lengths, sim_seq, _, _ in test_loader:
            input_sentence = input_sentence.to(device)
            lengths = lengths.to(device)
            sim_seq = sim_seq.to(device)

            encoded_input = encoder(utils.one_hot(input_sentence, data.getVocabSize()))

            seq_logprob = generator(encoded_input, teacher_forcing=False)
            seq_prob = torch.exp(seq_logprob)
            seq = net_utils.prob2pred(seq_logprob)
            # local loss criterion
            loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

            # compute local loss
            local_loss = loss(seq_logprob.permute(0, 2, 1), sim_seq)

            # get encoding from
            encoded_output = encoder(seq_prob)
            encoded_sim = encoder(utils.one_hot(sim_seq, data.getVocabSize()))
            # compute global loss
            global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)

            acc_local_loss += local_loss.item()
            acc_global_loss += global_loss.item()

            seq = seq.long()
            acc_seq += net_utils.decode_sequence(data.ix_to_word, seq)
            acc_real += net_utils.decode_sequence(data.ix_to_word, input_sentence)
            acc_out += net_utils.decode_sequence(data.ix_to_word, sim_seq)

        coco, cocoRes = getObjsForScores(acc_out, acc_seq)

        evalObj = COCOEvalCap(coco, cocoRes)

        evalObj.evaluate()

        for key in evalObj.eval:
            writer_val.add_scalar(key, evalObj.eval[key], log_idx)

        writer_val.add_scalar('local_loss', acc_local_loss/len(test_loader), log_idx)
        writer_val.add_scalar('global_loss', acc_global_loss/len(test_loader), log_idx)
        writer_val.add_scalar('total_loss', (acc_local_loss + acc_global_loss)/len(test_loader), log_idx)

        f_sample = open(file_sample + str(log_idx) + '.txt', 'w')

        idx = 1
        for r, s, t in zip(acc_real, acc_out, acc_seq):

            f_sample.write(str(idx) + '\nreal : ' + r + '\nout : ' + s + '\npred : ' + t + '\n\n')
            idx += 1

        f_sample.close()
        torch.cuda.empty_cache()


def train_epoch_EDLPS(encoder, generator, model_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()

        # zero all gradiants
        model_optim.zero_grad()

        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

        # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(logprobs.permute(0, 2, 1), sim_seq)

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = encoder(probs)
        encoded_sim = encoder(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)

        # take losses togather
        total_loss = local_loss + global_loss

        # backward propagation
        total_loss.backward()

        # update the parameters
        model_optim.step()

        # calculating losses
        epoch_local_loss += local_loss.item()
        epoch_global_loss += global_loss.item()

        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDL(encoder, generator, model_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()

        # zero all gradiants
        model_optim.zero_grad()

        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)

        # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(logprobs.permute(0, 2, 1), sim_seq)

        # backward propagation
        local_loss.backward()

        # update the parameters
        model_optim.step()

        # calculating losses
        epoch_local_loss += local_loss.item()
        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDP(encoder, generator, discriminator, model_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()
        discriminator.train()
        # zero all gradiants
        model_optim.zero_grad()

        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = discriminator(probs)
        encoded_sim = discriminator(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)

        # backward propagation
        global_loss.backward()

        # update the parameters
        model_optim.step()
        epoch_global_loss += global_loss.item()
        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDLPG(encoder, generator, discriminator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()
        discriminator.train()
        discriminatorg.train()
        # zero all gradiants
        model_optim.zero_grad()
        d_optim.zero_grad()

        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)
        d_loss = nn.BCELoss()
        # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(logprobs.permute(0, 2, 1), sim_seq)

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = discriminator(probs)
        encoded_sim = discriminator(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)
        d_fake = discriminatorg(encoded_output).view(-1)
        d_real = discriminatorg(encoded_sim).view(-1)
        # take losses togather
        total_loss = local_loss + global_loss
        d_loss_fake = d_loss(d_fake, torch.full(d_fake.size(), 1, device=d_fake.device))
        # backward propagation
        (total_loss + d_loss_fake).backward(retain_graph=True)
        # update the parameters
        model_optim.step()

        dd_loss = d_loss(d_fake, torch.full(d_fake.size(), 0, device=d_fake.device)) + d_loss(d_real, torch.full(d_real.size(), 1, device=d_real.device))
        # calculating losses
        d_optim.zero_grad()
        dd_loss.backward()
        d_optim.step()
        epoch_local_loss += local_loss.item()
        epoch_global_loss += global_loss.item()
        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDLPGS(encoder, generator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()
        discriminatorg.train()
        # zero all gradiants
        model_optim.zero_grad()
        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)
        d_loss = nn.BCELoss()
        # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(logprobs.permute(0, 2, 1), sim_seq)

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = encoder(probs)
        encoded_sim = encoder(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)
        d_fake = discriminatorg(encoded_output).view(-1)
        d_real = discriminatorg(encoded_sim).view(-1)
        # take losses togather
        total_loss = global_loss + local_loss
        d_loss_fake = d_loss(d_fake, torch.full(d_fake.size(), 1, device=d_fake.device))

        # backward propagation
        (d_loss_fake + total_loss).backward(retain_graph=True)

        # update the parameters
        model_optim.step()
        d_optim.zero_grad()

        dd_loss = d_loss(d_fake, torch.full(d_fake.size(), 0, device=d_fake.device)) + d_loss(d_real, torch.full(d_real.size(), 1, device=d_real.device))
        dd_loss.backward()
        d_optim.step()

        # calculating losses
        epoch_local_loss += local_loss.item()
        epoch_global_loss += global_loss.item()

        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDLP(encoder, generator, discriminator, model_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()
        discriminator.train()
        # zero all gradiants
        model_optim.zero_grad()

        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        # local loss criterion
        loss = nn.CrossEntropyLoss(ignore_index=data.PAD_token)
        # compute local loss
        '''loss : ([N, C, d1], [N, d1])'''
        local_loss = loss(logprobs.permute(0, 2, 1), sim_seq)

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = discriminator(probs)
        encoded_sim = discriminator(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)
        # take losses togather
        total_loss = local_loss + global_loss
        # backward propagation
        (total_loss).backward()
        # update the parameters
        model_optim.step()
        # calculating losses
        epoch_local_loss += local_loss.item()
        epoch_global_loss += global_loss.item()
        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDG(encoder, generator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()
        discriminatorg.train()
        # zero all gradiants
        model_optim.zero_grad()
        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        d_loss = nn.BCELoss()

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = encoder(probs)
        encoded_sim = encoder(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        d_fake = discriminatorg(encoded_output).view(-1)
        d_real = discriminatorg(encoded_sim).view(-1)
        d_loss_fake = d_loss(d_fake, torch.full(d_fake.size(), 1, device=d_fake.device))

        # backward propagation
        (d_loss_fake).backward(retain_graph=True)

        # update the parameters
        model_optim.step()
        d_optim.zero_grad()

        dd_loss = d_loss(d_fake, torch.full(d_fake.size(), 0, device=d_fake.device)) + d_loss(d_real, torch.full(d_real.size(), 1, device=d_real.device))
        dd_loss.backward()
        d_optim.step()

        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx


def train_epoch_EDPG(encoder, generator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=0):

    epoch_local_loss = 0
    epoch_global_loss = 0
    idx = 0
    for batch in train_loader:

        encoder.train()
        generator.train()
        discriminatorg.train()
        # zero all gradiants
        model_optim.zero_grad()
        # get new batch
        input_sentence, lengths, sim_seq, sim_seq_len, _ = batch

        '''
        input_sentence : [N, seq_len]
        length : [N]
        '''
        input_sentence = input_sentence.to(device)
        lengths = lengths.to(device)
        sim_seq = sim_seq.to(device)
        sim_seq_len = sim_seq_len.to(device)
        # forward propagation
        vocab_size = data.getVocabSize()
        encoded = encoder(utils.one_hot(input_sentence, vocab_size))
        logprobs = generator(encoded, true_out=sim_seq)
        probs = torch.exp(logprobs)
        '''
        probs: (batch_size, seq_len , vocab_size )
        encoded_input : (batch_size, emb_size)
        '''
        d_loss = nn.BCELoss()

        # get encoding from
        '''([N, d1, C])'''
        encoded_output = encoder(probs)
        encoded_sim = encoder(utils.one_hot(sim_seq, vocab_size))
        # compute global loss
        global_loss = net_utils.JointEmbeddingLoss(encoded_output, encoded_sim)
        d_fake = discriminatorg(encoded_output).view(-1)
        d_real = discriminatorg(encoded_sim).view(-1)
        # take losses togather
        d_loss_fake = d_loss(d_fake, torch.full(d_fake.size(), 1, device=d_fake.device))

        # backward propagation
        (d_loss_fake + global_loss).backward(retain_graph=True)

        # update the parameters
        model_optim.step()
        d_optim.zero_grad()

        dd_loss = d_loss(d_fake, torch.full(d_fake.size(), 0, device=d_fake.device)) + d_loss(d_real, torch.full(d_real.size(), 1, device=d_real.device))
        dd_loss.backward()
        d_optim.step()

        # calculating losses
        epoch_global_loss += global_loss.item()

        torch.cuda.empty_cache()
        idx += 1

    return epoch_local_loss, epoch_global_loss, log_idx
