import torch
import torch.optim as optim
from misc.LanguageModel import layer as LanguageModel
import misc.utils as utils
from misc.HybridCNNLong import HybridCNNLong as DocumentCNN
from tensorboardX import SummaryWriter
import subprocess
import torch.utils.data as Data
from misc.dataloader import Dataloader
import time
import os
from discriminator import Discriminator
from train_util import eval_batch, save_model, train_epoch_EDL,\
                train_epoch_EDP, train_epoch_EDLP, train_epoch_EDLPS,\
                train_epoch_EDLPG, train_epoch_EDLPGS, train_epoch_EDG,\
                train_epoch_EDPG

# get command line arguments into args
parser = utils.make_parser()
args = parser.parse_args()

torch.manual_seed(args.seed)


log_folder = 'logs'
save_folder = 'save'
folder = time.strftime("%d-%m-%Y_%H:%M:%S")
folder = args.name + folder

if args.start_from != 'None':
    folder = args.start_from.split('/')[-2]

if args.start_from != 'None':
    writer_train = SummaryWriter(os.path.join(log_folder, folder + 'train' + args.name))
    writer_val = SummaryWriter(os.path.join(log_folder, folder + 'val'+ args.name))
else:
    writer_train = SummaryWriter(os.path.join(log_folder, folder + 'train'))
    writer_val = SummaryWriter(os.path.join(log_folder, folder + 'val'))

subprocess.run(['mkdir', os.path.join(save_folder, folder)])
subprocess.run(['mkdir', os.path.join('samples', folder)])

file_sample = os.path.join('samples', folder, 'samples')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get dataloader
data = Dataloader(args.input_json, args.input_ques_h5)

train_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len)), batch_size=args.batch_size, shuffle=True)
test_loader = Data.DataLoader(Data.Subset(data, range(args.train_dataset_len, args.train_dataset_len + args.val_dataset_len)), batch_size=args.batch_size, shuffle=True)

encoder = DocumentCNN(data.getVocabSize(), args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)
if args.model == 'EDL' or args.model == 'EDLP' or args.model == 'EDLPG':
    discriminator = DocumentCNN(data.getVocabSize(), args.txtSize, dropout=args.drop_prob_lm, avg=1, cnn_dim=args.cnn_dim)
else:
    discriminator = None
if 'G' in args.model:
    discriminatorg = Discriminator(args.txtSize)
else:
    discriminatorg = None
generator = LanguageModel(args.input_encoding_size, args.rnn_size, data.getSeqLength(), data.getVocabSize(), num_layers=args.rnn_layers, dropout=args.drop_prob_lm)

print('Start the training...')

if discriminator is None:
    model_optim = optim.RMSprop(list(encoder.parameters()) + list(generator.parameters()), lr=args.learning_rate)
else:
    model_optim = optim.RMSprop(list(encoder.parameters()) + list(generator.parameters()) + list(discriminator.parameters()), lr=args.learning_rate)

if discriminatorg is not None:
    d_optim = optim.RMSprop(discriminatorg.parameters(), lr=args.learning_rate)

if args.start_from != 'None':
    print('loading model from ' + args.start_from)
    checkpoint = torch.load(args.start_from, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    start_epoch = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

encoder = encoder.to(device)
generator = generator.to(device)
n_epoch = args.n_epoch
log_idx = 0

for epoch in range(start_epoch, start_epoch + n_epoch):

    if args.model == 'EDL':
        local_loss, global_loss, log_idx = train_epoch_EDL(encoder, generator, model_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDP':
        local_loss, global_loss, log_idx = train_epoch_EDP(encoder, generator, discriminator, model_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDLP':
        local_loss, global_loss, log_idx = train_epoch_EDLP(encoder, generator, discriminator, model_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDLPS':
        local_loss, global_loss, log_idx = train_epoch_EDLPS(encoder, generator, model_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDLPG':
        local_loss, global_loss, log_idx = train_epoch_EDLPG(encoder, generator, discriminator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDLPGS':
        local_loss, global_loss, log_idx = train_epoch_EDLPGS(encoder, generator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDG':
        local_loss, global_loss, log_idx = train_epoch_EDG(encoder, generator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=log_idx)
    elif args.model == 'EDPG':
        local_loss, global_loss, log_idx = train_epoch_EDPG(encoder, generator, discriminatorg, model_optim, d_optim, train_loader, data, device, log_idx=log_idx)

    n_batch = args.train_dataset_len // args.batch_size

    local_loss = local_loss / n_batch
    global_loss = global_loss / n_batch

    writer_train.add_scalar('local_loss', local_loss, log_idx)
    writer_train.add_scalar('global_loss', global_loss, log_idx)
    writer_train.add_scalar('total_loss', local_loss + global_loss, log_idx)

    eval_batch(encoder, generator, data, test_loader, writer_val, file_sample, device, log_idx)
    log_idx += 1
    save_model(encoder, generator, model_optim, epoch + 1, -1, local_loss, global_loss, save_folder, folder)

print('Done !!!')
