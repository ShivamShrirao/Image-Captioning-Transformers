import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torch.cuda import amp

import timm         # torch image models
import wandb

from imcap.dataset import *
from imcap.dataloader import *
from imcap.layers import *
from imcap.utils import *

import argparse

parser = argparse.ArgumentParser(description='Image Captioning Training')
parser.add_argument('--data-dir',          dest='data-dir',          default="../datasets/COCO/",   metavar='', help='path to dataset')
parser.add_argument('--encoder',           dest='encoder',           default='seresnext50_32x4d',   metavar='', help='encoder architecture')
parser.add_argument('--epochs',            dest='NUM_EPOCHS',        default=50,         type=int,  metavar='', help='number of total epochs to run')
parser.add_argument('--batch_size',        dest='BATCH_SIZE',        default=256,        type=int,  metavar='', help='mini-batch size (default: 256)')
parser.add_argument('--d_model',           dest='d_model',           default=512,        type=int,  metavar='', help='embedding dimension')
parser.add_argument('--dim_feedforward',   dest='dim_feedforward',   default=2048,       type=int,  metavar='', help='dim feed forward')
parser.add_argument('--nheads',            dest='nheads',            default=8,          type=int,  metavar='', help='number of heads')
parser.add_argument('--num_decoder_layers',dest='num_decoder_layers',default=6,          type=int,  metavar='', help='number of decoder layers')
parser.add_argument('--dp_rate',           dest='dp_rate',           default=0.1,        type=float,metavar='', help='dropout rate')
parser.add_argument('--activation',        dest='activation',        default='gelu',     type=str,  metavar='', help='activation function')
parser.add_argument('--betas',             dest='betas',             default=[0.9, 0.98],type=float,metavar='', help='betas', nargs='+')
parser.add_argument('--eps',               dest='eps',               default=1e-9,       type=float,metavar='', help='epsilon')
parser.add_argument('--seed',              dest='seed',              default=62134,      type=int,  metavar='', help='seed')
parser.add_argument('--use_amp',           dest='use_amp',           default=True,       type=bool, metavar='', help='use mixed precision')
parser.add_argument('--use_pe',            dest='use_pe',            default=True,       type=bool, metavar='', help='use positional encoding')
parser.add_argument('--max_lr',            dest='max_lr',            default=3e-4,       type=float,metavar='', help='initial learning rate')
parser.add_argument('--log_interval',      dest='log_interval',      default=10,         type=int,  metavar='', help='Log interval (default: 10)')

args = parser.parse_args()
args = vars(args)
print(args)

DATA_DIR = args.pop('data-dir')
CONFIG = args

if __name__ == '__main__':
    run = wandb.init(project="Image_Captioning_Transformer", config=args)
    CONFIG = wandb.config

def seed_everything(seed=33):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    
seed_everything(CONFIG['seed'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Read COCO dataset"""

train_data = TensorCocoCaptions(root=DATA_DIR+"/train2017/",
                                annFile=DATA_DIR+"/annotations/captions_train2017.json")

val_data = TensorCocoCaptions(root=DATA_DIR+"/val2017/",
                              annFile=DATA_DIR+"/annotations/captions_val2017.json")

"""## Tokenizer and Build Vocab"""


tokenizer = get_tokenizer('basic_english')

def yield_tokens(cap_data):
    for ann in cap_data.coco.anns.values():
        yield tokenizer(ann['caption'])

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
en_vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=special_symbols, special_first=True)

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = en_vocab(special_symbols)
en_vocab.set_default_index(UNK_IDX)

train_data.fill_token_dict(tokenizer, en_vocab, BOS_IDX, EOS_IDX)
val_data.fill_token_dict(tokenizer, en_vocab, BOS_IDX, EOS_IDX)

"""# Load dataset into batches"""

nthreads = 2 * len(os.sched_getaffinity(0))

train_iter = ExternalInputIterator(train_data, CONFIG['BATCH_SIZE'], PAD_IDX)
pipe = ExternalSourcePipeline(batch_size=CONFIG['BATCH_SIZE'], num_threads=nthreads, device_id=0, external_data=train_iter, input_size=input_size)
train_loader = DALIClassificationIterator(pipe, dynamic_shape=True, auto_reset=True, last_batch_padded=True, size=len(train_iter))

val_iter = ExternalInputIterator(val_data, CONFIG['BATCH_SIZE'], PAD_IDX, training=False)
pipe = ExternalSourcePipeline(batch_size=CONFIG['BATCH_SIZE'], num_threads=nthreads, device_id=0, external_data=val_iter, input_size=input_size, training=False)
val_loader = DALIClassificationIterator(pipe, dynamic_shape=True, auto_reset=True, last_batch_padded=True, size=len(val_iter))

"""# Initialize Model"""

model = CaptionModel(encoder = timm.create_model(CONFIG['encoder'], pretrained=True, num_classes=0, global_pool=''),
                     vocab_size = len(en_vocab),
                     num_decoder_layers = CONFIG['num_decoder_layers'],
                     nheads = CONFIG['nheads'],
                     d_model = CONFIG['d_model'],
                     dim_feedforward = CONFIG['dim_feedforward'],
                     dp_rate = CONFIG['dp_rate'],
                     activation = CONFIG['activation']).to(DEVICE, non_blocking=True)

"""# Learning Rate Schedule"""

steps_per_epoch = len(train_loader)

# def lr_schedule(step, d_model=512, warmup_steps=2*steps_per_epoch):
#     return 1
    # step = max(1,step)
    # arg1 = step ** -0.5
    # arg2 = step * (warmup_steps ** -1.5)
    # return (d_model ** -0.6) * min(arg1, arg2)

"""# Loss Function and Optimizer"""

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['max_lr'],
    betas=CONFIG['betas'], eps=CONFIG['eps']
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['max_lr'], total_steps=50*steps_per_epoch, pct_start=0.0)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

scaler = amp.GradScaler(enabled=CONFIG['use_amp'])

if __name__ == '__main__':
    wandb.watch(model, log=None)

"""# Training functions"""

def train_epoch(model, train_loader, optimizer, scaler, scheduler, epoch=1, use_amp=True, log_interval=10):
    model.train()
    model.encoder.eval()
    losses = AverageMeter()
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
        for idx, batch in pbar:
            img, tgt = batch[0]['data'], batch[0]['label'].transpose(0,1)
            # img = img.to(DEVICE, non_blocking=True)
            # tgt = tgt.to(DEVICE, non_blocking=True)
            
            tgt_inp = tgt[:-1,:]      # give input until before the last word.
            tgt_out = tgt[1:, :]      # predict the last word based on input and already predicted sentence. (auto-regressive)

            tgt_mask, tgt_pad_mask = subsequent_mask(tgt_inp.size(0), DEVICE), padding_mask(tgt_inp, PAD_IDX)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=use_amp):
                logits = model(img, tgt_inp, tgt_mask, tgt_pad_mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.update(loss.detach_(), img.size(0))
            # del loss, logits, batch, img

            if not idx%log_interval:
                curr_lr = optimizer.param_groups[0]['lr']
                info = {'loss': float(losses.avg), 'lr': curr_lr}
                wandb.log(info)
                pbar.set_postfix(info)

    optimizer.zero_grad(set_to_none=True)
    return float(losses.avg)

@torch.no_grad()
def evaluate(model, val_loader, use_amp=True):
    model.eval()
    losses = AverageMeter()
    with tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating") as pbar:
        for idx, batch in pbar:
            img, tgt = batch[0]['data'], batch[0]['label'].transpose(0,1)
            # img = img.to(DEVICE, non_blocking=True)
            # tgt = tgt.to(DEVICE, non_blocking=True)

            tgt_inp = tgt[:-1,:]      # give input until before the last word.
            tgt_out = tgt[1:, :]      # predict the last word based on input and already predicted sentence. (auto-regressive)

            tgt_mask, tgt_pad_mask = subsequent_mask(tgt_inp.size(0), DEVICE), padding_mask(tgt_inp, PAD_IDX)
            
            with amp.autocast(enabled=use_amp):
                logits = model(img, tgt_inp, tgt_mask, tgt_pad_mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            losses.update(loss.detach_(), img.size(0))
            pbar.set_postfix({'val_loss': float(losses.avg)})
    return float(losses.avg)

"""# Functions to Make Predictions"""

@torch.no_grad()
def greedy_decode(model, img, max_len=100, start_symbol=BOS_IDX):
    model.eval()
    img = img.to(DEVICE, non_blocking=True)
    enc_output = model.encode_image(img)
    tgt = torch.ones(1, 1).fill_(start_symbol).long().to(DEVICE, non_blocking=True)
    for i in range(max_len):
        tgt_mask = subsequent_mask(tgt.size(0), DEVICE)
        out = model.decode_text(tgt, enc_output, tgt_mask)
        out = out.transpose(0,1)
        prob = model.generator(out[:,-1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        tgt = torch.cat([tgt, torch.ones(1, 1).fill_(next_word).long().to(DEVICE)], dim=0)
        if next_word == EOS_IDX:
            break
    return tgt.detach()

@torch.no_grad()
def generate_caption(model, img, tgt_vocab):
    tgt = greedy_decode(model, img, max_len=100, start_symbol=BOS_IDX).flatten()
    return " ".join(tgt_vocab.lookup_tokens(tgt.tolist())).replace("<bos>", "").replace("<eos>", "")

"""# Begin Training"""

NUM_EPOCHS = CONFIG["NUM_EPOCHS"]

import gc
gc.collect()
torch.cuda.empty_cache()

import glob
val_paths = glob.glob(DATA_DIR+"/val2017/*")

def main():
    init_epoch = 1
    #collapse-output
    for epoch in range(init_epoch, NUM_EPOCHS+1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, scheduler,
                                 epoch, CONFIG['use_amp'], CONFIG['log_interval'])
        # with torch.no_grad():
        val_loss = evaluate(model, val_loader, CONFIG['use_amp'])

        img = Image.open(random.choice(val_paths))
        caps = generate_caption(model, preproc['val'](img)[None,:], en_vocab)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, "predictions": wandb.Image(img, caption=caps)})
        print(f"\nEpoch: {epoch}/{NUM_EPOCHS}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}\n")
        gc.collect()
        # if not epoch%10:
        #     save_model(model, optimizer, epoch)

    return epoch

def save_model(model, optimizer, scheduler, epoch=0, path='/content/model.pth'):
    torch.save({
                'projection_head': model.projection_head.state_dict(),
                'decoder': model.decoder.state_dict(),
                'generator': model.generator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                }, path)

def load_model(model, optimizer, scheduler, path='/content/model.pth'):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.projection_head.load_state_dict(checkpoint['projection_head'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.generator.load_state_dict(checkpoint['generator'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

"""# Make Predictions"""

if __name__ == '__main__':
    main()
    run.finish()