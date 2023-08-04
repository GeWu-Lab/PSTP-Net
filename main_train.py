import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import json
import numpy as np

from dataloader import *
from nets.PSTP_Net_Ours import PSTP_Net
from configs.arguments_PSTP_Net import parser


import warnings
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')


print("\n--------------- PSPT-Net Training --------------- \n")


def train(args, model, train_loader, optimizer, criterion, writer, epoch):
    
    model.train()

    for batch_idx, sample in enumerate(train_loader):
        audios_feat, visual_feat, patch_feat, target, question, qst_word = sample['audios_feat'].to('cuda'), sample['visual_feat'], sample['patch_feat'], sample['answer_label'].to('cuda'), sample['question'].to('cuda'), sample['qst_word'].to('cuda')

        optimizer.zero_grad()
        output_qa = model(audios_feat, visual_feat, patch_feat, question, qst_word)  
        loss = criterion(output_qa, target)

        writer.add_scalar('run/both', loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(audios_feat), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader, writer, epoch):
    
    model.eval()
    total_qa = 0
    correct_qa = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audios_feat, visual_feat, patch_feat, target, question, qst_word = sample['audios_feat'].to('cuda'), sample['visual_feat'], sample['patch_feat'], sample['answer_label'].to('cuda'), sample['question'].to('cuda'), sample['qst_word'].to('cuda')

            preds_qa = model(audios_feat, visual_feat, patch_feat, question, qst_word)

            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Current Acc: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa



def main():

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    tensorboard_name = args.checkpoint
    writer = SummaryWriter('runs/strn/' + TIMESTAMP + '_' + tensorboard_name)

    model = PSTP_Net(args)
    model = nn.DataParallel(model).to('cuda')



    # -------------> Computation costs
    # from thop import profile
    # from thop import clever_format

    # model = PSTP_Net(args)
    # model = model.to('cuda')

    # input1 = torch.randn(1, 60, 128).to('cuda')
    # input2 = torch.randn(1, 60, 512).to('cuda')
    # input3 = torch.randn(1, 60, 50, 512).to('cuda')
    # input4 = torch.randn(1, 1, 512).to('cuda')
    # input5 = torch.randn(1, 77, 512).to('cuda')

    # flops, params = profile(model, inputs=(input1, input2, input3, input4, input5))
    # print("profile: ", flops, params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print("clever: ", flops, params) 

    # -------------> Computation costs end


    train_dataset = AVQA_dataset(label = args.label_train, 
                                 args = args, 
                                 audios_feat_dir = args.audios_feat_dir, 
                                 visual_feat_dir = args.visual_feat_dir,
                                 clip_vit_b32_dir = args.clip_vit_b32_dir,
                                 clip_qst_dir = args.clip_qst_dir, 
                                 clip_word_dir = args.clip_word_dir, 
                                 transform = transforms.Compose([ToTensor()]), 
                                 mode_flag = 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_dataset = AVQA_dataset(label = args.label_val, 
                               args = args, 
                               audios_feat_dir = args.audios_feat_dir, 
                               visual_feat_dir = args.visual_feat_dir,
                               clip_vit_b32_dir = args.clip_vit_b32_dir,
                               clip_qst_dir = args.clip_qst_dir,  
                               clip_word_dir = args.clip_word_dir,  
                               transform = transforms.Compose([ToTensor()]), 
                               mode_flag = 'val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    

    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        train(args, model, train_loader, optimizer, criterion, writer, epoch=epoch)

        # evaluate on validation set
        scheduler.step(epoch)
        current_acc = eval(model, val_loader, writer, epoch)
        if current_acc >= best_acc:
            best_acc = current_acc
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")

        print("Best Acc: %.2f %%"%best_acc)
        print("Best Epoch: ", best_epoch)
        print("*"*20)


if __name__ == '__main__':
    main()