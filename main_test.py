import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import ast
import json
import numpy as np

from dataloader import *
from nets.PSTP_Net_Ours import PSTP_Net
from configs.arguments_PSTP_Net import parser


print("\n--------------- Spatio-temporal Reasoning Network(PSTP-Net) --------------- \n")

def test(model, val_loader):
    
    model.eval()
    
    total = 0
    correct = 0
    samples = json.load(open('./dataset/split_que_id/music_avqa_test.json', 'r'))
    
    # prediction save
    A_count = []
    A_compt = []
    V_count = []
    V_local = []
    AV_exist = []
    AV_count = []
    AV_local = []
    AV_compt = []
    AV_templ = []

    # results save
    que_id = []
    pred_results =[]
    grd_target = []
    pred_label = []


    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):

            audios_feat, visual_feat, patch_feat, target, question, question_id, qst_word = sample['audios_feat'].to('cuda'), sample['visual_feat'], sample['patch_feat'], sample['answer_label'].to('cuda'), sample['question'].to('cuda'), sample['question_id'], sample['qst_word'].to('cuda')

            preds_qa = model(audios_feat, visual_feat, patch_feat, question, qst_word)  

            preds = preds_qa

            _, predicted = torch.max(preds.data, 1)
            # print(preds.data, predicted, target)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            # result
            grd_target.append(target.cpu().item())
            pred_label.append(predicted.cpu().item())

            pred_bool = predicted == target
            for index in range(len(pred_bool)):
                pred_results.append(pred_bool[index].cpu().item())
                que_id.append(question_id[index].item())


            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_compt.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_local.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_exist.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_local.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_compt.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_templ.append((predicted == target).sum().item())

    print('\nAudio Count Acc: %.2f %%' % (100 * sum(A_count)/len(A_count)))
    print('Audio Compt Acc: %.2f %%' % (100 * sum(A_compt) / len(A_compt)))
    print('Audio Averg Acc: %.2f %%' % (100 * (sum(A_count) + sum(A_compt)) / (len(A_count) + len(A_compt))))
    
    print('\nVisual Count Acc: %.2f %%' % (100 * sum(V_count) / len(V_count)))
    print('Visual Local Acc: %.2f %%' % (100 * sum(V_local) / len(V_local)))
    print('Visual Averg Acc: %.2f %%' % (100 * (sum(V_count) + sum(V_local)) / (len(V_count) + len(V_local))))
    
    print('\nAudio-Visual Exist Acc: %.2f %%' % (100 * sum(AV_exist) / len(AV_exist)))
    print('Audio-Visual Count Acc: %.2f %%' % (100 * sum(AV_count) / len(AV_count)))
    print('Audio-Visual Local Acc: %.2f %%' % (100 * sum(AV_local) / len(AV_local)))
    print('Audio-Visual Compt Acc: %.2f %%' % (100 * sum(AV_compt) / len(AV_compt)))
    print('Audio-Visual Templ Acc: %.2f %%' % (100 * sum(AV_templ) / len(AV_templ)))
    print('Audio-Visual Averg Acc: %.2f %%' % (100 * (sum(AV_count) + sum(AV_local) + sum(AV_exist) + sum(AV_templ) + sum(AV_compt)) / 
                                                     (len(AV_count) + len(AV_local) + len(AV_exist) + len(AV_templ) + len(AV_compt))))
    
    print('\n---->Overall Accuracy: %.2f %%' % (100 * correct / total), "\n")

    with open("results/PSTP_Net.txt", 'w') as f:
        # print("len q: ", len(que_id))
        # print("len pred: ", len(pred_results))
        for index in range(len(que_id)):
            # print(que_id[index],' \t ',pred_results[index],' \t ',grd_target[index],' \t ',pred_label[index])
            f.write(str(que_id[index])+' \t '+str(pred_results[index])+' \t '+str(grd_target[index])+' \t '+str(pred_label[index])+'\n')

    return 100 * correct / total



def main():

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)

    model = PSTP_Net(args)
    model = nn.DataParallel(model)
    model = model.to('cuda')

    test_dataset = AVQA_dataset(args = args, 
                                label = args.label_test,
                                audios_feat_dir = args.audios_feat_dir,
                                visual_feat_dir = args.visual_feat_dir,
                                clip_vit_b32_dir = args.clip_vit_b32_dir,
                                clip_qst_dir = args.clip_qst_dir,
                                clip_word_dir = args.clip_word_dir, 
                                transform = transforms.Compose([ToTensor()]), 
                                mode_flag ='test')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
    test(model, test_loader)


if __name__ == '__main__':
    main()