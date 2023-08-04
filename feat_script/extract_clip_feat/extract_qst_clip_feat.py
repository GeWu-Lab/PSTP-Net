import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob
import json
import ast
import csv

import clip_net.clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_net.clip.load("ViT-B/32", device=device)


def qst_feat_extract(qst):

    text = clip_net.clip.tokenize(qst).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    return text_features


def QstCLIP_feat(json_path, dst_qst_path):

    samples = json.load(open(json_path, 'r'))
    
    ques_vocab = ['<pad>']

    i = 0
    for sample in samples:
        i += 1
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        question_id = sample['question_id']
        print("\n")
        print("question id: ", question_id)

        save_file = os.path.join(dst_qst_path, str(question_id) + '.npy')

        if os.path.exists(save_file):
            print(question_id, " is already exist!")
            continue

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        question = ' '.join(question)
        print(question)
        
        qst_feat = qst_feat_extract(question)
        print(qst_feat.shape)

        qst_features = qst_feat.float().cpu().numpy()

        np.save(save_file, qst_features)



if __name__ == "__main__":

    json_path = "../../dataset/split_que_id/music_avqa.json"
    
    dst_qst_path = "/home/data/MUSIC-AVQA/clip_word/"

    QstCLIP_feat(json_path, dst_qst_path)


    