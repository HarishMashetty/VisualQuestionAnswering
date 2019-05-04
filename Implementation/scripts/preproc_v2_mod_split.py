from __future__ import division, print_function, absolute_import


import sys
import csv
import json
import os
import re
import pdb

csv.field_size_limit(5000000)
    

import base64
import pickle

import numpy as np
import nltk
nltk.data.path.append('data')
nltk.download('punkt', download_dir='data')
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.externals import joblib

train_anno_path = os.path.join('data', 'v2_mscoco_train2014_annotations.json')
val_anno_path = os.path.join('data', 'v2_mscoco_val2014_annotations.json')
train_ques_path = os.path.join('data', 'v2_OpenEnded_mscoco_train2014_questions.json')
val_ques_path = os.path.join('data', 'v2_OpenEnded_mscoco_val2014_questions.json')
glove_emb_path = os.path.join('data', 'glove', 'glove.6B.300d.txt')
visual_feats_path = os.path.join('data', 'trainval_resnet101_faster_rcnn_genome_36.tsv')

word_contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_mapping = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

articles_list = ['a', 'an', 'the']
strip_period = re.compile("(?!<=\d)(\.)(?!\d)")
strip_comma = re.compile("(\d)(\,)(\d)")
punct = [
    ';', r"/", '[', ']', '"', '{', '}',
    '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!'
]


def punctuation_processing(in_Text):
    out_Text = in_Text
    for p in punct:
        if (p + ' ' in in_Text or ' ' + p in in_Text) \
        or (re.search(strip_comma, in_Text) != None):
            out_Text = out_Text.replace(p, '')
        else:
            out_Text = out_Text.replace(p, ' ')
    out_Text = strip_period.sub("", out_Text, re.UNICODE)
    return out_Text

def digit_article_processing(inp_text):
    out_text = []
    temp_text = inp_text.lower().split()
    for word in temp_text:
        word = manual_mapping.setdefault(word, word)
        if word not in articles_list:
            out_text.append(word)
        else:
            pass
    for wordId, word in enumerate(out_text):
        if word in word_contractions:
            out_text[wordId] = word_contractions[word]
    out_text = ' '.join(out_text)
    return out_text


def anno_processing(freq_thr=9):
    ta_0 = json.load(open(train_anno_path))['annotations']
    va_0 = json.load(open(val_anno_path))['annotations']

	
    
    
    annos_0 = ta_0 + va_0
    annos=[]
	

    print("Calculating the freq of each multiple choice answers:")

    for anno in tqdm(annos_0):
        if anno['image_id']%7==3:
            annos.append(anno)
	
	
		
		
        
            
    mca_frequency = {}
    for anno in tqdm(annos):
        mca = digit_article_processing(punctuation_processing(anno['multiple_choice_answer']))
        mca = mca.replace(',', '')
        mca_frequency[mca] = mca_frequency.get(mca, 0) + 1

    # filter rare ans out
    for a, freq in list(mca_frequency.items()):
        if freq < freq_thr:
            mca_frequency.pop(a)

    print("Number of answers appear more than {} times: {}".format(freq_thr - 1, len(mca_frequency)))

    # generating answer dict
    idx_to_ans = []
    ans_to_idx = {}
    for i, a in enumerate(mca_frequency):
        idx_to_ans.append(a)
        ans_to_idx[a] = i

    print("Generating the soft scores:")
    trgt = []
    for anno in tqdm(annos):
        anss = anno['answers']

        # calculating individual answers freq
        ans_frequency = {}
        for a in anss:
            ans_frequency[a['answer']] = ans_frequency.get(a['answer'], 0) + 1

        soft_scores = []
        for a, freq in ans_frequency.items():
            if a in ans_to_idx:
                soft_scores.append((a, min(1, freq / 3)))

        trgt.append({
            'question_id': anno['question_id'],
            'image_id': anno['image_id'],
            'answer': soft_scores    # [(ans1, score1), (ans2, score2), (ansn, scoren ...]
        })

    pickle.dump([idx_to_ans, ans_to_idx], open(os.path.join('data', 'dict_ans.pkl'), 'wb'))
    return trgt


def tokenize_str(t):
    t = t.lower().replace(',', '').replace('?', '')
    return word_tokenize(t)


def qa_processing(trgt, max_words=14):

    print("Merging que and ans:")
    idx_to_word = []
    word_to_idx = {}

    tq=[]
    vq=[]
    tq_0 = json.load(open(train_ques_path))['questions']
    vq_0 = json.load(open(val_ques_path))['questions']

    qs=[]
    for qs_i in tqdm(tq_0):
        if qs_i['image_id']%7==3:
            tq.append(qs_i)

    for qs_i in tqdm(vq_0):
        if qs_i['image_id']%7==3:
            vq.append(qs_i)


    qs = tq + vq
    que_ans_s = []
	
	
    

 
    for i, q in enumerate(tqdm(qs)):
        tokens = tokenize_str(q['question'])
        for t in tokens:
            if not t in word_to_idx:
                idx_to_word.append(t)
                word_to_idx[t] = len(idx_to_word) - 1

        assert q['question_id'] == trgt[i]['question_id'],\
                "Question ID doesn't match ({}: {})".format(q['question_id'], trgt[i]['question_id'])

        que_ans_s.append({
            'image_id': q['image_id'],
            'question': q['question'],
            'question_id': q['question_id'],
            'question_toked': tokens,
            'answer': trgt[i]['answer']
        })

    pickle.dump(que_ans_s[:len(tq)], open(os.path.join('data', 'train_qa.pkl'), 'wb'))
    pickle.dump(que_ans_s[len(tq):], open(os.path.join('data', 'val_qa.pkl'), 'wb'))
    pickle.dump([idx_to_word, word_to_idx], open(os.path.join('data', 'dict_q.pkl'), 'wb'))
	
	
   
    return idx_to_word


def word_emb_processing(idx_to_word):
    print("Generating pre-trained word embedding wts:")
    word_to_emb = {}
    dim_emb = int(glove_emb_path.split('.')[-2].split('d')[0])
    with open(glove_emb_path, encoding = 'utf-8') as f:
        for entry in f:
            values = entry.split(' ')
            word = values[0]
            word_to_emb[word] = np.asarray(values[1:], dtype=np.float32)

    pretrained_wts = np.zeros((len(idx_to_word), dim_emb), dtype=np.float32)
    for idx, word in enumerate(idx_to_word):
        if word not in word_to_emb:
            continue
        pretrained_wts[idx] = word_to_emb[word]

    np.save(os.path.join('data', 'glove_pretrained_{}.npy'.format(dim_emb)), pretrained_wts)


def visual_feats_processing():
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    tq = json.load(open(train_ques_path))['questions']
    vq = json.load(open(val_ques_path))['questions']
    t_ids = set([ q['image_id'] for q in tq if q['image_id']%7==3])
    v_ids = set([ q['image_id'] for q in vq if q['image_id']%7==3])
    #t_ids = set([q['image_id'] for q in train_ques])
    #v_ids = set([q['image_id'] for q in val_ques])
	
    


    print("Reading tsv, total number of iterations: {}".format(len(t_ids)+len(v_ids)))
    tv_features = {}
    vv_features = {}
    with open(visual_feats_path, encoding='utf-8') as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for i, item in enumerate(tqdm(reader)):
            image_id = int(item['image_id'])
            if image_id%7!=3:
                continue
			
				
			
            feats = np.frombuffer(base64.b64decode(item['features']), 
                dtype=np.float32).reshape((int(item['num_boxes']), -1))

            if image_id in t_ids:
                tv_features[image_id] = feats
            elif image_id in v_ids:
                vv_features[image_id] = feats
            else:
                raise ValueError("Image_id: {} not in training or validation set".format(image_id))

    print("Converting tsv to pickle...hang on")
    joblib.dump(tv_features, open(os.path.join('data', 'train_vfeats.pkl'), 'wb'))
    joblib.dump(vv_features, open(os.path.join('data', 'val_vfeats.pkl'), 'wb'))


if __name__ == '__main__':
    trgt = anno_processing()
    idx_to_word = qa_processing(trgt)
    word_emb_processing(idx_to_word)
    visual_feats_processing()
    print("Ok Done")
