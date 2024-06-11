import argparse
from utils.QAutils import load_data, quick_tokenize, evaluation, define_logger
import random
import numpy as np
import torch
from model.discriminate_model import pretrained_model
from transformers import AdamW
import sys
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import datetime
from sklearn.manifold import TSNE
import logging
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import networkx as nx
import nltk
from nltk.corpus import wordnet

# 设置 NLTK 数据路径
nltk.data.path.append('/home/amax/nltk_data')

# # 查看 NLTK 数据路径，确认已经包含了我们上传的路径
# print(nltk.data.path)


def main():
    parser = argparse.ArgumentParser(description='xCAR')

    # Data Paths
    parser.add_argument('--data_dir', type=str, default='./data/final_data/data/', help='The dataset directory')
    parser.add_argument('--model_dir', type=str, default='../../huggingface_transformers/xlnet-base-cased/',
                        help='The pretrained model directory')
    parser.add_argument('--save_dir', type=str, default='./output/saved_model', help='The model saving directory')
    parser.add_argument('--log_dir', type=str, default='./output/log', help='The training log directory')
    parser.add_argument('--apex_dir', type=str, default='./output/log', help='The apex directory')

    # Data names
    parser.add_argument('--train', type=str, default='train.pkl', help='The train data directory')
    parser.add_argument('--dev', type=str, default='dev.pkl', help='The dev data directory')
    parser.add_argument('--test', type=str, default='test.pkl', help='The test data directory')

    # Model Settings
    parser.add_argument('--model_name', type=str, default='xlnet', help='Pretrained model name')
    parser.add_argument('--data_name', type=str, default='copa')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu ids for training')
    # parser.add_argument('--apex', type=bool, default=False, help='Whether to use half precision')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=200, help='training iterations')
    parser.add_argument('--evaluation_step', type=int, default=20,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1024, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=10, help='the patient of early-stopping')
    parser.add_argument('--loss_func', type=str, default='BCE', help="loss function of output")
    parser.add_argument('--hyp_only', type=bool, default=False, help="If set True, Only send hypothesis into model")
    # parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup settings')

    # parsing the hyper-parameters from command line and define logger
    hps = parser.parse_args()
    logger, formatter = define_logger()
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if hps.hyp_only:
        log_path = os.path.join(hps.log_dir, 'discriminate_'+hps.model_name+'_hyp'+'_{}.txt'.format(nowtime))
    else:
        log_path = os.path.join(hps.log_dir, 'discriminate_'+hps.model_name+'_{}.txt'.format(nowtime))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # fix random seed
    if hps.set_seed:
        random.seed(hps.seed)
        np.random.seed(hps.seed)
        torch.manual_seed(hps.seed)
        torch.cuda.manual_seed(hps.seed)

    # load data
    # logger.info("[Pytorch] %s", torch.)
    logger.info("[MODEL] {}".format(hps.model_name))
    logger.info("[INFO] Loading Data")
    logger.info("[INFO] Hypothesis Only: {}".format(hps.hyp_only))
    train_data = load_data(os.path.join(hps.data_dir, hps.train))

    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    test_data = load_data(os.path.join(hps.data_dir, hps.test))

    def load_causal_graph(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        G = nx.DiGraph()

        for edge in graph_data['links']:
            G.add_edge(edge['source'], edge['target'])

        return G


    def get_synonyms(word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    # 定义一个函数来替换文本中的词语
    def replace_synonyms(text):
        tokens = nltk.word_tokenize(text)
        for i in range(len(tokens)):
            word = tokens[i]
            synonyms = get_synonyms(word)
            if synonyms:
                # 选择第一个同义词替换原词
                replaced_word = list(synonyms)[0]
                tokens[i] = replaced_word
        return ' '.join(tokens)

    def add_reason_to_train_data(train_data, causal_graph):
        for sample in train_data:
            premise = sample['premise']
            successors = list(causal_graph.successors(premise))
            if successors:
                next_node = successors[0]
                reason_text = "{Tag}KeyWords:"+str(next_node)
                sample['premise'] = premise + reason_text

    def add_new_data(train_data):
        train_data_copy = train_data.copy()
        augmented_data = []
        for sample in train_data_copy:
            premise = sample['premise']  # 提取样本中的 premise 特征
            tag_index = premise.find('{Tag}KeyWords:')  # 查找 {Tag}KeyWords: 的位置
            if tag_index != -1:
                text_before_tag = premise[:tag_index]  # 截取 {Tag}KeyWords: 前面的内容
                text_after_tag = premise[tag_index + len('{Tag}KeyWords:'):]  # 截取 {Tag} 后面的内容
                words_before_tag = text_before_tag.split()  # 将 {Tag}KeyWords: 前面的内容按空格分隔成单词列表
                words_after_tag = text_after_tag.split()  # 将 {Tag} 后面的内容按空格分隔成单词列表
                if words_before_tag and words_after_tag:
                    random_word_index_before = random.randint(0, len(words_before_tag) - 1)  # 随机选择一个前面单词的索引
                    random_word_index_after = random.randint(0, len(words_after_tag) - 1)  # 随机选择一个后面单词的索引
                    word_to_replace_before = words_before_tag[random_word_index_before]  # 获取要替换的前面单词
                    word_to_replace_after = words_after_tag[random_word_index_after]  # 获取要替换的后面单词
                    perturbed_word_before = replace_synonyms(word_to_replace_before)  # 替换单词的同义词
                    perturbed_word_after = replace_synonyms(word_to_replace_after)  # 替换单词的同义词
                    words_before_tag[random_word_index_before] = perturbed_word_before  # 将替换后的前面单词放回列表中
                    words_after_tag[random_word_index_after] = perturbed_word_after  # 将替换后的后面单词放回列表中
                    perturbed_text_before = ' '.join(words_before_tag)  # 将替换后的前面单词列表重新组合成文本
                    perturbed_text_after = ' '.join(words_after_tag)  # 将替换后的后面单词列表重新组合成文本
                    # print(perturbed_text_before)
                    # print(perturbed_text_after)
                    perturbed_text = perturbed_text_before + ' {Tag}KeyWords: ' + perturbed_text_after
                    sample['premise'] = perturbed_text
                    augmented_data.append(sample)  # 添加替换后的文本到增强数据列表中
        return augmented_data

    def addonly_new_data(train_data):
        train_data_copy = train_data.copy()
        augmented_data = []
        for sample in train_data_copy:
            premise = sample['premise']  # 提取样本中的 premise 特征
            words = premise.split()  # 将 premise 按空格分割成单词列表
            if words:
                random_word_index = random.randint(0, len(words) - 1)  # 随机选择一个单词索引
                word_to_replace = words[random_word_index]  # 获取要替换的单词
                perturbed_word = replace_synonyms(word_to_replace)  # 替换单词的同义词
                words[random_word_index] = perturbed_word  # 将替换后的单词放回列表中
                perturbed_text = ' '.join(words)  # 将替换后的单词列表重新组合成文本
                sample['premise'] = perturbed_text
                augmented_data.append(sample)  # 添加替换后的文本到增强数据列表中
        return augmented_data

    causal_graph = load_causal_graph('/home/amax/ghh/e-CARE-main/dataset/QArelated.json')
    add_reason_to_train_data(train_data, causal_graph)
    new_data1 = add_new_data(train_data)
    # print("new", new_data1)
    new_data2 = add_new_data(new_data1)
    train_data = train_data + new_data1
    add_reason_to_train_data(dev_data, causal_graph)
    add_reason_to_train_data(test_data, causal_graph)
    # print(train_data)
    # print(dev_data)
    # print(test_data)

    # Tokenization
    logger.info("[INFO] Tokenization and Padding for Data")
    train_ids, train_mask, train_seg_ids, train_labels, train_length = quick_tokenize(train_data, hps)
    dev_ids, dev_mask, dev_seg_ids, dev_labels, dev_length = quick_tokenize(dev_data, hps)
    test_ids, test_mask, test_seg_ids, test_labels, test_length = quick_tokenize(test_data, hps)

    # Dataset and DataLoader
    logger.info("[INFO] Creating Dataset and splitting batch for data")
    TRAIN = TensorDataset(train_ids, train_seg_ids, train_mask, train_labels, train_length)
    DEV = TensorDataset(dev_ids, dev_seg_ids, dev_mask, dev_labels, dev_length)
    TEST = TensorDataset(test_ids, test_seg_ids, test_mask, test_labels, test_length)
    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')
    model = pretrained_model(hps)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    if hps.loss_func == "CrossEntropy":
        loss_function = nn.CrossEntropyLoss(reduction='mean')
    else:
        loss_function = nn.BCEWithLogitsLoss(reduction='mean')

    # Multi-Gpu training
    if hps.cuda:
        gpu_ids = [int(x) for x in hps.gpu.split(' ')]
        model = model.cuda()
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)




    # training
    logger.info("[INFO] Start Training")
    step = 0
    patient = 0
    best_accuracy = 0
    stop_train = False

    for epoch in range(hps.epochs):
        logger.info('[Epoch] {}'.format(epoch))
        t = trange(len(train_dataloader))
        epoch_step = 0
        total_loss = 0
        for i, batch in zip(t, train_dataloader):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                batch = tuple(term.cuda() for term in batch)

            sent, seg_id, atten_mask, labels, length = batch
            probs = model(sent, atten_mask, seg_ids=seg_id, length=length)


            if hps.loss_func == 'CrossEntropy':
                loss = loss_function(probs, labels)
            else:
                loss = loss_function(probs.squeeze(1), labels.float())

            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss/(epoch_step+1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()


            if step % hps.evaluation_step == 0 and step != 0:
                model.eval()

                with torch.no_grad():
                    print('\n')
                    logger.info("[Dev Evaluation] Strain Evaluation on Dev Set")
                    if hps.loss_func == 'CrossEntropy':
                        dev_accu, dev_exact_accu, dev_loss = evaluation(hps, dev_dataloader, model, loss_function)
                        print('\n')
                        logger.info("[Dev Metrics] Dev Soft Accuracy: \t{}".format(dev_accu))
                        logger.info("[Dev Metrics] Dev Exact Accuracy: \t{}".format(dev_exact_accu))
                    else:
                        dev_accu, dev_loss = evaluation(hps, dev_dataloader, model, loss_function)
                        print('\n')
                        logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(dev_accu))
                    logger.info("[Dev Metrics] Dev Loss: \t{}".format(dev_loss))

                    if dev_accu >= best_accuracy:
                        patient = 0
                        best_accuracy = dev_accu
                        logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
                        if hps.hyp_only:
                            torch.save(model, os.path.join(hps.save_dir, 'discriminate_'+hps.model_name + '_hyp'))
                        else:
                            torch.save(model, os.path.join(hps.save_dir, 'discriminate_'+hps.model_name))
                        logger.info("[Test Evaluation] Strain Evaluation on Test Set")
                        if hps.loss_func == 'CrossEntropy':
                            te_soft_accu, te_exact_accu, te_loss = evaluation(hps, test_dataloader, model, loss_function)
                            print('\n')
                            logger.info("[Test Metrics] Test Soft Accuracy: \t{}".format(te_soft_accu))
                            logger.info("[Test Metrics] Test Exact Accuracy: \t{}".format(te_exact_accu))
                        else:
                            te_accu, te_loss = evaluation(hps, test_dataloader, model, loss_function)
                            print('\n')
                            logger.info("[Test Metrics] Test Accuracy: \t{}".format(te_accu))
                        logger.info("[Test Metrics] Test Loss: \t{}".format(te_loss))
                    else:
                        patient += 1

                    logger.info("[Patient] {}".format(patient))

                    if patient >= hps.patient:
                        logger.info("[INFO] Stopping Training by Early Stopping")
                        stop_train = True
                        break
            step += 1

        if stop_train:
            break


if __name__ == '__main__':
    main()





