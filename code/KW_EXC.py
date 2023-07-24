# -*- coding: utf-8 -*-

import math
import jieba
import jiagu
from harvesttext import HarvestText
import jieba.posseg as psg
from gensim import corpora, models
from jieba import analyse
import functools
import time
from langconv import *
from collections import Counter
# from roformer测试 import gen_synonyms


# 繁体转简体
def TraditionalToSimplified(content):
    line = Converter("zh-hans").convert(content)
    return line


def keyword_list(keypath='../data/test_trg.txt'):
    inputs = open(keypath, 'r', encoding='utf-8')
    keyword_list = []
    for line in inputs:
        line = TraditionalToSimplified(line)
        keyword_list.append(line.strip().replace(';', ' ').split())
    inputs.close()
    return keyword_list


def get_first_three_keywords(keywordslist, row=1):
    length = len(keywordslist[row - 1])
    first_word = keywordslist[row - 1][0]
    m = min(length, 3)
    first_three_keywords = set(keywordslist[row - 1][:m])
    return first_word, first_three_keywords


# 停用词表加载方法
def get_stopword_list():
    # 停用词表存储路径，每一行为一个词，按行读取进行加载
    # 进行编码转换确保匹配准确率
    stop_word_path = '../data/stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path, encoding='utf-8').readlines()]
    return stopword_list


# 分词方法，调用结巴接口
def seg_to_list(sentence, pos=False):
    if not pos:
        # 不进行词性标注的分词方法
        # seg_list = jieba.cut(sentence)
        sentence = TraditionalToSimplified(sentence)
        seg_list = sentence.strip().replace(';', ' ').split()
    else:
        # 进行词性标注的分词方法
        seg_list = psg.cut(sentence)
    return seg_list


# 去除干扰词
def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    # 根据POS参数选择是否词性过滤
    ## 不进行词性过滤，则将词性都标记为n，表示全部保留
    for seg in seg_list:
        if not pos:
            word = seg
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag
        if not flag.startswith('n'):
            continue
        # 过滤停用词表中的词，以及长度为<2的词
        if not word in stopword_list:
            filter_list.append(word)

    return filter_list


# 数据加载，pos为是否词性标注的参数，corpus_path为数据集路径
def load_data(pos=False, corpus_path='../data/test_src.txt'):
    # 调用上面方式对数据集进行处理，处理后的每条数据仅保留非干扰词
    doc_list = []
    for line in open(corpus_path, 'r', encoding='utf-8'):
        seg_list = seg_to_list(line, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)

    return doc_list


# idf值统计方法
def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    tt_count = len(doc_list)

    # 每个词出现的文档数
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0.0) + 1.0

    # 按公式转换为idf值，分母加1进行平滑处理
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(tt_count / (1.0 + v))

    # 对于没有在字典中的词，默认其仅在一个文档出现，得到默认idf值
    default_idf = math.log(tt_count / (1.0))
    return idf_dic, default_idf


#  排序函数，用于topK关键词的按值排序
def cmp(e1, e2):
    import numpy as np
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


# 主题模型
class TopicModel(object):
    # 三个传入参数：处理后的数据集，关键词数量，具体模型（LSI、LDA），主题数量
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        # 使用gensim的接口，将文本转为向量化表示
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词，根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics
        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 得到数据集的主题-词分布
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        return lsi

    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics)
        # lda.save('model1')
        # lda = lda.load('./model/model1')
        # print("lad model load successfully")
        return lda

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}

        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        l = sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]
        if len(l) == 0:
            first_keyword = ''
        else:
            first_keyword = l[0][0]
        first_three_keywords = set()

        n = 1
        for k, v in l:
            first_three_keywords.add(k)
            if n == 3:
                break
            n = n + 1
        return first_keyword, first_three_keywords, l

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)
        dictionary = list(set(dictionary))
        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list


def LDA(savepath, keyword_num=5, end=2, save=True):
    n = 1  # 当前的行数
    string = ''  # 储存预测结果用于存储
    topic_model = TopicModel(doc_list_train_src, keyword_num, model='LDA')
    for text in doc_list_test_src:

        first_keyword_pred, first_three_keywords_pred, l = topic_model.get_simword(text)
        if first_keyword_pred == '':
            n = n + 1
            continue
        if 'none' in first_three_keywords_pred:
            first_three_keywords_pred = {}
        str2 = str(n) + ':' + '\t' + str(list(first_three_keywords_pred)) + '\n'
        string = string + str2
        if n == end:
            break
        n = n + 1
    print('LDA model result :')

    if save:  # 是否保存结果
        outputs = open(savepath, 'w', encoding='utf-8')
        outputs.write('id' + '\t' + 'pred' + '\t' + '\n')
        outputs.write(string)

        outputs.close()
        print("result save successfully")


def keyword_exc(savepath, data_aug, keyword_num=5, end=2, save=True):
    n = 1  # 当前的行数
    string = ''  # 储存预测结果用于存储
    topic_model = TopicModel(doc_list_train_src, keyword_num, model='LDA')
    ht = HarvestText()
    for idx, text in enumerate(doc_list_test_src):
        if idx % 100 == 0:
            print(idx)
        titles = title_src[idx]
        joined_titles = ''.join(titles)
        joined_text = ''.join(text)
        first_keyword_pred, first_three_keywords_pred, l = topic_model.get_simword(text)
        first_three_keywords_pred = list(first_three_keywords_pred)
        _, title_words, _ = topic_model.get_simword(titles)
        title_words = list(title_words)
        textrank_keyword = ht.extract_keywords(joined_text, 3, method="textrank")
        textrank_titles = ht.extract_keywords(joined_titles, 2, method="textrank")
        tfidf_keyword = ht.extract_keywords(joined_text, 3, method="jieba_tfidf")
        tfidf_titles = ht.extract_keywords(joined_titles, 2, method="jieba_tfidf")

        if first_keyword_pred == '':
            n = n + 1
            continue
        if 'none' in first_three_keywords_pred:
            first_three_keywords_pred = {}
            collection_titles = Counter(title_words)
            most_title = collection_titles.most_common(3)
            str2 = str(n) + ':' + '\t' + str(list(most_title)) + '\n'
        else:
            textrank_pred = textrank_keyword
            tfidf_pred = tfidf_keyword
            textrank_pred_titles = textrank_titles
            tfidf_pred_titles = tfidf_titles

            union_keyword = first_three_keywords_pred + tfidf_pred + textrank_pred + title_words \
                            + textrank_pred_titles + tfidf_pred_titles
            collection_keywords = Counter(union_keyword)
            most_keyword = collection_keywords.most_common(3)

            str2 = str(n) + ':' + '\t' + str(list(most_keyword)) + '\n'
        string = string + str2
        if n == end:
            break
        n = n + 1
    print('union model result :')

    if save:  # 是否保存结果
        outputs = open(savepath, 'w', encoding='utf-8')
        outputs.write('id' + '\t' + 'pred' + '\t' + '\n')
        outputs.write(string)

        outputs.close()
        print("result save successfully")


def TFIDF(savepath, keyword_num=5, end=2, save=True):
    n = 1  # 当前的行数

    string = ''  # 储存预测结果用于存储
    ht = HarvestText()
    for text in doc_list_test_src:
        text = ''.join(text)
        tfidf_keyword = ht.extract_keywords(text, 5, method="jieba_tfidf")
        str2 = str(n) + ':' + '\t' + str(tfidf_keyword) + '\n'
        string = string + str2
        if n == end:
            break
        n = n + 1
    print('tfidf model result :')

    if save:  # 是否保存结果
        outputs = open('../result/tfidf.txt', 'w', encoding='utf-8')
        outputs.write('id' + '\t' + 'pred' + '\t' + '\n')
        outputs.write(string)

        outputs.close()
        print("result save successfully")


def TEXTRANK(savepath, keyword_num=5, end=2, save=True):
    n = 1  # 当前的行数

    string = ''  # 储存预测结果用于存储
    ht = HarvestText()
    for text in doc_list_test_src:
        text = ''.join(text)
        textrank_keyword = ht.extract_keywords(text, 5, method="textrank")
        str2 = str(n) + ':' + '\t' + str(textrank_keyword) + '\n'
        string = string + str2
        if n == end:
            break
        n = n + 1
    print('textrank model result :')

    if save:  # 是否保存结果
        outputs = open('../result/textrank.txt', 'w', encoding='utf-8')
        outputs.write('id' + '\t' + 'pred' + '\t' + '\n')
        outputs.write(string)

        outputs.close()
        print("result save successfully")


if __name__ == '__main__':
    print(time.strftime('Strat to prepare:%H:%M:%S', time.localtime(time.time())))
    # 预处理加载好训练集和关键词
    doc_list_train_src = load_data(corpus_path='../data/aug_zhuanli.txt')
    title_src = load_data(corpus_path='../data/cor_title.txt')
    doc_list_test_src = doc_list_train_src
    keywords_list = keyword_list('../data/test_trg.txt')  # 它和doc_list_test_trg区别就是没有去停用词

    # TFIDF(savepath = '../result/tfidf.txt', # TF-IDF结果保存路径
    #       keyword_num=5, # 候选词个数
    #       end = 0,  # end代表运行到第几行，设置为0时表示全部
    #       save=True) # 是否保存结果
    # LDA(savepath = '../result/lda.txt', # LDA结果保存路径
    #     keyword_num=5, # 候选词个数
    #     end = 0, # end代表运行到第几行，设置为0时表示全部
    #     save=True) # 是否保存结果
    # TEXTRANK(savepath = '../result/textrank.txt', # LDA结果保存路径
    #     keyword_num=5, # 候选词个数
    #     end = 0, # end代表运行到第几行，设置为0时表示全部
    #     save=True) # 是否保存结果
    keyword_exc(savepath='../result/aug_union_model.txt',  # LDA结果保存路径
                data_aug=False,
                keyword_num=5,  # 候选词个数
                end=0,  # end代表运行到第几行，设置为0时表示全部
                save=True,)  # 是否保存结果
    print(time.strftime('End time:%H:%M:%S', time.localtime(time.time())))