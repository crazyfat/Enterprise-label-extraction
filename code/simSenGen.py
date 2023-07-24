#! -*- coding: utf-8 -*-
# RoFormer-Sim base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
maxlen = 64
#
# 模型配置
config_path = '/data/zhangyuanqing/PLM_pocket/chinese_roformer-sim-char_L-6_H-384_A-6/bert_config.json'
checkpoint_path = '/data/zhangyuanqing/PLM_pocket/chinese_roformer-sim-char_L-6_H-384_A-6/bert_model.ckpt'
dict_path = '/data/zhangyuanqing/PLM_pocket/chinese_roformer-sim-char_L-6_H-384_A-6/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear'
)

encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])
#
#
class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]
#
#
# synonyms_generator = SynonymsGenerator(
#     start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
# )
#
#
# def gen_synonyms(text, n=100, k=20, mask_idxs=[]):
#     ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
#     做法：用seq2seq生成，并用encoder算相似度并排序。
#     '''
#     r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
#     r = [i for i in set(r) if i != text]
#     r = [text] + r
#     X, S = [], []
#     for t in r:
#         x, s = tokenizer.encode(t)
#         X.append(x)
#         S.append(s)
#     X = sequence_padding(X)
#     S = sequence_padding(S)
#     Z = encoder.predict([X, S])
#     Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
#     argsort = np.dot(Z[1:], -Z[0]).argsort()
#     return [r[i + 1] for i in argsort[:k]]
#
#
# out = gen_synonyms(u'先进制造与自动化 电力系统与设备 配电与用电技术 中群电气有限公司', k=5)
# print(out)


class Roformer():
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
        self.maxlen = 64

        # 模型配置
        self.config_path = '/data/zhangyuanqing/PLM_pocket/chinese_roformer-sim-char_L-6_H-384_A-6/bert_config.json'
        self.checkpoint_path = '/data/zhangyuanqing/PLM_pocket/chinese_roformer-sim-char_L-6_H-384_A-6/bert_model.ckpt'
        self.dict_path = '/data/zhangyuanqing/PLM_pocket/chinese_roformer-sim-char_L-6_H-384_A-6/vocab.txt'

        # 建立分词器
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)  # 建立分词器

        # 建立加载模型
        self.roformer = build_transformer_model(
            self.config_path,
            self.checkpoint_path,
            model='roformer',
            application='unilm',
            with_pool='linear'
        )

        self.encoder = keras.models.Model(self.roformer.inputs, self.roformer.outputs[0])
        self.seq2seq = keras.models.Model(self.roformer.inputs, self.roformer.outputs[1])
        self.synonyms_generator = SynonymsGenerator(
            start_id=None, end_id=self.tokenizer._token_end_id, maxlen=maxlen
        )

    def gen_synonyms(self, text, n=100, k=20, mask_idxs=[]):
        ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
        做法：用seq2seq生成，并用encoder算相似度并排序。
        '''
        r = self.synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
        r = [i for i in set(r) if i != text]
        r = [text] + r
        X, S = [], []
        for t in r:
            x, s = self.tokenizer.encode(t)
            X.append(x)
            S.append(s)
        X = sequence_padding(X)
        S = sequence_padding(S)
        Z = self.encoder.predict([X, S])
        Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5
        argsort = np.dot(Z[1:], -Z[0]).argsort()
        return [r[i + 1] for i in argsort[:k]]
