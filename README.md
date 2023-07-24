# enterprise--label-extraction

基于关键词提取技术的企业标签提取项目。

## 项目介绍

**文本清洗：** 主要采取了去停用词以及分词，参考[jieba](https://github.com/topics/jieba?l=rust&o=asc&s=stars);

**文本增强：** 基于语义增强的transformer模型[Roformer](https://huggingface.co/docs/transformers/model_doc/roformer) 生成增强后的文本对;

**关键词抽取：** 采用textRank、TF-IDF算法以及向量化的LDA模型进行关键词抽取；

**集成学习：** 使用集成投票法选择最终关键词。


## 项目启动

文本清洗和增强：[`./code/data_process.py`](https://github.com/crazyfat/enterprise--label-extraction/blob/main/code/data_process.py)

启动文件：[`./code/KW_EXC.py`](https://github.com/crazyfat/enterprise--label-extraction/blob/main/code/KW_EXC.py)


## 说明

训练数据涉及企业信息，本仓库并未公开；

抽取结果展示示例：[`./result/union_model.txt`](https://github.com/crazyfat/enterprise--label-extraction/blob/main/result/union_model.txt)



