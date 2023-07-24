# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/1 15:39
# @author  : Mo
# @function:

from nlg_yongzhuo import lda, lsi, nmf

docs = """AutoML机器学习自动化与NNI
原创大漠帝国 最后发布于2020-02-29 19:46:21 阅读数 221  收藏
编辑 展开
一、AutoML简介
        AutoML(Automated Machine Learning)，中文可以翻译为自动机器学习，我比较喜欢叫它“机器学习自动化”，更加接近人们所津津乐道的通用人工智能吧。
        人们一直有个朴素的想法，可以有一个通用的AI系统，它包罗万象，能够对整个宇宙进行建模，对我们遇到的一切问题，都给出解决办法。这在幻想书籍中数见不新鲜，比如漫威电影中钢铁侠的人工智能贾维斯，又比如说芯片系统流派的网络小说。不过这些大概可以算是人工智能的高级模式了吧，人们还是很宽容的，没有期待一步到位。
       现在算是AI的高潮期，尤其是以深度学习DL为代表的当代人工智能技术的成功，给以人类以无限的想象空间。那么，降低要求，以DL技术为基础，去开发一个低配版通用人工智能，也是可以的吧。所以，随着人工智能的火爆，2014年以来，AutoML也越发火热起来。
       深度学习时代的鲜明特征是大数据量、深层次网络、特征学习与端到端学习。我们希望能够从数据一步得到模型，而不需要其他的什么人为参与过程。如果再加上语音助手什么的，或许我们就能达到浅层次通用人工智能的目标呢。在深度学习DL模型架构难以取得更大突破的时候，给它再开辟一条道路呢。一如蒸馏模型，又如MobileNet。
        工程化和应用级市场，更能带来意想不到的惊喜。这一点，从近年来微软开源的AutoML工具NNI大受欢迎中，可以管中窥豹。

二、AutoML特性
        从比较出名的开源Auto平台、互联网大厂AutoML云产品，以及AI公司的AutoML软件来看，一般包括特征工程(FE，Auto feature engine)、神经网络搜索(NAS，Neural Architecture Search) 和超参数优化(HPO，Hyper-parameter optimization) 等功能，如下图所示：
        可能还存在其他一些小功能，如数据增强(几何,颜色), 激活函数(swish,Hybrid DNN), 归一化方法(Switchable Normalization, BN, IN, LN, GN), 优化方法(Neural Optimizer Search, sgd，rmsprop，adam, 衰减, 函数的组合), 优化目标(AM-LFS, Learning to teach with dynamic loss functions), 模型剪枝(AMC), 模型量化(HAQ), 部署上线等。
        AutoML优点：可用于传统机器学习、图像等较成熟领域，自动化摒弃了人为因素的干扰、增强泛化性；
                     缺点：消耗资源大、优化方法可能达不到经验模型甚至是严重偏向。

三、 NNI
        NNI (Neural Network Intelligence，[翻译为神经网络智能？]) 是微软开源的自动机器学习（AutoML）的Python工具包。NNI 通过 nni_manager模块 等管理 AutoML 的 Experiment (实验)，调度并运行各种调优算法生成的 Trial (尝试) 任务，来完成搜索最优神经网络架构、超参数等。同时支持本机，远程服务器，单机，多机，OpenPAI，Kubeflow，K8S和其它云服务等训练环境。
        对比其他开源项目，或大公司产品可以发现，NNI支持的神经网络结构搜索、超参数优化等调优算法更多，功能最强大。
        以我的使用体验来看，NNI更像一个黑盒，浅度用户使用可能比较舒服。使用nni的SDK可以完美嵌入自己的网络结构进行超参数优化，详情如下:
        超参数优化需要定义搜索空间search_space.json，NNI配置config.yml，以及主程序调用main.py函数。
        此外，NNI还需要用特定命令行启动，自由度似乎不太够。
希望对你有所帮助!
————————————————
版权声明：本文为CSDN博主「大漠帝国」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/rensihui/article/details/104578756""".replace(" ", "").replace('"', '')

sums_nmf = nmf.summarize(docs, num=6, topic_min=8)
print("nmf:")
for sum_ in sums_nmf:
    print(sum_)

sums_lsi = lsi.summarize(docs, num=6, topic_min=8)
print("lsi:")
for sum_ in sums_lsi:
    print(sum_)

sums_lda = lda.summarize(docs, num=6, topic_min=8)
print("lda:")
for sum_ in sums_lda:
    print(sum_)