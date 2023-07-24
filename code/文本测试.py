import jiagu
import jieba
from nlpcda import Similarword, Homophone
from harvesttext import HarvestText
text = "一种应急启动的漏电断路器 小型漏电断路器 漏电断路器 交流接触器 塑壳断路器 一种微型断路 器插接连接器 一种漏电断路器本地兼远程分合闸装置 一种具有零线取源功能的断路器 一种后拉式漏电脱扣器 一种分合闸通断显示装置 快接线式断路器 一种微型断路器插接连接器 重合闸塑壳断路器 后拉式漏电脱扣器 一种微型断路器插接连接器 一种断路器大电流电子脱扣装置 小型断路器 一种电力采集执行装置 一种基于漏电断路器的网关连接装置 网关 断路器（带飞梭按键的重合闸塑壳断路器） 一种塑壳断路器脱扣机构用调试装置 一种进线端互感器装置 一种万能式断路器用灭弧室 一种温度检测系统及其断路器 一种漏电断路器 一种带有飞梭按键的重合闸塑壳断路器 重合闸断路器 一种具有物联功能的漏电断路器 一种用于万能式断路器的端子安装座 一种手动方式分合闸的齿轮组结构 一种本地锁定装置 一种新式塑壳断路器 一种双棘轮传动装置 小型漏电断路器 一种微型断路器插接连接器 一种塑壳断路器的快速拼接弹锁装置  "
text3 = '本实用新型为一种应急启动的漏电断路器，包括至少一个断路器模块和自动分合闸模块，所述断路器模块设置在相应的断路器壳体内，所述自动分合闸模块设置在相应的自动分合闸壳体内，所述自动分合闸模块包括第一线路板、第二线路板以及分合闸机构，所述断路器模块包括漏电互感器，所述漏电互感器与所述第一线路板连接，所述第一线路板竖直设置在自动分合闸壳体内，分合闸机构设置在第一线路板的内侧，所述第二线路板位于所述自动分合闸壳体的顶部且与所述第一线路板垂直设置，所述第一线路板与所述第二线路板电连接，所述第二线路板上设置有开关以及光耦，所述第一线路板上设置有控制模块，所述控制模块能够接收光耦电信号并控制漏电互感器断开。'
text2 = '''
该研究主持者之一、波士顿大学地球与环境科学系博士陈池（音）表示，“尽管中国和印度国土面积仅占全球陆地的9%，但两国为这一绿化过程贡献超过三分之一。考虑到人口过多的国家一般存在对土地过度利用的问题，这个发现令人吃惊。”
NASA埃姆斯研究中心的科学家拉玛·内曼尼（Rama Nemani）说，“这一长期数据能让我们深入分析地表绿化背后的影响因素。我们一开始以为，植被增加是由于更多二氧化碳排放，导致气候更加温暖、潮湿，适宜生长。”
“MODIS的数据让我们能在非常小的尺度上理解这一现象，我们发现人类活动也作出了贡献。”
NASA文章介绍，在中国为全球绿化进程做出的贡献中，有42%来源于植树造林工程，对于减少土壤侵蚀、空气污染与气候变化发挥了作用。
据观察者网过往报道，2017年我国全国共完成造林736.2万公顷、森林抚育830.2万公顷。其中，天然林资源保护工程完成造林26万公顷，退耕还林工程完成造林91.2万公顷。京津风沙源治理工程完成造林18.5万公顷。三北及长江流域等重点防护林体系工程完成造林99.1万公顷。完成国家储备林建设任务68万公顷。
'''
print(text)
name = '先进制造与自动化 电力系统与设备 配电与用电技术 中群电气有限公司'
print(name)
keywords = jiagu.keywords(text3, 6) # 关键词
print("before_aug:", keywords)

smw = Similarword(create_num=3, change_rate=0.3)
rs1 = smw.replace(text)

print('随机同义词替换>>>>>>')
a = ''
for s in rs1:
    a += s
smw = Homophone(create_num=3, change_rate=0.3)
rs1 = smw.replace(text)

print('随机近义字替换>>>>>>')
for s in rs1:
    a += s

config = {
        'model_path': '/data/zhangyuanqing/PLM_pocket/chinese_simbert_L-4_H-312_A-12',
        'CUDA_VISIBLE_DEVICES': '0,1',
        'max_len': 32,
        'seed': 1
}
# simbert = Simbert(config=config)
sent = '把我的一个亿存银行安全吗'
# synonyms = simbert.replace(sent=sent, create_num=3)
# print(synonyms)
keywords = jiagu.keywords(a, 6) # 关键词
print("after_aug:", keywords)
print("title_topic:", jiagu.keywords(name, 3))
ht = HarvestText()
seg_list = jieba.cut(sent)
print(list(seg_list))
tfidf_keyword = ht.extract_keywords(text, 6, method="jieba_tfidf")
textrank_keyword = ht.extract_keywords(text, 6, method="textrank")
print(tfidf_keyword)
print(textrank_keyword)