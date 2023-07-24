import xlrd
import jieba
import pandas as pd
from roformer测试 import gen_synonyms
# data = pd.DataFrame(pd.read_excel(r'../data/专利表.xls', skiprows=1))
df = xlrd.open_workbook(r'../data/企业专利信息.xls')
df_sheet = df.sheet_by_name('企业专利信息')
df_sheet2 = df.sheet_by_name('企业基本信息')
k = 1
nRows = df_sheet.nrows
first_cor = df_sheet.row_values(2)[3]
# 不添加title

# s = first_cor + ':'
print('total rows: ', nRows)
for i in range(nRows):
    s = ''
    if i == 0:
        continue
    if i % 100 == 0:
        print(i)
    row = df_sheet.row_values(i)
    if first_cor == row[3]:
        tmp = row[1]
        aug_sen = gen_synonyms(tmp, k=2)
        if len(aug_sen) >= 2:
            sen = aug_sen[0] + aug_sen[1]
            textlist = jieba.cut(sen)
            text = ' '.join(textlist)
            s = text
        else:
            s = tmp
    else:
        s = '\n'
        k = k + 1
        name = df_sheet2.row_values(k)[0]
        # print(name)
        while row[3] != name:
            s = s + 'none' + '\n'
            k = k + 1
            name = df_sheet2.row_values(k)[0]
        # 不添加title
        # s = s + row[3] + ':'
        first_cor = row[3]
        tmp = row[1]
        aug_sen = gen_synonyms(tmp, k=2)
        if len(aug_sen) >= 2:
            sen = aug_sen[0] + aug_sen[1]
            textlist = jieba.cut(sen)
            text = ' '.join(textlist)
            s = s + text
        else:
            s = s + tmp
    with open('../data/aug_zhuanli.txt', 'a', encoding='utf-8') as f:
        f.write(s)
        f.close()

# nRows2 = df_sheet2.nrows
# p = ''
# for j in range(nRows2):
#     if j == 0 :
#         continue
#     row = df_sheet2.row_values(j)
#     sen1 = jieba.cut(row[0])
#     sen2 = jieba.cut(row[9])
#     sen3 = jieba.cut(row[10])
#     sen4 = jieba.cut(row[11])
#     corp1 = ' '.join(sen1)
#     corp2 = ' '.join(sen2)
#     corp3 = ' '.join(sen3)
#     corp4 = ' '.join(sen4)
#     p = p + ' ' + corp1 + ' ' + corp2 + ' ' + corp3 + ' ' + corp4 + '\n'
# with open('../data/cor_title.txt', 'w', encoding='utf-8') as t:
#     t.write(p)



