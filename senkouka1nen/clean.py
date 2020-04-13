import numpy as np
import time
import pandas as pd
import csv

#endings_n = choice_matrix_s[i][n]
#label = correct_label
#startphrase = sent
#sent2 = MASKより後:sent2
#sent1 = MASKより前:sent1

sent1 = []
sent2 = []
correct_label = []
comp = []

# correct[] : 正解ラベル
with open('oda-correct.txt', 'r', encoding='utf-8') as f:
    correct = f.readlines()
for i, line in enumerate(correct):
    correct[i] = line.replace('\n', '')

# choice_matrix[][] : 3717行*4列　行：各文章　列：候補前置詞
# choice_matrix_s[][] : 列要素シャッフル　こっち使う.
with open('oda-predicted.txt', 'r', encoding='utf-8') as f:
    choice = f.readlines()
choice_matrix = np.zeros((len(choice), 4), dtype=object)

for i, line in enumerate(choice):  
    line = line.split()
    for j in range(4):
        choice_matrix[i][j] = line[-(j+1)]
        if choice_matrix[i][j] not in comp:
            comp.append(choice_matrix[i][j])
    '''choice_matrix[i][0] = 'of'
    choice_matrix[i][1] = 'to'
    choice_matrix[i][2] = 'in'
    choice_matrix[i][3] = 'for'
    '''
    if correct[i] not in choice_matrix[i, ]:
        choice_matrix[i][3] = correct[i]

choice_matrix_s = choice_matrix
for column in choice_matrix_s:
    np.random.shuffle(column)

#ラベル付け
for i, line in enumerate(choice_matrix_s):
    for j, vocab in enumerate(line):
        if correct[i] == vocab:
            correct_label.append(j)
    

# sentence : データセット
with open('sentence.tsv', 'r', encoding='utf-8') as f:
    sentence = f.readlines()

sent = []
line_sentence = []

for line in sentence:
    line = line.replace('"', '')
    line_sentence.append(line.replace('\n', ''))
    sent.append(line.split('[MASK]'))

for bun in sent:
    sent1.append(bun[0].replace('\n', ''))
    sent2.append(bun[1].replace('\n', ''))
    
for i, line in enumerate(choice_matrix_s):
    for j, vocab in enumerate(line):
        choice_matrix_s[i][j] = vocab + sent2[i]
#choice_matrix_sがending0~4に入る

with open('train.csv', 'r', encoding='utf-8') as f:
    data = list(csv.reader(f))
    for i in range(len(correct_label)):
        data[i+1][4] = line_sentence[i]
        data[i+1][5] = sent1[i]
        data[i+1][7] = choice_matrix_s[i][0]
        data[i+1][8] = choice_matrix_s[i][1]
        data[i+1][9] = choice_matrix_s[i][2]
        data[i+1][10] = choice_matrix_s[i][3]
        data[i+1][11] = correct_label[i]

with open('train_sample.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)
