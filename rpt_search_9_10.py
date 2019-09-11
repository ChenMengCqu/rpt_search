
# coding: utf-8

# In[26]:


# coding: utf-8

import pandas as pd
import numpy as np
#dcg
def dcg(label,pos): #  通过求dcg和idcg来计算ndcg
    dcg = 0
    for i in range(len(label)):
        dcg += (label[i] * np.reciprocal(np.log2(pos[i]+1)))
    return dcg

#auc
def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return auc / (len(pos)*len(neg))
#logloss
def logloss(label, pre, eps=1e-15):
    y_true = np.array(label)
    y_pred = np.array(pre)
    p = np.clip(y_pred, eps, 1-eps)
    loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1-p))
#     print(loss / len(y_true))
    return loss / len(y_true)

# mrr  rank是模型预测的rank
def mrr(file):  
    file_1=file[file['y']==1]
    label = file['y'].tolist()
    qurry_1_pos={}
    sum = 0
    #list(v)[0] 返回预测的第一个相关的item在搜索结果中所在位置
    for k,v in file_1.groupby('qurry_id')['rank_id']:
        qurry_1_pos[k]= list(v)[0]
        sum += 1 / qurry_1_pos[k]
    avg = sum/len(label)
    return avg
#idcg
def idcg(label,pos):
    idcg = 0
    for i in range(len(label)):
        idcg += (label[i] * np.reciprocal(np.log2(pos[i]+1)))
    return idcg
#MAP
def MAP_compare(file):
    # query_id数量
    qurry_num = len(file['qurry_id'].unique())
    #每个qurry_id对应搜索出的记录数
    qurry_len={}
    for k,v in file.groupby('qurry_id')['y']:
        qurry_len[k]=len(v)
   #  所有正样本对应的记录
    file_1=file[file['y']==1]
   #每一个query对应预测的item的排序位置
    qurry_1_pos={}
    for k,v in file_1.groupby('qurry_id')['rank_pre']:
        qurry_1_pos[k]=list(v)
    #每个qurry真实的item排序位置
    qurry_1_true={}
    for k,v in file_1.groupby('qurry_id')['rank_id']:
        qurry_1_true[k]=list(v)
    # 所有query对应相关的值（不包含没有任何匹配的query）
    query_sum=0
    #预测MAP
    for k,v in qurry_1_pos.items():
        v_len=len(v) #同一个qurry中所有正样本数
        temp=0
        cnt=0
         #第一层，在同一个qurry内
        for i in range(v_len):
            cnt=cnt+1
            temp=temp+cnt/v[i] 
        query_sum=query_sum+temp/qurry_len[k]
        
    result_pre=query_sum/qurry_num
    
    #真实MAP
    query_sum_ture = 0
    for k_true,v_true in qurry_1_true.items():
        v_len_true = len(v_true) 
        temp=0
        cnt=0
        for i in range(v_len_true):
            cnt=cnt+1
            temp += cnt/v_true[i] 
        query_sum_ture += temp/qurry_len[k_true]
        
    result_true=query_sum_ture/qurry_num
    delta_map = result_true - result_pre
    return (result_pre,result_true,delta_map)

if __name__ == '__main__':
    #文件路径定义
    #path = “”
    file = pd.read_csv('test.txt',sep='\t')

    #dcg在源文件条件下求得
    label = file['y'].tolist()
    pos = file['rank_id'].tolist()
    dcg = dcg(label, pos)
    #新增一列rank_pre，表示预测的排序
    # 返回值是一个带有multiindex的dataframe数据，其中level=0为groupby的by列，而level=1为原index，通过设置group_keys=False去掉第一层index
    file1 = file.groupby('qurry_id',group_keys=False).apply(lambda x: x.sort_values('y_pred', ascending=False))
    l_all = []
    for (k,v) in file1.groupby('qurry_id'):
        l =[]
        for i in range(1,len(v)+1):
            l.append(i)
        l_all = l_all + l
    file1['rank_pre']=l_all
    #计算指标 auc logloss  mrr  ndcg map
    #auc
    y_true = file1['y'].tolist()
    y_pre = file1['y_pred'].tolist()
    auc = AUC(y_true, y_pre)
    #logloss
    logloss = logloss(y_true, y_pre)
    #mrr
    rank = file1['rank_pre'].tolist()
    mrr = mrr(file1)
    #idcg
    pos1 = file1['rank_pre'].tolist()
    idcg = idcg(y_true,pos1)
    #ndcg 暂时不要这个指标？
    ndcg = dcg / idcg
    #MAP
    map_list = MAP_compare(file1)
    qurry_arry = file1['qurry_id'].unique()
    print("auc:" + str(auc) + "  logloss:"+ str(logloss) + "  mrr:"+ str(mrr)+ "  ndcg:"+ str(ndcg)+ "  map_value:" + str(map_list))


