# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:13:14 2020

@author: 47250
"""

import os
import pandas as pd
from collections import Counter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from sklearn import preprocessing,model_selection
import itertools
import xgboost as xgb
from xgboost import plot_importance
import warnings
import random
import datetime
warnings.filterwarnings("ignore")


os.chdir("D:\数据集\B")
beh = pd.read_csv('训练数据集_beh.csv',encoding='gbk',engine='python')
tag = pd.read_csv('训练数据集_tag.csv',encoding='gbk',engine='python')
trd = pd.read_csv('训练数据集_trd.csv',encoding='gbk',engine='python')

        
beh = beh.sort_values('id',axis = 0,ascending = True)
fwcishu = pd.DataFrame.from_dict(Counter(list(beh['id'])), orient='index',columns=['fwcishu'])
fwcishu = fwcishu.reset_index().rename(columns={'index':'id'})
count = beh.groupby(['id', 'page_no'])["id"].count().reset_index(name="Count")
maxpage = count.sort_values('Count', ascending=False).groupby('id', as_index=False).first()

beh = beh.drop('page_tm',axis=1)
beh.rename(columns={'Unnamed: 3':'page_tm'}, inplace = True)
#每个用户的最晚登陆时间
beh_lasttime = beh.groupby('id').max()
beh_lasttime['page_tm'] = beh_lasttime['page_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
beh_lasttime['page_tm'] = pd.to_datetime(beh_lasttime['page_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
max_time = max(beh_lasttime['page_tm'])#最大时间
beh_lasttime['day_diff_last'] = (max_time - beh_lasttime['page_tm'])#两个相减得到timedelta的时间差
beh_lasttime['day_diff_last'] = (beh_lasttime['day_diff_last'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
beh_lasttime = beh_lasttime.drop(['flag','page_no','page_tm'],axis=1)
#每个用户的最早登陆时间
beh_firsttime = beh.groupby('id').min()
beh_firsttime['page_tm'] = beh_firsttime['page_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
beh_firsttime['page_tm'] = pd.to_datetime(beh_firsttime['page_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
min_time = min(beh_firsttime['page_tm'])#最大时间
beh_firsttime['day_diff_first'] = (beh_firsttime['page_tm'] - min_time)#两个相减得到timedelta的时间差
beh_firsttime['day_diff_first'] = (beh_firsttime['day_diff_first'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
beh_firsttime = beh_firsttime.drop(['flag','page_no','page_tm'],axis=1)
behtime = pd.merge(beh_firsttime,beh_lasttime,on='id',how='outer')

late_time = '23'#晚于某个时间交易
early_time = '06'#早于某个时间交易 可以更改尝试最佳参数
beh['hour'] = beh['page_tm'].apply(lambda x:x[11:13])
beh_untime = beh[(beh['hour'] >= late_time) | (beh['hour'] <= early_time)]
#sum(beh_untime['flag'])/len(beh_untime)#不良率0.185
#sum(beh['flag'])/len(beh)#不良率0.157
beh_untime_count = beh_untime.groupby('id').count()
beh_untime_count = beh_untime_count.drop(['flag','page_no','hour'],axis=1).rename(columns={'page_tm':'untimecountapp'})#删除多余列
beh_untime_count = pd.merge(beh_untime_count,fwcishu,on='id',how='outer')
beh_untime_count['untimerate1'] = beh_untime_count['untimecountapp']/beh_untime_count['fwcishu']
beh_untime_count = beh_untime_count.drop(['fwcishu'],axis=1,inplace=False)

behnew = pd.merge(fwcishu,maxpage.drop(['Count'],axis=1,inplace=False),on='id',how='outer')
behnew = pd.merge(behnew,behtime,on='id',how='outer')
behnew = pd.merge(behnew,beh_untime_count,on='id',how='outer')
behnew = behnew.fillna(0)
tag1 = pd.merge(tag,behnew,on='id',how='left')

#temp = tag1[np.isnan(tag1['fwcishu']) == True]
#sum(temp['flag']) / len(temp)
#temp1 = tag1[np.isnan(tag1['fwcishu']) == False]
#sum(temp1['flag']) / len(temp1)

fwqueshi = []
for i in tag1['fwcishu']:
    if np.isnan(i) == True:
        fwqueshi.append('T')
    else:
        fwqueshi.append('F')  
tag1['fwqueshi'] = fwqueshi 



trd = trd.sort_values('id',axis = 0,ascending = True)
jycishu = pd.DataFrame.from_dict(Counter(list(trd['id'])), orient='index',columns=['jycishu'])
jycishu = jycishu.reset_index().rename(columns={'index':'id'})
trdlarge = trd[abs(trd['cny_trx_amt'])>=1000]
#trdcount = trd[abs(10<=trd['cny_trx_amt'])<1000]
trdsmall = trd[abs(trd['cny_trx_amt'])<10]
countbc = trd.groupby(['id','Dat_Flg1_Cd'])["id"].count().reset_index(name="bcCount")
countbclarge = trdlarge.groupby(['id','Dat_Flg1_Cd'])["id"].count().reset_index(name="bclargeCount")
countbcsmall = trdsmall.groupby(['id','Dat_Flg1_Cd'])["id"].count().reset_index(name="bcsmallCount")
countzhichu = countbc[countbc['Dat_Flg1_Cd']=='B'].rename(columns={'bcCount':'zhichucount'})
countshouru = countbc[countbc['Dat_Flg1_Cd']=='C'].rename(columns={'bcCount':'shourucount'})
countlargezhichu = countbclarge[countbclarge['Dat_Flg1_Cd']=='B'].rename(columns={'bclargeCount':'zhichulargecount'})
countlargeshouru = countbclarge[countbclarge['Dat_Flg1_Cd']=='C'].rename(columns={'bclargeCount':'shourulargecount'})
countsmallzhichu = countbcsmall[countbcsmall['Dat_Flg1_Cd']=='B'].rename(columns={'bcsmallCount':'zhichusmallcount'})
countsmallshouru = countbcsmall[countbcsmall['Dat_Flg1_Cd']=='C'].rename(columns={'bcsmallCount':'shourusmallcount'})
zhichu = trd[trd['Dat_Flg1_Cd']=='B']
zhichujine = pd.DataFrame(zhichu ['cny_trx_amt'].groupby(zhichu['id']).sum())
zhichujine = zhichujine.reset_index().rename(columns={'index':'id','cny_trx_amt':'zhichu'})
shouru = trd[trd['Dat_Flg1_Cd']=='C']
shourujine = pd.DataFrame(shouru['cny_trx_amt'].groupby(shouru['id']).sum())
shourujine = shourujine.reset_index().rename(columns={'index':'id','cny_trx_amt':'shouru'})
yingyu = pd.merge(zhichujine,shourujine,on='id',how='outer')
yingyu = yingyu.fillna(0)
yingyu['yingyu']=yingyu['zhichu']+yingyu['shouru']
countfangshi = trd.groupby(['id','Dat_Flg3_Cd'])["id"].count().reset_index(name="fangshiCount")
maxfangshi = countfangshi.sort_values('fangshiCount', ascending=False).groupby('id', as_index=False).first()
countfenlei = trd.groupby(['id','Trx_Cod1_Cd'])["id"].count().reset_index(name="fenleiCount")
maxfenlei = countfenlei.sort_values('fenleiCount', ascending=False).groupby('id', as_index=False).first()

late_time = '23'#晚于某个时间交易
early_time = '06'#早于某个时间交易 可以更改尝试最佳参数
trd['hour'] = trd['trx_tm'].apply(lambda x:x[11:13])
trd_untime = trd[(trd['hour'] >= late_time) | (trd['hour'] <= early_time)]
#sum(trd_untime['flag'])/len(trd_untime)#不良率0.185
#sum(trd['flag'])/len(trd)#不良率0.157
trd_untime_count = trd_untime.groupby('id').count()
trd_untime_count = trd_untime_count.drop(['flag','Dat_Flg1_Cd','Dat_Flg3_Cd','cny_trx_amt','Trx_Cod1_Cd','Trx_Cod2_Cd','hour'],axis=1).rename(columns={'trx_tm':'untimecount'})#删除多余列

#每个用户的最晚登陆时间
trd_lasttime = trd.groupby('id').max()
trd_lasttime['trx_tm'] = trd_lasttime['trx_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
trd_lasttime['trx_tm'] = pd.to_datetime(trd_lasttime['trx_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
max_time1 = max(trd_lasttime['trx_tm'])#最大时间
trd_lasttime['day_last'] = (max_time1 - trd_lasttime['trx_tm'])#两个相减得到timedelta的时间差
trd_lasttime['day_last'] = (trd_lasttime['day_last'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
trd_lasttime = trd_lasttime.drop(['flag','Dat_Flg1_Cd','Dat_Flg3_Cd','cny_trx_amt','Trx_Cod1_Cd','Trx_Cod2_Cd','hour','trx_tm'],axis=1)
#每个用户的最早登陆时间
trd_firsttime = trd.groupby('id').min()
trd_firsttime['trx_tm'] = trd_firsttime['trx_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
trd_firsttime['trx_tm'] = pd.to_datetime(trd_firsttime['trx_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
min_time1 = min(trd_firsttime['trx_tm'])#最大时间
trd_firsttime['day_first'] = (trd_firsttime['trx_tm'] - min_time1)#两个相减得到timedelta的时间差
trd_firsttime['day_first'] = (trd_firsttime['day_first'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
trd_firsttime = trd_firsttime.drop(['flag','Dat_Flg1_Cd','Dat_Flg3_Cd','cny_trx_amt','Trx_Cod1_Cd','Trx_Cod2_Cd','hour','trx_tm'],axis=1)
trdtime = pd.merge(trd_firsttime,trd_lasttime,on='id',how='outer')




trd1 = pd.merge(countzhichu.drop(['Dat_Flg1_Cd'],axis=1),countshouru.drop(['Dat_Flg1_Cd'],axis=1),on='id',how='outer')
trd2 = pd.merge(countlargezhichu.drop(['Dat_Flg1_Cd'],axis=1),countlargeshouru.drop(['Dat_Flg1_Cd'],axis=1),on='id',how='outer')
trd3 = pd.merge(countsmallzhichu.drop(['Dat_Flg1_Cd'],axis=1),countsmallshouru.drop(['Dat_Flg1_Cd'],axis=1),on='id',how='outer')
trd4 = pd.merge(maxfangshi.drop(['fangshiCount'],axis=1),maxfenlei.drop(['fenleiCount'],axis=1),on='id',how='outer')   
trd5 = pd.merge(yingyu,trd_untime_count,on='id',how='outer')
trdnew = pd.merge(trd1,trd2,on='id',how='outer')
trdnew = pd.merge(trd3,trdnew,on='id',how='outer')
trdnew = pd.merge(trd4,trdnew,on='id',how='outer')
trdnew = pd.merge(trd5,trdnew,on='id',how='outer')
trdnew = trdnew.fillna(0)

#去相反数，加正反列
zhengfu = []
for i in trdnew['yingyu']:
    if i>= 0:
        zhengfu.append('A')
    else:
        zhengfu.append('B')        
trdnew['zhengfu'] = zhengfu
trdnew['zhichu'] = trdnew['zhichu'].apply(lambda x:abs(x))
trdnew['yingyuabs'] = trdnew['yingyu'].apply(lambda x:abs(x))


trdnew1 = pd.merge(trdtime,trdnew,on='id',how='outer')
#trdnew[['day_first','day_last']].fillna(60)


#算比例
trdnew1['jycishu'] = jycishu['jycishu']
trdnew1['untimerate'] = trdnew1['untimecount']/trdnew1['jycishu']
trdnew1['largezhichurate'] = trdnew1['zhichulargecount']/trdnew1['jycishu']
trdnew1['largeshoururate'] = trdnew1['shourulargecount']/trdnew1['jycishu']
trdnew1['largecha'] = trdnew1['largeshoururate']-trdnew1['largezhichurate']
trdnew1['largechaabs'] = trdnew1['largecha'].apply(lambda x:abs(x))
trdnew1['largerate'] = (trdnew1['shourulargecount']+trdnew1['zhichulargecount'])/trdnew1['jycishu']
trdnew1['zhichurate'] = trdnew1['zhichucount']/(trdnew1['zhichucount']+trdnew1['shourucount'])


#tag2 = pd.merge(tag1,trdnew1,on='id',how='left')
#tag3 = tag2[['id','fwcishu','page_no','Count','zhichu','shouru','yingyu','zhichucount','shourucount','zhichulargecount','shourulargecount','Dat_Flg3_Cd','Trx_Cod1_Cd']].fillna(0)
#tagnew = pd.merge(tag,tag3,on='id',how='left')
tag2 = pd.merge(tag1,trdnew1,on='id',how='left')


jyqueshi = []
for j in tag2['jycishu']:
    if np.isnan(j) == True:
        jyqueshi.append('T')
    else:
        jyqueshi.append('F')  
tag2['jyqueshi'] = jyqueshi 



tagnew = tag2.drop([
 'deg_cd'],axis=1,inplace=False)

traindf = tagnew.replace('\\N', np.NAN)
#traindf = traindf.fillna(0)

#traindf.describe()
#traindf.drop(index=(traindf.loc[(traindf['acdm_deg_cd']==0)].index),inplace=True)

traindf['job_year'] = traindf['job_year'].astype(float)#因为nan是float
traindf['job_age'] = traindf['age'] -traindf['job_year']#剔除过于奇怪的数值，例如工龄99年的这种，
sorted(traindf['job_age'])#排序看一下，可能十一二三开始工作的id都是不靠谱的
traindf.drop(index=(traindf.loc[(traindf['job_age']<=14)].index),inplace=True)#找出这部分id来，可以拉黑。
traindf = traindf.drop('job_age', axis=1)
traindf.isnull().sum()

#看某一变量为0的里面 flag为0的比例
#temp = traindf[traindf['zhichu'] == 0]
#(len(temp) - sum(temp['flag'])) / len(temp)

beht = pd.read_csv('评分数据集_beh_b.csv',encoding='gbk',engine='python')
tagt = pd.read_csv('评分数据集_tag_b.csv',encoding='gbk',engine='python')
trdt = pd.read_csv('评分数据集_trd_b.csv',encoding='gbk',engine='python')

        
beht = beht.sort_values('id',axis = 0,ascending = True)
fwcishut = pd.DataFrame.from_dict(Counter(list(beht['id'])), orient='index',columns=['fwcishu'])
fwcishut = fwcishut.reset_index().rename(columns={'index':'id'})
countt = beht.groupby(['id', 'page_no'])["id"].count().reset_index(name="Count")
maxpaget = countt.sort_values('Count', ascending=False).groupby('id', as_index=False).first()

beht = beht.drop('page_tm',axis=1)
beht.rename(columns={'Unnamed: 2':'page_tm'}, inplace = True)

#每个用户的最晚登陆时间
beht_lasttime = beht.groupby('id').max()
beht_lasttime['page_tm'] = beht_lasttime['page_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
beht_lasttime['page_tm'] = pd.to_datetime(beht_lasttime['page_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
maxt_time = max(beht_lasttime['page_tm'])#最大时间
beht_lasttime['day_diff_last'] = (maxt_time - beht_lasttime['page_tm'])#两个相减得到timedelta的时间差
beht_lasttime['day_diff_last'] = (beht_lasttime['day_diff_last'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
beht_lasttime = beht_lasttime.drop(['page_no','page_tm'],axis=1)
#每个用户的最早登陆时间
beht_firsttime = beht.groupby('id').min()
beht_firsttime['page_tm'] = beht_firsttime['page_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
beht_firsttime['page_tm'] = pd.to_datetime(beht_firsttime['page_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
mint_time = min(beht_firsttime['page_tm'])#最大时间
beht_firsttime['day_diff_first'] = (beht_firsttime['page_tm'] - mint_time)#两个相减得到timedelta的时间差
beht_firsttime['day_diff_first'] = (beht_firsttime['day_diff_first'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
beht_firsttime = beht_firsttime.drop(['page_no','page_tm'],axis=1)
behtimet = pd.merge(beht_firsttime,beht_lasttime,on='id',how='outer')

beht['hour'] = beht['page_tm'].apply(lambda x:x[11:13])
beh_untimet = beht[(beht['hour'] >= late_time) | (beht['hour'] <= early_time)]
#sum(beh_untime['flag'])/len(beh_untime)#不良率0.185
#sum(beh['flag'])/len(beh)#不良率0.157
beh_untime_countt = beh_untimet.groupby('id').count()
beh_untime_countt = beh_untime_countt.drop(['page_no','hour'],axis=1).rename(columns={'page_tm':'untimecountapp'})#删除多余列
beh_untime_countt = pd.merge(beh_untime_countt,fwcishut,on='id',how='outer')
beh_untime_countt['untimerate1'] = beh_untime_countt['untimecountapp']/beh_untime_countt['fwcishu']
beh_untime_countt = beh_untime_countt.drop(['fwcishu'],axis=1,inplace=False)


behnewt = pd.merge(fwcishut,maxpaget.drop(['Count'],axis=1,inplace=False),on='id',how='outer')
behnewt = pd.merge(behnewt,behtimet,on='id',how='outer')
behnewt = pd.merge(behnewt,beh_untime_countt,on='id',how='outer')
behnewt = behnewt.fillna(0)
tag1t = pd.merge(tagt,behnewt,on='id',how='left')


fwqueshit = []
for i in tag1t['fwcishu']:
    if np.isnan(i) == True:
        fwqueshit.append('T')
    else:
        fwqueshit.append('F')  
tag1t['fwqueshi'] = fwqueshit 



trdt = trdt.sort_values('id',axis = 0,ascending = True)
jycishut = pd.DataFrame.from_dict(Counter(list(trdt['id'])), orient='index',columns=['jycishu'])
jycishut = jycishut.reset_index().rename(columns={'index':'id'})
trdlarget = trdt[abs(trdt['cny_trx_amt'])>=1000]
#trdcountt = trdt[abs(10<=trdt['cny_trx_amt'])<1000]
trdsmallt = trdt[abs(trdt['cny_trx_amt'])<10]
countbct = trdt.groupby(['id','Dat_Flg1_Cd'])["id"].count().reset_index(name="bcCount")
countbclarget = trdlarget.groupby(['id','Dat_Flg1_Cd'])["id"].count().reset_index(name="bclargeCount")
countbcsmallt = trdsmallt.groupby(['id','Dat_Flg1_Cd'])["id"].count().reset_index(name="bcsmallCount")
countzhichut = countbct[countbct['Dat_Flg1_Cd']=='B'].rename(columns={'bcCount':'zhichucount'})
countshourut = countbct[countbct['Dat_Flg1_Cd']=='C'].rename(columns={'bcCount':'shourucount'})
countlargezhichut = countbclarget[countbclarget['Dat_Flg1_Cd']=='B'].rename(columns={'bclargeCount':'zhichulargecount'})
countlargeshourut = countbclarget[countbclarget['Dat_Flg1_Cd']=='C'].rename(columns={'bclargeCount':'shourulargecount'})
countsmallzhichut = countbcsmallt[countbcsmallt['Dat_Flg1_Cd']=='B'].rename(columns={'bcsmallCount':'zhichusmallcount'})
countsmallshourut = countbcsmallt[countbcsmallt['Dat_Flg1_Cd']=='C'].rename(columns={'bcsmallCount':'shourusmallcount'})
zhichut = trdt[trdt['Dat_Flg1_Cd']=='B']
zhichujinet = pd.DataFrame(zhichut ['cny_trx_amt'].groupby(zhichut['id']).sum())
zhichujinet = zhichujinet.reset_index().rename(columns={'index':'id','cny_trx_amt':'zhichu'})
shourut = trdt[trdt['Dat_Flg1_Cd']=='C']
shourujinet = pd.DataFrame(shourut['cny_trx_amt'].groupby(shourut['id']).sum())
shourujinet = shourujinet.reset_index().rename(columns={'index':'id','cny_trx_amt':'shouru'})
yingyut = pd.merge(zhichujinet,shourujinet,on='id',how='outer')
yingyut = yingyut.fillna(0)
yingyut['yingyu']=yingyut['zhichu']+yingyut['shouru']
countfangshit = trdt.groupby(['id','Dat_Flg3_Cd'])["id"].count().reset_index(name="fangshiCount")
maxfangshit = countfangshit.sort_values('fangshiCount', ascending=False).groupby('id', as_index=False).first()
countfenleit = trdt.groupby(['id','Trx_Cod1_Cd'])["id"].count().reset_index(name="fenleiCount")
maxfenleit = countfenleit.sort_values('fenleiCount', ascending=False).groupby('id', as_index=False).first()
 
trdt['hour'] = trdt['trx_tm'].apply(lambda x:x[11:13])
trd_untimet = trdt[(trdt['hour'] >= late_time) | (trd['hour'] <= early_time)]
trd_untime_countt = trd_untimet.groupby('id').count()
trd_untime_countt = trd_untime_countt.drop(['Dat_Flg1_Cd','Dat_Flg3_Cd','cny_trx_amt','Trx_Cod1_Cd','Trx_Cod2_Cd','hour'],axis=1).rename(columns={'trx_tm':'untimecount'})#删除多余列

#每个用户的最晚登陆时间
trdt_lasttime = trdt.groupby('id').max()
trdt_lasttime['trx_tm'] = trdt_lasttime['trx_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
trdt_lasttime['trx_tm'] = pd.to_datetime(trdt_lasttime['trx_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
max_time1t = max(trdt_lasttime['trx_tm'])#最大时间
trdt_lasttime['day_last'] = (max_time1t - trdt_lasttime['trx_tm'])#两个相减得到timedelta的时间差
trdt_lasttime['day_last'] = (trdt_lasttime['day_last'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
trdt_lasttime = trdt_lasttime.drop(['Dat_Flg1_Cd','Dat_Flg3_Cd','cny_trx_amt','Trx_Cod1_Cd','Trx_Cod2_Cd','hour','trx_tm'],axis=1)
#每个用户的最早登陆时间
trdt_firsttime = trdt.groupby('id').min()
trdt_firsttime['trx_tm'] = trdt_firsttime['trx_tm'].apply(lambda x:x[0:10])#只取到天级别的日期
trdt_firsttime['trx_tm'] = pd.to_datetime(trdt_firsttime['trx_tm'], format='%Y-%m-%d')#把str的时间转换成datetime格式的
min_time1t = min(trdt_firsttime['trx_tm'])#最大时间
trdt_firsttime['day_first'] = (trdt_firsttime['trx_tm'] - min_time1t)#两个相减得到timedelta的时间差
trdt_firsttime['day_first'] = (trdt_firsttime['day_first'] / np.timedelta64(1, 'D')).astype(int)#timedelta转化为int
trdt_firsttime = trdt_firsttime.drop(['Dat_Flg1_Cd','Dat_Flg3_Cd','cny_trx_amt','Trx_Cod1_Cd','Trx_Cod2_Cd','hour','trx_tm'],axis=1)
trdtimet = pd.merge(trdt_firsttime,trdt_lasttime,on='id',how='outer')


   
trd1t = pd.merge(countzhichut.drop(['Dat_Flg1_Cd'],axis=1),countshourut.drop(['Dat_Flg1_Cd'],axis=1),on='id',how='outer')
trd2t = pd.merge(countlargezhichut.drop(['Dat_Flg1_Cd'],axis=1),countlargeshourut.drop(['Dat_Flg1_Cd'],axis=1),on='id',how='outer')
trd3t = pd.merge(countsmallzhichut.drop(['Dat_Flg1_Cd'],axis=1),countsmallshourut.drop(['Dat_Flg1_Cd'],axis=1),on='id',how='outer')
trd4t = pd.merge(maxfangshit.drop(['fangshiCount'],axis=1),maxfenleit.drop(['fenleiCount'],axis=1),on='id',how='outer')   
trd5t = pd.merge(yingyut,trd_untime_countt,on='id',how='outer')
trdnewt = pd.merge(trd1t,trd2t,on='id',how='outer')
trdnewt = pd.merge(trd3t,trdnewt,on='id',how='outer')
trdnewt = pd.merge(trd4t,trdnewt,on='id',how='outer')
trdnewt = pd.merge(trd5t,trdnewt,on='id',how='outer')
trdnewt = trdnewt.fillna(0)

#去相反数，加正反列
zhengfut = []
for i in trdnewt['yingyu']:
    if i>= 0:
        zhengfut.append('A')
    else:
        zhengfut.append('B')        
trdnewt['zhengfu'] = zhengfut
trdnewt['zhichu'] = trdnewt['zhichu'].apply(lambda x:abs(x))
trdnewt['yingyuabs'] = trdnewt['yingyu'].apply(lambda x:abs(x))



trdnewt1 = pd.merge(trdtimet,trdnewt,on='id',how='outer')

#算比例
trdnewt1['jycishu'] = jycishut['jycishu']
trdnewt1['untimerate'] = trdnewt1['untimecount']/trdnewt1['jycishu']
trdnewt1['largezhichurate'] = trdnewt1['zhichulargecount']/trdnewt1['jycishu']
trdnewt1['largeshoururate'] = trdnewt1['shourulargecount']/trdnewt1['jycishu']
trdnewt1['largecha'] = trdnewt1['largeshoururate']-trdnewt1['largezhichurate']
trdnewt1['largechaabs'] = trdnewt1['largecha'].apply(lambda x:abs(x))
trdnewt1['largerate'] = (trdnewt1['shourulargecount']+trdnewt1['zhichulargecount'])/trdnewt1['jycishu']
trdnewt1['zhichurate'] = trdnewt1['zhichucount']/(trdnewt1['zhichucount']+trdnewt1['shourucount'])



tag2t = pd.merge(tag1t,trdnewt1,on='id',how='left')
#tag2t = pd.merge(tag1t,trdnewt1,on='id',how='left')
#tag3t = tag2t[['id','fwcishu','page_no','Count','zhichu','shouru','yingyu','zhichucount','shourucount','zhichulargecount','shourulargecount','Dat_Flg3_Cd','Trx_Cod1_Cd']].fillna(0)
#tagnewt = pd.merge(tagt,tag3t,on='id',how='left')

jyqueshit = []
for j in tag2t['jycishu']:
    if np.isnan(j) == True:
        jyqueshit.append('T')
    else:
        jyqueshit.append('F')  
tag2t['jyqueshi'] = jyqueshit 


tagnewt = tag2t.drop([
 'deg_cd'],axis=1,inplace=False)
testdf = tagnewt.replace('\\N', np.NAN)
testdf.isnull().sum()
#
traindf['Trx_Cod1_Cd'] = traindf['Trx_Cod1_Cd'].replace(1, 'A')
traindf['Trx_Cod1_Cd'] = traindf['Trx_Cod1_Cd'].replace(2, 'B')
traindf['Trx_Cod1_Cd'] = traindf['Trx_Cod1_Cd'].replace(3, 'C')
testdf['Trx_Cod1_Cd'] = testdf['Trx_Cod1_Cd'].replace(1, 'A')
testdf['Trx_Cod1_Cd'] = testdf['Trx_Cod1_Cd'].replace(2, 'B')
testdf['Trx_Cod1_Cd'] = testdf['Trx_Cod1_Cd'].replace(3, 'C')
#

import seaborn as sns
import matplotlib.pyplot as plt
import graphviz  
from sklearn import preprocessing,model_selection
import itertools
import xgboost as xgb
import warnings
import random
warnings.filterwarnings("ignore")

#把评分集没有的page类型替换
a = list(traindf['page_no'].unique())
b = list(testdf['page_no'].unique())
duoyu = []
for i in a :
    if i not in b:
        duoyu.append(i)  

for i in duoyu:
    traindf['page_no'] = traindf['page_no'].replace(i, np.nan)
    #traindf.drop(index=(traindf.loc[(traindf['page_no']==i)].index),inplace=True)

duoyu1 = []
for i in b :
    if i not in a:
        duoyu1.append(i)  

for i in duoyu1:
    testdf['page_no'] = testdf['page_no'].replace(i, np.nan)

a1 = list(traindf['edu_deg_cd'].unique())
b1 = list(testdf['edu_deg_cd'].unique())
duoyu3 = []
for i in a1 :
    if i not in b1:
        duoyu3.append(i)  

for i in duoyu3:
    traindf['edu_deg_cd'] = traindf['edu_deg_cd'].replace(i, np.nan)



traindf['page_no'].describe()
testdf['page_no'].describe()

traindf['edu_deg_cd'].describe()
testdf['edu_deg_cd'].describe()


traindf = traindf.sort_values('id',axis = 0,ascending = True)
train1 = pd.get_dummies(traindf[['Trx_Cod1_Cd','fwqueshi','jyqueshi','edu_deg_cd','zhengfu','gdr_cd','mrg_situ_cd','acdm_deg_cd','Dat_Flg3_Cd','page_no']])
testdf = testdf.sort_values('id',axis = 0,ascending = True)
test1 = pd.get_dummies(testdf[['Trx_Cod1_Cd','fwqueshi','jyqueshi','edu_deg_cd','zhengfu','gdr_cd','mrg_situ_cd','acdm_deg_cd','Dat_Flg3_Cd','page_no']])

traindf1= traindf.drop(['Trx_Cod1_Cd','fwqueshi','jyqueshi','edu_deg_cd',
 'zhengfu',
 'gdr_cd',
 'mrg_situ_cd',
 'acdm_deg_cd',
 'Dat_Flg3_Cd',
 'page_no'],axis=1,inplace=False)

testdf1= testdf.drop(['Trx_Cod1_Cd','fwqueshi','jyqueshi','edu_deg_cd',
 'zhengfu',
 'gdr_cd',
 'mrg_situ_cd',
 'acdm_deg_cd',
 'Dat_Flg3_Cd',
 'page_no'],axis=1,inplace=False)

train = traindf1.join(train1)
test = testdf1.join(test1)

train_labels = train['flag']
# Align the training and testing data, keep only columns present in both dataframes
train, test = train.align(test, join = 'inner', axis = 1)
# Add the target back in
train['flag'] = train_labels


# Find correlations with the target and sort
correlations = train.corr()['flag'].sort_values()
# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

trainx= train.drop(['id','flag'],axis=1,inplace=False)
trainy = train['flag']
test= test.sort_values('id',axis = 0,ascending = True)
testx= test.drop(['id'],axis=1,inplace=False)

train_ids, test_ids = train['id'],test['id']


trainx[:] = trainx[:].astype(float)
testx[:] = testx[:].astype(float)
trainy[:] = trainy[:].astype(float)

#from bayes_opt import BayesianOptimization
#from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,cross_val_score
#from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
#from sklearn.preprocessing import MinMaxScaler
#import contextlib
#import gc 


#def scale_data(X, scaler=None):
#    if not scaler:
#        scaler = MinMaxScaler(feature_range=(-1, 1))
#        scaler.fit(X)
#    X = scaler.transform(X)
#    return X
#
#trainx_scaled= scale_data(trainx)
#testx_scaled= scale_data(testx)

#def XGB_CV(eta,
#          max_depth,
#          gamma,
#          subsample,
#          colsample_bytree
#         ):
#
#    global AUCbest
#    global ITERbest
#
#    paramt = {
#              'booster' : 'gbtree',
#              'max_depth' : int(max_depth),
#              'gamma' : gamma,
#              'eta' : eta,
#              'objective' : 'binary:logistic',
#              'nthread' : 4,
#              'silent' : 0,
#              'eval_metric': 'auc',
#              'subsample' : max(min(subsample, 1), 0),
#              'colsample_bytree' : max(min(colsample_bytree, 1), 0),
#              'seed' : 1001
#              }
#
#    folds = 5
#    cv_score = 0
#
#    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )
#    log_file.flush()
#
#    xgbc = xgb.cv(
#                    paramt,
#                    dtrain,
#                    num_boost_round = 1000,
#                    stratified = True,
#                    nfold = folds,
#                    early_stopping_rounds = 30,
#                    metrics = 'auc',
#                    show_stdv = True
#               )
#
#    val_score = xgbc['test-auc-mean'].iloc[-1]
#    train_score = xgbc['train-auc-mean'].iloc[-1]
#    print(' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' % ( len(xgbc), train_score, val_score, (train_score - val_score), (train_score*2-1),
#(val_score*2-1)) )
#    if ( val_score > AUCbest ):
#        AUCbest = val_score
#        ITERbest = len(xgbc)
#
#    return (val_score*2) - 1
#
#
#log_file = open('Porto-AUC-5fold-XGB-run-01-v1-full.log', 'a')
#AUCbest = -1.
#ITERbest = 0
## Load data set and target values
#n_train = trainx.shape[0]
#train_test = pd.concat((trainx, testx)).reset_index(drop=True)
#
#
##train_test_scaled, scaler = scale_data(train_test)
##trainx1 = train_test_scaled[:n_train, :]
##testx1 = train_test_scaled[n_train:, :]
#print('\n Shape of processed train data:', trainx.shape)
#print(' Shape of processed test data:', testx.shape)
#
#dtrain = xgb.DMatrix(trainx, label = trainy)
#
#XGB_BO = BayesianOptimization(XGB_CV, {
#                                     'eta':(0.01,0.1),
#                                     'max_depth':(2, 8),
#                                     'gamma': (0.001, 3),
#                                     'subsample': (0.5, 1.0),
#                                     'colsample_bytree' :(0.5, 1.0)
#                                    })
#
#XGB_BO.maximize()
#
#index = []
#for i in XGB_BO.res:
#    index.append(i['target'])
#max_index = index.index(max(index))

def xgbCV(eta,max_depth,sub_sample,colsample_bytree):
    #trainy = traindf['flag'] # label for training data
    #trainx = traindf.drop(['flag','id'],axis=1,inplace=False) # feature for training data
    #testx = testdf.drop(['id'],axis=1,inplace=False) # feature for testing data
    skf = model_selection.StratifiedKFold(n_splits=5,random_state=1000) # stratified sampling
    train_performance ={} 
    val_performance={}
    for each_param in itertools.product(eta,max_depth,sub_sample,colsample_bytree): # iterative over each combination in parameter space
        xgb_params = {
                    'eta':each_param[0],
                    'max_depth':each_param[1],
                    'sub_sample':each_param[2],
                    'colsample_bytree':each_param[3],
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }
        best_iteration =[]
        best_score=[]
        training_score=[]
        for train_ind,val_ind in skf.split(trainx,trainy): # five fold stratified cross validation
            X_train,X_val = trainx.iloc[train_ind,],trainx.iloc[val_ind,] # train X and train y
            y_train,y_val = trainy.iloc[train_ind],trainy.iloc[val_ind] # validation X and validation y
            dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns) # convert into DMatrix (xgb library data structure)
            dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns) # convert into DMatrix (xgb library data structure)
            model = xgb.train(xgb_params,dtrain,num_boost_round=1000, 
                              evals=[(dtrain,'train'),(dval,'val')],verbose_eval=False,early_stopping_rounds=30) # train the model
            best_iteration.append(model.attributes()['best_iteration']) # best iteration regarding AUC in valid set
            best_score.append(model.attributes()['best_score']) # best score regarding AUC in valid set
            training_score.append(model.attributes()['best_msg'].split()[1][10:]) # best score regarding AUC in training set
        valid_mean = (np.asarray(best_score).astype(np.float).mean()) # mean AUC in valid set
        train_mean = (np.asarray(training_score).astype(np.float).mean()) # mean AUC in training set
        val_performance[each_param] =  train_mean
        train_performance[each_param] =  valid_mean
        print ("Parameters are {}. Training performance is {:.4f}. Validation performance is {:.4f}".format(each_param,train_mean,valid_mean))
    return (train_performance,val_performance)
#xgbCV(eta=[0.02,0.03,0.04],max_depth=[6,7,8],sub_sample=[0.8],colsample_bytree=[0.5]) 
#xgbCV(eta=[0.02],max_depth=[6],sub_sample=[0.8],colsample_bytree=[0.5]) 没改之前
xgbCV(eta=[0.03],max_depth=[6],sub_sample=[0.8],colsample_bytree=[0.5]) #0.7510


any(trainx.columns == testx.columns)
train_ = xgb.DMatrix(trainx,trainy,feature_names=trainx.columns)
test_ = xgb.DMatrix(testx,feature_names=testx.columns)
xgb_params = {
                    'eta':0.03,
                    'max_depth':6,
                    'sub_sample':0.8,
                    'colsample_bytree':0.5,
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }

final_model = xgb.train(xgb_params,train_,num_boost_round=1000)
y = final_model.predict(test_)


SUB = pd.DataFrame({'Id':test.id.values,'Probability':y})
SUB.to_csv('./最后一次.txt',sep='\t',index=False,header=None)
