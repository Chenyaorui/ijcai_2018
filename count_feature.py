#统计特征
import gc
import numpy as np
import  pandas as pd
import dask.dataframe as dd
def count_cross_feat(df,feat_1,feat_2):
    cname = feat_1+"_"+feat_2
    add = df.groupby([feat_1,feat_2 ],sort = False).size().reset_index().rename(columns={0: cname})
    df = df.merge(add, 'left', on=[feat_1, feat_2])
    df[cname] = df[cname].astype(np.int32)
    del add
    gc.collect()
    return df

def count_day_feat(df,feat_1):
    cname = feat_1+"_"+'day'
    user_query_day = df.groupby([feat_1,"day" ],sort = False).size().reset_index().rename(columns={0: cname})
    df = df.merge(user_query_day, 'left', on=[feat_1, 'day'])
    df[cname] = df[cname].astype(np.int32)
    return df
def count_hour_feat(df,feat_1):
    cname = feat_1+"_"+'hour'
    user_query_day = df.groupby([feat_1,"hour" ],sort = False).size().reset_index().rename(columns={0: cname})
    df = df.merge( user_query_day, 'left', on=[feat_1, 'hour'])
    df[cname] = df[cname].astype(np.int32)
    return df

def count_cross_feat_hour(df,feat_1,feat_2):
    cname = feat_1+"_"+feat_2+"hour"
    add = df.groupby([feat_1,feat_2,"hour" ],sort = False).size().reset_index().rename(columns={0: cname})
    df = dd.merge(df, add, 'left', on=[feat_1, feat_2,"hour"])
    df[cname] = df[cname].astype(np.int32)
    return df
def count_hour_mean(df,feat_1):
    cname = feat_1+"_"+'hour'
    user_query_day = df.groupby([feat_1,"hour" ]).mean().reset_index().rename(columns={0: cname})
    df = df.merge(user_query_day, how = 'left', on=[feat_1, 'hour'])
    df[cname] = df[cname].astype(np.float32)
    del user_query_day
    return df
def merge_sum(df,columns,value):
    add = pd.DataFrame(df.groupby([columns],sort = False)[value].sum()).reset_index()
    cname = columns+"_"+value+"_sum"
    add.columns=[columns]+[cname]
    df = df.merge(add,on=[columns],how="left")
    df[cname] = df[cname].astype(np.int32)
    return df
def merge_max(df,columns,value):
    add = pd.DataFrame(df.groupby([columns],sort = False)[value].max()).reset_index()
    cname = columns+"_"+value+"_max"
    add.columns=[columns]+[cname]
    df = df.merge(add,on=[columns],how="left")
    df[cname] = df[cname].astype(np.int32)
    return df
def merge_min(df,columns,value):
    add = pd.DataFrame(df.groupby([columns],sort = False)[value].min()).reset_index()
    cname = columns+"_"+value+"_min"
    add.columns=[columns]+[cname]
    df = df.merge(add,on=[columns],how="left")
    df[cname] = df[cname].astype(np.int32)
    return df
def merge_nunique(df,columns,value):
    add = pd.DataFrame(df.groupby([columns],sort = False)[value].nunique()).reset_index()
    cname = columns+"_"+value+"_nunique"
    add.columns=[columns]+[cname]
    df = df.merge(add,on=[columns],how="left")
    df[cname] = df[cname].astype(np.int32)
    return df
def merge_mean(df,columns,value):
    add = pd.DataFrame(df.groupby([columns],sort = False)[value].mean()).reset_index()
    cname = columns+"_"+value+"_mean"
    add.columns=[columns]+[cname]
    df = df.merge(add,on=[columns],how="left")
    df[cname] = df[cname].astype(np.float32)
    return df
def merge_mean_hour(df,columns,value):
    add = pd.DataFrame(df.groupby([columns,"hour"],sort = False)[value].mean()).reset_index()
    cname = columns+"_"+value+"_mean_hour"
    add.columns=[columns,"hour"]+[cname]
    df = df.merge(add,on=[columns,"hour"],how="left")
    return df
def count_cross_feat_day(df,feat_1,feat_2):
    cname = feat_1+"_"+feat_2+"_day"
    add = df.groupby([feat_1,feat_2],sort = False).size().reset_index().rename(columns={0: cname})
    df = dd.merge(df, add, 'left', on=[feat_1, feat_2])
    df[cname] = df[cname].astype(np.int32)
    return df
def count_cross_feat_minute(train_data,feat_1,feat_2):
    cname = feat_1+"_"+feat_2+"_minute"
    add = train_data.groupby([feat_1,feat_2,"minute"],sort = False).size().reset_index().rename(columns={0: cname})
    train_data = pd.merge(train_data, add,how =  'left', on=[feat_1, feat_2,"minute"])

    return train_data