import gc
import numpy as np
import  pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelEncoder
import sklearn.externals.joblib as joblib
from sklearn.metrics import log_loss
from count_feature import *
from leak_feature import  *

drop_cate = ['instance_id' ,'context_id','context_timestamp',"item_category_list","item_property_list","predict_category_property",
             'day', 'time',"user_id", "is_trade"]

#读取贝叶斯平滑后的统计值文件：点击数，购买数，平滑转化率
df_shop_id = pd.read_csv("shop_id_smooth.csv",header = 0,sep = ",")
df_item_id = pd.read_csv("item_id_smooth.csv",header = 0,sep = ",")
df_user_id = pd.read_csv("user_id.csv",header = 0,sep = ",")
df_city_id = pd.read_csv("item_city_id_smooth.csv",header = 0,sep = ",")
df_item_id_age = pd.read_csv(r"item_id_user_age_levelsmooth.csv",header = 0,sep = ",")
df_item_cate_1_age = pd.read_csv(r"item_cate_1_user_age_levelsmooth.csv",header = 0,sep = ",")
df_item_id_age = df_item_id_age[df_item_id_age["user_age_level"] != -1]
df_item_cate_1_age = df_item_cate_1_age[df_item_cate_1_age["user_age_level"] != -1]
df_city_id = df_city_id[df_city_id["item_city_id"] != -1]
le_item = joblib.load("leE_item_id")
df_item_id["item_id"] = le_item.transform(df_item_id["item_id"])
le_shop = joblib.load("leE_shop_id")
df_shop_id["shop_id"] = le_shop.transform(df_shop_id["shop_id"])
le_user = joblib.load("leE_user_id")
df_user_id["user_id"] = le_user.transform(df_user_id["user_id"])
le_city = joblib.load("leE_item_city_id")
df_city_id["item_city_id"] = le_city.transform(df_city_id["item_city_id"])
leE_item_cate_2 = joblib.load("leE_item_cate_2")
df_item_cate_1_age["item_cate_1"] = leE_item_cate_2.transform(df_item_cate_1_age["item_cate_1"].astype("str"))

df_item_id_age["item_id"] = le_item.transform(df_item_id_age["item_id"])
df_item_cate_1_age = df_item_cate_1_age.rename(columns = {"item_cate_1":"item_cate_2"})
#对用户的统计值做平滑
df_user_id["user_cvr"] = (1.7+df_user_id["user_id_1"] -1)/(1.7+94+df_user_id["user_id_all"]-2)
df_page_id = pd.read_csv(r"context_page_id_smooth.csv",header = 0,sep = ",")
df_page_id = df_page_id[df_page_id["context_page_id"] != -1]
df_brand_id = pd.read_csv(r"item_brand_id_smooth.csv",header = 0,sep = ",")
le_brand = joblib.load("leE_item_brand_id")
df_brand_id= df_brand_id[df_brand_id["item_brand_id"] != -1]
df_brand_id["item_brand_id"] = le_brand.transform(df_brand_id["item_brand_id"])

df_cate_2_id = pd.read_csv(r"item_cate_1_smooth.csv",header = 0,sep = ",")
# df_cate_2_id = df_cate_2_id.rename(columns = {"item_cate_1":"item_cate_2"})
leE_item_cate_2 = joblib.load("leE_item_cate_2")
df_cate_2_id = df_cate_2_id[df_cate_2_id["item_cate_2"] != -1]
df_cate_2_id["item_cate_2"] = leE_item_cate_2.transform(df_cate_2_id["item_cate_2"].astype("str"))

df_occupation_id = pd.read_csv(r"user_occupation_id_smooth.csv",header = 0,sep = ",")
df_occupation_id = df_occupation_id[df_occupation_id["user_occupation_id"] != -1]

df_star_id = pd.read_csv(r"user_star_level_smooth.csv",header = 0,sep = ",")
df_star_id = df_star_id[df_star_id["user_star_level"] != -1]

df_gender_id = pd.read_csv(r"user_gender_id_smooth.csv",header = 0,sep = ",")
df_gender_id= df_gender_id[df_gender_id["user_gender_id"] != -1]

df_age_id = pd.read_csv(r"user_age_level_smooth.csv",header = 0,sep = ",")
df_age_id= df_age_id[df_age_id["user_age_level"] != -1]
df_shop_level_id = pd.read_csv(r"shop_review_num_level_smooth.csv",header = 0,sep = ",")
df_shop_level_id = df_shop_level_id[df_shop_level_id["shop_review_num_level"] != -1]

# import warnings
# warnings.filterwarnings("ignore")
#使用 h5读取文件，读写效果比较高
store = pd.HDFStore("store_v2.h5")
seven_data = store["seven_data"]
online_data = store["online_data"]
seven_data  = pd.concat([seven_data,online_data])
store.close()
# seven_data = seven_data.sample(frac = 0.01)
del  online_data
gc.collect()
seven_data["hour"] = seven_data["context_timestamp"].dt.hour

seven_data = gen_is_last_hour(seven_data)
# seven_data = gen_is_last_feat_hour(seven_data,"item_id")
seven_data = gen_is_first_hour(seven_data)
seven_data = gen_is_first(seven_data)
seven_data = gen_is_last_feat(seven_data,"item_id")
seven_data = gen_is_first_feat(seven_data,"item_id")
#类目的leak feature 效果很好
seven_data = gen_is_first_feat(seven_data,"item_cate_2")
seven_data = gen_is_last_feat(seven_data,"item_cate_2")
seven_data = gen_is_first_feat(seven_data,"item_cate_3")
seven_data = gen_is_last_feat(seven_data,"item_cate_3")
# 将predict_category_property 拆分开，分析第一次和最后一次点击的的类目属性，效果不错
seven_data["p_item_pro_list"] = seven_data["predict_category_property"].str.split(";")
for i in range(2):
    leE = LabelEncoder()
    seven_data["p_item_pro_"+str(i)] = leE.fit_transform(seven_data["p_item_pro_list"].apply(lambda x:x[i] if len(x) >i else "0"))
    joblib.dump(leE,r"leE_"+"p_item_pro_"+str(i))
    seven_data["p_item_pro_"+str(i)] = seven_data["p_item_pro_"+str(i)].astype(np.int32)
    seven_data = count_hour_feat(seven_data,"p_item_pro_"+str(i))
    seven_data = gen_is_first_feat(seven_data,"p_item_pro_"+str(i))
    seven_data = gen_is_last_feat(seven_data,"p_item_pro_"+str(i))



seven_data = count_hour_feat(seven_data,"item_pro_1")
seven_data = gen_is_first_feat(seven_data,"item_pro_1")
seven_data = gen_is_last_feat(seven_data,"item_pro_1")
#拼接历史统计特征，包括点击数，购买数，平滑转化率
seven_data = seven_data.merge(df_shop_level_id,on =["shop_review_num_level","day"],how = "left")
seven_data = seven_data.merge(df_star_id,on = ["user_star_level","day"],how = "left")
seven_data = seven_data.merge(df_occupation_id,on =["user_occupation_id","day"],how = "left")
seven_data = seven_data.merge(df_gender_id,on =["user_gender_id","day"],how = "left")
seven_data = seven_data.merge(df_age_id,on = ["user_age_level","day"],how = "left")
seven_data = seven_data.merge(df_item_id,on =["item_id","day"],how = "left")
seven_data = seven_data.merge(df_shop_id,on =["shop_id","day"],how = "left")
seven_data = seven_data.merge(df_user_id,on =["user_id","day"],how = "left")
seven_data = seven_data.merge(df_city_id,on = ["item_city_id","day"],how = "left")
#用均值填充拼接后缺失的值
seven_data["item_id_smooth"] = (seven_data["item_id_smooth"].fillna(df_item_id["item_id_smooth"].mean())).astype(np.float32)
seven_data["item_id_all"] = (seven_data["item_id_all"].fillna(df_item_id["item_id_all"].mean())).astype(np.int32)
seven_data["item_id_1"] = (seven_data["item_id_1"].fillna(df_item_id["item_id_1"].mean())).astype(np.int32)

seven_data["shop_id_smooth"] = (seven_data["shop_id_smooth"].fillna(df_shop_id["shop_id_smooth"].mean())).astype(np.float32)
seven_data["shop_id_all"] = (seven_data["shop_id_all"].fillna(df_shop_id["shop_id_all"].mean())).astype(np.int32)
seven_data["shop_id_1"] = (seven_data["shop_id_1"].fillna(df_shop_id["shop_id_1"].mean())).astype(np.int32)

seven_data["item_city_id_smooth"] = (seven_data["item_city_id_smooth"].fillna(df_city_id["item_city_id_smooth"].mean())).astype(np.float32)
seven_data["item_city_id_all"] = (seven_data["item_city_id_all"].fillna(df_city_id["item_city_id_all"].mean())).astype(np.int32)
seven_data["item_city_id_1"] = (seven_data["item_city_id_1"].fillna(df_city_id["item_city_id_1"].mean())).astype(np.int32)

seven_data["user_cvr"] = (seven_data["user_cvr"].fillna(df_user_id["user_cvr"].mean())).astype(np.float32)
seven_data["user_id_all"] = (seven_data["user_id_all"].fillna(df_user_id["user_id_all"].mean()))
seven_data["user_id_1"] = (seven_data["user_id_1"].fillna(df_user_id["user_id_1"].mean())).astype(np.int32)

seven_data = seven_data.merge(df_page_id,on = ["context_page_id","day"],how = "left")
seven_data["context_page_id_smooth"] = (seven_data["context_page_id_smooth"].fillna(df_page_id["context_page_id_smooth"].mean())).astype(np.float32)
seven_data["context_page_id_all"] = (seven_data["context_page_id_all"].fillna(df_page_id["context_page_id_all"].mean())).astype(np.int32)
seven_data["context_page_id_1"] = (seven_data["context_page_id_1"].fillna(df_page_id["context_page_id_1"].mean())).astype(np.int32)

seven_data = seven_data.merge(df_brand_id,on = ["item_brand_id","day"],how = "left")
seven_data["item_brand_id_smooth"] = (seven_data["item_brand_id_smooth"].fillna(df_brand_id["item_brand_id_smooth"].mean())).astype(np.float32)
seven_data["item_brand_id_all"] = (seven_data["item_brand_id_all"].fillna(df_brand_id["item_brand_id_all"].mean())).astype(np.int32)
seven_data["item_brand_id_1"] = (seven_data["item_brand_id_1"].fillna(df_brand_id["item_brand_id_1"].mean())).astype(np.int32)

seven_data = seven_data.merge(df_cate_2_id,on = ["item_cate_2","day"],how = "left")
seven_data["item_cate_1_smooth"] = (seven_data["item_cate_1_smooth"].fillna(df_cate_2_id["item_cate_1_smooth"].mean())).astype(np.float32)
seven_data["item_cate_1_all"] = (seven_data["item_cate_1_all"].fillna(df_cate_2_id["item_cate_1_all"].mean())).astype(np.int32)
seven_data["item_cate_1_1"] = (seven_data["item_cate_1_1"].fillna(df_cate_2_id["item_cate_1_1"].mean())).astype(np.int32)

#------------------------------------
# item_all = np.sum(seven_data["item_id_all"])
# item_1 = np.sum(seven_data["item_id_1"])
# seven_data["item_hot_all"] = seven_data["item_id_all"] / item_all
# seven_data["item_hot_1"] = seven_data["item_id_1"] / item_1

#---------------------------冷启动特征，效果不好-----------------------------------------
# d_item = df_item_id[(df_item_id.day ==6)]
# d_item = pd.DataFrame(list(d_item["item_id"].unique()),columns = ["item_id"])
# d_item = joblib.load("d_item.pkl")
# d_item["is_old_item"] = 1
# d_item = d_item.set_index("user_id")
# seven_data = seven_data.join(d_item,on = "user_id")
# seven_data["is_old_item"] = seven_data["is_old_item"].fillna(0)

#item，shop的均值特征，比较有效的是shop和item的各种平滑cvr均值
seven_data = merge_mean(seven_data,"shop_id","item_collected_level")
seven_data = merge_mean(seven_data,"shop_id","item_pv_level")
seven_data = merge_mean(seven_data,"shop_id","item_sales_level")
seven_data = merge_mean(seven_data,"shop_id","item_brand_id_smooth")
seven_data = merge_mean(seven_data,"shop_id","item_city_id_smooth")
seven_data = merge_mean(seven_data,"shop_id","item_cate_1_smooth")
seven_data = merge_mean(seven_data,"shop_id","item_id_smooth")
# seven_data = merge_mean(seven_data,"item_cate_3","hour")
# # seven_data = count_hour_feat(seven_data,"item_brand_id")
seven_data = merge_mean(seven_data,"item_id","user_star_level_smooth")
seven_data = merge_mean(seven_data,"item_brand_id","user_star_level_smooth")
seven_data = merge_mean(seven_data,"shop_id","user_star_level_smooth")
seven_data = merge_mean(seven_data,"item_cate_2","user_star_level_smooth")
seven_data = merge_mean(seven_data,"item_cate_3","user_star_level_smooth")

seven_data = merge_mean(seven_data,"item_id","user_age_level_smooth")
seven_data = merge_mean(seven_data,"item_brand_id","user_age_level_smooth")
seven_data = merge_mean(seven_data,"shop_id","user_age_level_smooth")
seven_data = merge_mean(seven_data,"item_cate_2","user_age_level_smooth")
seven_data = merge_mean(seven_data,"item_cate_3","user_age_level_smooth")

seven_data = merge_mean(seven_data,"item_id","user_gender_id_smooth")
seven_data = merge_mean(seven_data,"item_brand_id","user_gender_id_smooth")
seven_data = merge_mean(seven_data,"shop_id","user_gender_id_smooth")
seven_data = merge_mean(seven_data,"item_cate_2","user_gender_id_smooth")
seven_data = merge_mean(seven_data,"item_cate_3","user_gender_id_smooth")

seven_data = seven_data.drop(["user_gender_id_smooth","user_gender_id_1","user_gender_id_all",
                            "user_age_level_smooth","user_age_level_1","user_age_level_all",
                             "user_star_level_1","user_star_level_all","item_cate_1_1","item_cate_1_all",
                             "item_city_id_all","item_city_id_1"],axis = 1)
# seven_data = merge_max(seven_data,"shop_id","item_sales_level")
seven_data["item_pro_1"] = seven_data["item_property_list"].apply(lambda x: x.split(";")[1] if len(str(x).split(';')) >=2 else '0')
leE = LabelEncoder()
seven_data["item_pro_1"] = leE.fit_transform(seven_data["item_pro_1"])
joblib.dump(leE,r"leE_"+"item_pro_1")
seven_data["item_pro_1"] = seven_data["item_pro_1"].astype(np.int32)

seven_data = count_cross_feat(seven_data,"user_id","shop_id")
seven_data = count_cross_feat(seven_data,"user_id","item_id")
#item，shop等的小时均值，有提升
seven_data = merge_mean(seven_data,"item_id","hour")
# seven_data = merge_mean(seven_data,"item_brand_id","hour")
seven_data = merge_mean(seven_data,"shop_id","hour")
seven_data = merge_mean(seven_data,"item_cate_2","hour")
seven_data = merge_mean(seven_data,"item_cate_3","hour")
seven_data = merge_mean(seven_data,"p_item_pro_0","hour")
seven_data = merge_mean(seven_data,"p_item_pro_1","hour")
#user的均值特征，时间关系没机会做更多尝试
seven_data = merge_mean(seven_data,"user_id","shop_review_num_level_smooth")
seven_data = merge_mean(seven_data,"user_id","item_price_level")
seven_data = count_day_feat(seven_data,"item_id")
seven_data = count_day_feat(seven_data,"shop_id")
seven_data = count_day_feat(seven_data,"user_id")
seven_data = count_day_feat(seven_data,"item_cate_2")
seven_data = count_day_feat(seven_data,"item_brand_id")
seven_data = count_day_feat(seven_data,"item_city_id")
seven_data = count_day_feat(seven_data,"item_cate_3")
# seven_data = count_day_feat(seven_data,"user_occupation_id")

# seven_data = merge_sum(seven_data,"shop_id","item_sales_level")
#当天的点击统计
seven_data = count_hour_feat(seven_data,"item_id")
seven_data = count_hour_feat(seven_data,"shop_id")
seven_data = count_hour_feat(seven_data,"user_id")
seven_data = count_hour_feat(seven_data,"item_brand_id")
seven_data = count_hour_feat(seven_data,"item_cate_2")

# seven_data = gen_is_first_feat(seven_data,"item_brand_id")
# seven_data = gen_is_last_feat(seven_data,"item_brand_id")
# seven_data = gen_is_first_feat(seven_data,"context_page_id")
# seven_data = gen_is_last_feat(seven_data,"context_page_id")
# seven_data = gen_is_first_feat(seven_data,"item_category_list")
# seven_data = gen_is_last_feat(seven_data,"item_category_list")
# seven_data = gen_is_first_feat(seven_data,"item_price_level")
# seven_data = gen_is_last_feat(seven_data,"item_price_level")
# seven_data = gen_is_first_feat(seven_data,"item_sales_level"
# seven_data = gen_is_last_feat(seven_data,"item_sales_level")
# seven_data = gen_is_first_feat(seven_data,"item_collected_level"
# seven_data = gen_is_last_feat(seven_data,"item_collected_level")
#用户与下一次的点击时间差（上一次）
temp = seven_data.sort_values(by = ["user_id","context_timestamp"],ascending = True)
temp["diff_gap"] = -temp["context_timestamp"].diff(-1).dt.total_seconds()/60
temp["diff_gap_before"] = -temp["context_timestamp"].diff(1).dt.total_seconds()/60
df_diff = temp[["instance_id","user_id","context_timestamp","diff_gap","diff_gap_before"]]
seven_data = seven_data.merge(df_diff,on = ["instance_id","user_id","item_id",
                                                   "context_timestamp"],how = "left")

dtrain = seven_data[seven_data.hour < 10]
dtest = seven_data[(seven_data.hour >= 10)&(seven_data.hour <12)]
donline = seven_data[seven_data.hour >= 12]
#抽取按线上比例构造线下的测试集，线上比例约0.036，本地测试集约0.042
dtest_1 = dtest[dtest["is_trade"] == 1].sample(frac = 0.83,random_state = 1024)
dtest = pd.concat([dtest[dtest["is_trade"] == 0],dtest_1])
dtest_y = dtest.is_trade
dtrain_y = dtrain.is_trade
store = pd.HDFStore("store_v_final.h5")
store["seven_data_v3"] = seven_data
store.close()
del seven_data
gc.collect()
#读取当天（7号）的统计特征，包括点击数，购买数，转换率
store = pd.HDFStore("store_v7.h5")
shop_id_7_cvr  = store["shop_id_7_cvr"]
item_id_7_cvr = store["item_id_7_cvr"]
user_id_7_cvr = store["user_id_7_cvr"]
item_cate_2_7_cvr = store["item_cate_2_7_cvr"]
item_cate_3_7_cvr = store["item_cate_3_7_cvr"]
store.close()
#对转换率做平滑
shop_id_7_cvr["shop_cvr_7"] = (1.7+shop_id_7_cvr["day_7_shop_id_1"] -1)/(1.7+94+shop_id_7_cvr["day_7_shop_id_all"]-2).astype(np.float32)
item_id_7_cvr["item_cvr_7"] = (1.7+item_id_7_cvr["day_7_item_id_1"] -1)/(1.7+94+item_id_7_cvr["day_7_item_id_all"]-2).astype(np.float32)
user_id_7_cvr["user_cvr_7"] = (1.7+user_id_7_cvr["day_7_user_id_1"] -1)/(1.7+94+user_id_7_cvr["day_7_user_id_all"]-2).astype(np.float32)
item_cate_2_7_cvr["item_cate_2_cvr_7"] = (1.7+item_cate_2_7_cvr["day_7_item_cate_2_1"] -1)/(1.7+94+item_cate_2_7_cvr["day_7_item_cate_2_all"]-2).astype(np.float32)
item_cate_3_7_cvr["item_cate_3_cvr_7"] = (1.7+item_cate_3_7_cvr["day_7_item_cate_3_1"] -1)/(1.7+94+item_cate_3_7_cvr["day_7_item_cate_3_all"]-2).astype(np.float32)
#线上的当天统计转换率采用11点的转换率和统计特征
shop_id_7_cvr_11 = shop_id_7_cvr[shop_id_7_cvr.hour == 11].drop("hour",axis = 1)
item_id_7_cvr_11 = item_id_7_cvr[item_id_7_cvr.hour == 11].drop("hour",axis = 1)
item_cate_2_7_cvr_11= item_cate_2_7_cvr[item_cate_2_7_cvr.hour == 11].drop("hour",axis = 1)
item_cate_3_7_cvr_11 = item_cate_3_7_cvr[item_cate_3_7_cvr.hour == 11].drop("hour",axis = 1)
user_id_7_cvr_11 = user_id_7_cvr[user_id_7_cvr.hour == 11].drop("hour",axis = 1)
#拼接训练集
dtrain = dtrain.merge(shop_id_7_cvr,on = ["shop_id","hour"],how = "left")
dtrain = dtrain.merge(item_id_7_cvr,on = ["item_id","hour"],how = "left")
dtrain = dtrain.merge(user_id_7_cvr,on = ["user_id","hour"],how = "left")
dtrain = dtrain.merge(item_cate_2_7_cvr,on = ["item_cate_2","hour"],how = "left")
dtrain = dtrain.merge(item_cate_3_7_cvr,on = ["item_cate_3","hour"],how = "left")

dtest = dtest.merge(shop_id_7_cvr,on = ["shop_id","hour"],how = "left")
dtest =dtest.merge(item_id_7_cvr,on = ["item_id","hour"],how = "left")
dtest = dtest.merge(user_id_7_cvr,on = ["user_id","hour"],how = "left")
dtest = dtest.merge(item_cate_2_7_cvr,on = ["item_cate_2","hour"],how = "left")
dtest = dtest.merge(item_cate_3_7_cvr,on = ["item_cate_3","hour"],how = "left")

donline = donline.merge(shop_id_7_cvr_11,on = ["shop_id"],how = "left")
donline = donline.merge(item_id_7_cvr_11,on = ["item_id"],how = "left")
donline = donline.merge(user_id_7_cvr_11,on = ["user_id"],how = "left")
donline = donline.merge(item_cate_2_7_cvr_11,on = ["item_cate_2"],how = "left")
donline = donline.merge(item_cate_3_7_cvr_11,on = ["item_cate_3"],how = "left")

#特征的交叉组合，没时间尝试更多，内存够的可尝试poly 二阶，曾经尝试一次内存爆了
dtrain["pv_uv"] =  (dtrain["item_pv_level"]*dtrain["item_id_day"]).astype(np.int32)
dtest["pv_uv"] =  (dtest["item_pv_level"]*dtest["item_id_day"]).astype(np.int32)
donline["pv_uv"] =  (donline["item_pv_level"]*donline["item_id_day"]).astype(np.int32)
dtrain["price_sale"] =  (dtrain["item_price_level"]*dtrain["item_sales_level"]).astype(np.int32)
dtest["price_sale"] =  (dtest["item_price_level"]*dtest["item_sales_level"]).astype(np.int32)
donline["price_sale"] =  (donline["item_collected_level"]*donline["item_sales_level"]).astype(np.int32)
dtrain["collcet_sale"] = ( dtrain["item_collected_level"]*dtrain["item_sales_level"]).astype(np.int32)
dtest["collcet_sale"] = (dtest["item_collected_level"]*dtest["item_sales_level"]).astype(np.int32)
donline["collcet_sale"] =  (donline["item_collected_level"]*donline["item_sales_level"]).astype(np.int32)

dtrain["pv_uv_sale"] =  (dtrain["item_id_smooth"]*dtrain["item_sales_level"]).astype(np.float32)
dtest["pv_uv_sale"] =  (dtest["item_id_smooth"]*dtest["item_sales_level"]).astype(np.float32)
donline["pv_uv_sale"] =  (donline["item_id_smooth"]*donline["item_sales_level"]).astype(np.float32)
dtrain["shop_cvr_sale"] =  (dtrain["shop_id_smooth"]*dtrain["item_sales_level"]).astype(np.float32)
dtest["shop_cvr_sale"] =  (dtest["shop_id_smooth"]*dtest["item_sales_level"]).astype(np.float32)
donline["shop_cvr_sale"] = (donline["shop_id_smooth"]*donline["item_sales_level"]).astype(np.float32)

dtrain["pv_uv_2"] =  dtrain["item_pv_level"]*dtrain["item_id_1"]
dtest["pv_uv_2"] =  dtest["item_pv_level"]*dtest["item_id_1"]
donline["pv_uv_2"] =  donline["item_pv_level"]*donline["item_id_1"]

dtrain["pv_uv_price"] =  dtrain["item_id_smooth"]*dtrain["item_price_level"]
dtest["pv_uv_price"] =  dtest["item_id_smooth"]*dtest["item_price_level"]
donline["pv_uv_price"] =  donline["item_id_smooth"]*donline["item_price_level"]
#重要性最强，衡量了商品销量和质量的指标
dtrain["shop_cvr_score"] =  (dtrain["item_sales_level"]*dtrain["shop_score_description"]).astype(np.float32)
dtest["shop_cvr_score"] =  (dtest["item_sales_level"]*dtest["shop_score_description"]).astype(np.float32)
donline["shop_cvr_score"] =  (donline["item_sales_level"]*donline["shop_score_description"]).astype(np.float32)

# dtrain["shop_cvr_desc"] =  dtrain["shop_cvr"]*dtrain["shop_score_description"]
# dtest["shop_cvr_desc"] =  dtest["shop_cvr"]*dtest["shop_score_description"]
# donline["shop_cvr_desc"] =  donline["shop_cvr"]*donline["shop_score_description"]

dtrain["shop_score_delivery_cvr"] =  (dtrain["shop_review_num_level"]*dtrain["shop_score_delivery"]).astype(np.float32)
dtest["shop_score_delivery_cvr"] =  (dtest["shop_review_num_level"]*dtest["shop_score_delivery"]).astype(np.float32)
donline["shop_score_delivery_cvr"] =  (donline["shop_review_num_level"]*donline["shop_score_delivery"]).astype(np.float32)

dtrain["user_item_cvr"] =  (dtrain["item_id_smooth"]*dtrain["user_cvr"]).astype(np.float32)
dtest["user_item_cvr"] =  (dtest["item_id_smooth"]*dtest["user_cvr"]).astype(np.float32)
donline["user_item_cvr"] =  (donline["item_id_smooth"]*donline["user_cvr"]).astype(np.float32)
dtrain["user_shop_cvr"] =  (dtrain["shop_id_smooth"]*dtrain["user_cvr"]).astype(np.float32)
dtest["user_shop_cvr"] =  (dtest["shop_id_smooth"]*dtest["user_cvr"]).astype(np.float32)
donline["user_shop_cvr"] =  (donline["shop_id_smooth"]*donline["user_cvr"]).astype(np.float32)

#lgb
import lightgbm as lgb
ltrain = lgb.Dataset(dtrain.drop(drop_cate+["item_id","user_id","shop_id"] ,axis=1), label=dtrain_y)
ltest = lgb.Dataset(dtest.drop(drop_cate+["item_id","user_id","shop_id"], axis=1), label=dtest_y)
#'num_leaves': 2 ** 5,
lgb_params1 = {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss', 'max_depth':8,
             'lambda_l2':50, 'subsample': 0.7,'colsample_bytree': 0.7,'colsample_bylevel': 0.7,'learning_rate': 0.1,
                'tree_method': 'exact','seed': 2017, 'nthread': 15,'silent': True, "scale_pos_weight" :0.7812,}
num_round =500
model = lgb.train(lgb_params1, ltrain, num_boost_round = num_round,verbose_eval=10,valid_sets = [ltest,ltrain],
                  early_stopping_rounds=20)
train_lgb_pred1 = model.predict(dtrain.drop(drop_cate+["item_id","user_id","shop_id"], axis=1))
valid_lgb_pred1 = model.predict(dtest.drop(drop_cate+["item_id","user_id","shop_id"], axis=1))
print(log_loss(dtrain_y,train_lgb_pred1))
print(log_loss(dtest_y,valid_lgb_pred1))

online_lgb_pred1 = model.predict(donline.drop(drop_cate+["item_id","user_id","shop_id"], axis=1))


#线上提交
store = pd.HDFStore("store_v2.h5")
test_data = store["online_data"]
test_b = store["test_b"]
print(test_b["instance_id"].head())
store.close()
test_data["predicted_score"] = online_lgb_pred1
test_data = test_data[['instance_id', 'predicted_score']]
test_b =pd.merge(test_b,test_data,on = "instance_id",how = "left")

test_data_online = test_b[['instance_id', 'predicted_score']]
test_data_online[['instance_id', 'predicted_score']].to_csv('round2_5_14_night_2.txt', index=False, sep=' ')
print(test_data_online.head())
del test_data,test_b
gc.collect()