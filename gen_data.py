import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from datetime import datetime
from scipy.stats import mode
import sklearn.externals.joblib as joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import dask.dataframe as dd
from scipy import sparse
import gc
import time

onehot_category = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                   'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_age_level', 'user_occupation_id',
                   'user_star_level', 'context_page_id', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                   ]
continuous_cate = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description',
                   'num_of_category', 'num_of_item_proprety', 'num_of_predict_proprety']

dummy_cate = ['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
astype_list = ["item_id", 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
               'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id',
               'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_page_id', 'shop_id',
               'shop_review_num_level', 'is_trade', 'day', 'num_of_category',
               'item_cate_1', 'item_cate_2', 'num_of_item_proprety', 'num_of_predict_proprety', 'shop_star_level']
astype_list_float = ["shop_review_positive_rate", 'shop_score_service', 'shop_score_service', 'shop_score_description']
def fun_1(x):
    try:
        string_2 = x.split(";")[2]
        return string_2
    except:
        return -1


def count_predict_property(x):
    num = 0
    for i in range(14):
        if type(x[i]) == str and x[i] != "-1":
            proper = x[i].split(":")[1]
            if proper != "-1":
                num += len(proper.split(","))
    return num


def predictHasCateNum(x):
    num = 0
    for i in range(14):
        if type(x[i]) == str and x[i] != "-1":
            proper = x[i].split(":")[0]
            if proper == x["item_cate_1"] or proper == x["item_cate_2"] or proper == x["item_cate_3"]:
                num += 1
    return num


def gen_data(train_data):
    train_data["context_timestamp"] = train_data["context_timestamp"].apply(lambda x: datetime.fromtimestamp(x))

    train_data["day"] = train_data["context_timestamp"].dt.day
    train_data["time"] = train_data["context_timestamp"].dt.time

    train_data["num_of_category"] = train_data["item_category_list"].apply(lambda x: len(x.split(";")))
    #     train_data["num_of_category"] = train_data["item_category_list"].apply(lambda x: len(x.split(";")), axis=1)

    train_data["item_cate_1"] = train_data["item_category_list"].apply(lambda x: x.split(";")[0])
    train_data["item_cate_2"] = train_data["item_category_list"].apply(lambda x: x.split(";")[1])
    train_data["item_cate_3"] = train_data["item_category_list"].apply(
        lambda x: x.split(";")[2] if len(str(x).split(';')) > 2 else '0')

    train_data["num_of_item_proprety"] = train_data["item_property_list"].apply(lambda x: len(x.split(";")))
    train_data["num_of_predict_proprety"] = train_data["predict_category_property"].apply(lambda x: len(x.split(";")))

    #     train_data["has_item_cate_3"] = train_data.apply(lambda x:1 if x["item_cate_3"] != -1 else 0, axis=1 )
    #     df = pd.DataFrame(train_data.apply(lambda x: pd.Series(x["predict_category_property"].split(";")), axis=1))
    print("done num_of_predict_proprety!")
    return train_data


def preprocess(train_data):
    labelEncoder_cate = ["shop_id", "item_brand_id", "item_city_id", "item_id",
                         "user_id", "item_cate_2", "item_cate_1", "item_cate_3", "context_id"]
    # 中值填充连续值特征
    for i in continuous_cate:
        train_data[i] = train_data[i].replace([-1], train_data[i].median())
    # 众数填充类别型特征
    for i in onehot_category:
        train_data[i] = train_data[i].replace([-1], train_data[i].mode()[0])
    leE = LabelEncoder()
    for i in labelEncoder_cate:
        train_data[i] = leE.fit_transform(train_data[i])
        joblib.dump(leE, r"leE_" + i)

    print("done tranfrom!")
    return train_data


chunks = pd.read_table(r"round2_train.txt"
                       , encoding='gb2312', delim_whitespace=True, chunksize=10 ** 8)
chunk_list = []
for chunk in chunks:
    chunk_list.append(chunk)
data = pd.concat(chunk_list)
chunks = pd.read_table(r"round2_ijcai_18_test_a_20180425.txt"
                       , encoding='gb2312', delim_whitespace=True, chunksize=10 ** 8)
chunk_list = []
for chunk in chunks:
    chunk_list.append(chunk)
test_data = pd.concat(chunk_list)
chunks = pd.read_table(r"round2_ijcai_18_test_b_20180510.txt"
                       , encoding='gb2312', delim_whitespace=True, chunksize=10 ** 8)
chunk_list = []
for chunk in chunks:
    chunk_list.append(chunk)
test_b = pd.concat(chunk_list)
test_data["is_trade"] = 0
test_b["is_trade"] = 0
data = pd.concat([data, test_data, test_b], axis=0)
data = gen_data(data)
data = preprocess(data)

for i in astype_list_float:
    data[i] = data[i].astype(np.float32)


for i in astype_list:
    data[i] = data[i].astype(np.int32)

store = pd.HDFStore("store_v2.h5")
store["online_data"] = data[10432036:]
store["seven_data"] = data[:10432036][data[:10432036].day == 7]
store["test_b"] = test_b
store.close()