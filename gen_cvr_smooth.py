#coding:utf-8
import pandas as pd

# 生成7号当天的转换率等统计特征
def gen_7_cvr(feat):
    res=pd.DataFrame()
    day_7_cvr = seven_data[[feat,"hour","is_trade"]]
    for hour in range(0,12):
        count=day_7_cvr.groupby([feat],sort = False).apply(lambda x: x['is_trade'][(x['hour']<hour).values].count()).reset_index(name='day_7_'+feat+'_all')
        count1=day_7_cvr.groupby([feat],sort = False).apply(lambda x: x['is_trade'][(x['hour']<hour).values].sum()).reset_index(name='day_7_'+feat+'_1')
        count['day_7_'+feat+'_1']=count1['day_7_'+feat+'_1']
        count.fillna(value=0, inplace=True)
        count['hour']= hour
        res=res.append(count,ignore_index=True)
        store = pd.HDFStore("store_v7.h5")
        store[feat+'_7_cvr'] = res
        store.close()
    return res

store = pd.HDFStore("store_v_final.h5")
seven_data = store["seven_data_v3"]
store.close()

price_level_7_cvr = gen_7_cvr("item_price_level")
sales_level_7_cvr = gen_7_cvr("item_sales_level")
user_id_7_cvr = gen_7_cvr("user_id")
item_cate_2_7_cvr = gen_7_cvr("item_cate_2")
item_cate_3_7_cvr = gen_7_cvr("item_cate_3")
item_brand_id_7_cvr = gen_7_cvr("item_brand_id")
item_city_id_7_cvr = gen_7_cvr("item_city_id")
shop_id_7_cvr = gen_7_cvr("shop_id")
item_id_7_cvr = gen_7_cvr("item_id")

#生成历史统计特征
# pass