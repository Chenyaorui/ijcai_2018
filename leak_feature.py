import gc
import numpy as np
import  pandas as pd
import dask.dataframe as dd


''''
    leak 特征，是否第一次点击，是否最后一次，
    是否在某个小时第一次，最后一次点击
    是否第一次，最后一次点击某品类等的组合
    时间间隔等
'''


def gen_is_first_hour(train_data):
    train_data_2 = train_data.sort_values(by=["user_id", "context_timestamp"], ascending=True)
    first = train_data_2.drop_duplicates(["user_id", "hour", ])
    first['is_first_hour'] = 1
    first = first[["user_id", "hour", "context_timestamp", "is_first_hour"]]
    train_data = dd.merge(train_data, first, how="left", on=["user_id", "hour", "context_timestamp"])
    train_data = train_data.fillna({"is_first_hour": 0})
    first = first.rename(columns={"context_timestamp": "is_first_time_gap_hour"})[
        ["user_id", "hour", "is_first_time_gap_hour"]]
    train_data = dd.merge(train_data, first, on=["user_id", "hour"], how="left")
    train_data["is_first_time_gap_hour"] = (
    train_data["is_first_time_gap_hour"] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_first_time_gap_hour"] = train_data["is_first_time_gap_hour"].astype(np.int32)
    train_data['is_first_hour'] = train_data['is_first_hour'].astype(np.int32)
    return train_data


def gen_is_last_hour(train_data):
    train_data_2 = train_data.sort_values(by=["user_id", "context_timestamp"], ascending=False)
    last = train_data_2.drop_duplicates(["user_id", "hour"])
    last['is_last_hour'] = 1
    last = last[["user_id", "hour", "context_timestamp", "is_last_hour"]]
    train_data = dd.merge(train_data, last, how="left", on=["user_id", "hour", "context_timestamp"])
    train_data = train_data.fillna({"is_last_hour": 0})
    last = last.rename(columns={"context_timestamp": "is_last_time_gap_hour"})[["user_id", "is_last_time_gap_hour"]]
    train_data = dd.merge(train_data, last, on=["user_id"], how="left")
    train_data["is_last_time_gap_hour"] = (
    train_data["is_last_time_gap_hour"] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_last_time_gap_hour"] = train_data["is_last_time_gap_hour"].astype(np.int32)
    train_data['is_last_hour'] = train_data['is_last_hour'].astype(np.int32)
    del train_data_2, last
    return train_data


def gen_is_first_feat_hour(train_data, feat):
    train_data_2 = train_data.sort_values(by=["user_id", feat, "context_timestamp"], ascending=True)
    first = train_data_2.drop_duplicates(["user_id", "hour", feat])
    first['is_first_hour_' + feat] = 1
    first = first[["user_id", "hour", feat, "context_timestamp", 'is_first_hour_' + feat]]
    train_data = dd.merge(train_data, first, how="left", on=["user_id", feat, "hour", "context_timestamp"])
    train_data = train_data.fillna({'is_first_hour_' + feat: 0})
    first = first.rename(columns={"context_timestamp": "is_first_time_gap_hour_" + feat})[
        ["user_id", feat, "hour", "is_first_time_gap_hour_" + feat]]
    train_data = dd.merge(train_data, first, on=["user_id", "hour", feat], how="left")
    train_data["is_first_time_gap_hour_" + feat] = (
    train_data["is_first_time_gap_hour_" + feat] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_first_time_gap_hour_" + feat] = train_data["is_first_time_gap_hour_" + feat].astype(np.int32)
    train_data['is_first_hour_' + feat] = train_data['is_first_hour_' + feat].astype(np.int32)
    del train_data_2, first
    return train_data


def gen_is_last_feat_hour(train_data, feat):
    train_data_2 = train_data.sort_values(by=["user_id", feat, "context_timestamp"], ascending=False)
    last = train_data_2.drop_duplicates(["user_id", "hour", feat])
    last['is_last_hour_' + feat] = 1
    last = last[["user_id", "hour", feat, "context_timestamp", 'is_last_hour_' + feat]]
    train_data = dd.merge(train_data, last, how="left", on=["user_id", feat, "hour", "context_timestamp"])
    train_data = train_data.fillna({'is_last_hour_' + feat: 0})
    last = last.rename(columns={"context_timestamp": "is_last_time_gap_hour_" + feat})[
        ["user_id", feat, "hour", "is_last_time_gap_hour_" + feat]]
    train_data = dd.merge(train_data, last, on=["user_id", "hour", feat], how="left")
    train_data["is_last_time_gap_hour_" + feat] = (
    train_data["is_last_time_gap_hour_" + feat] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_last_time_gap_hour_" + feat] = train_data["is_last_time_gap_hour_" + feat].astype(np.int32)
    train_data['is_last_hour'] = train_data['is_last_hour'].astype(np.int32)
    del train_data_2, last
    return train_data


def gen_is_first(train_data):
    train_data_2 = train_data.sort_values(by=["user_id", "context_timestamp"], ascending=True)
    first = train_data_2.drop_duplicates(["user_id"])
    first['is_first'] = 1
    first = first[["user_id", "context_timestamp", "is_first"]]
    train_data = dd.merge(train_data, first, how="left", on=["user_id", "context_timestamp"])
    train_data = train_data.fillna({"is_first": 0})
    first = first.rename(columns={"context_timestamp": "is_first_time_gap"})[["user_id", "is_first_time_gap"]]
    train_data = dd.merge(train_data, first, on=["user_id"], how="left")
    train_data["is_first_time_gap"] = (
    train_data["is_first_time_gap"] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_first_time_gap"] = train_data["is_first_time_gap"].astype(np.int32)
    train_data['is_first'] = train_data['is_first'].astype(np.int32)
    del train_data_2, first
    return train_data


def gen_is_last(train_data):
    train_data_2 = train_data.sort_values(by=["user_id", "context_timestamp"], ascending=False)
    last = train_data_2.drop_duplicates(["user_id"])
    last['is_last'] = 1
    last = last[["user_id", "context_timestamp", "is_last"]]
    train_data = dd.merge(train_data, last, how="left", on=["user_id", "context_timestamp"])
    train_data = train_data.fillna({"is_last": 0})
    last = last.rename(columns={"context_timestamp": "is_last_time_gap"})[["user_id", "is_last_time_gap"]]
    train_data = dd.merge(train_data, last, on=["user_id"], how="left")
    train_data["is_last_time_gap"] = (
    train_data["is_last_time_gap"] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_last_time_gap"] = train_data["is_last_time_gap"].astype(np.int32)
    train_data['is_last'] = train_data['is_last'].astype(np.int32)
    del train_data_2, last
    return train_data


def gen_is_first_feat(train_data, feat):
    train_data_2 = train_data.sort_values(by=["user_id", feat, "context_timestamp"], ascending=True)
    first = train_data_2.drop_duplicates(["user_id", feat])
    first['is_first_user_' + feat] = 1
    first = first[["user_id", feat, "context_timestamp", 'is_first_user_' + feat]]
    train_data = dd.merge(train_data, first, how="left", on=["user_id", feat, "context_timestamp"])
    train_data = train_data.fillna({'is_first_user_' + feat: 0})
    first = first.rename(columns={"context_timestamp": "is_first_time_gap_" + feat})[
        ["user_id", feat, "is_first_time_gap_" + feat]]
    train_data = dd.merge(train_data, first, on=["user_id", feat], how="left")
    train_data["is_first_time_gap_" + feat] = (
    train_data["is_first_time_gap_" + feat] - train_data["context_timestamp"]).dt.total_seconds()

    train_data["is_first_time_gap_" + feat] = train_data["is_first_time_gap_" + feat].astype(np.int32)
    train_data['is_first_user_' + feat] = train_data['is_first_user_' + feat].astype(np.int32)
    del train_data_2, first
    return train_data


def gen_is_last_feat(train_data, feat):
    train_data_2 = train_data.sort_values(by=["user_id", "context_timestamp"], ascending=False)
    last = train_data_2.drop_duplicates(["user_id", feat])
    last['is_last_user_' + feat] = 1
    last = last[["user_id", feat, "context_timestamp", 'is_last_user_' + feat]]
    train_data = dd.merge(train_data, last, how="left", on=["user_id", feat, "context_timestamp"])
    train_data = train_data.fillna({'is_last_user_' + feat: 0})
    last = last.rename(columns={"context_timestamp": "is_last_time_gap_" + feat})[
        ["user_id", feat, "is_last_time_gap_" + feat]]
    train_data = dd.merge(train_data, last, on=["user_id", feat], how="left")
    train_data["is_last_time_gap_" + feat] = (
    train_data["is_last_time_gap_" + feat] - train_data["context_timestamp"]).dt.total_seconds()
    train_data["is_last_time_gap_" + feat] = train_data["is_last_time_gap_" + feat].astype(np.int32)
    train_data['is_last_user_' + feat] = train_data['is_last_user_' + feat].astype(np.int32)
    return train_data