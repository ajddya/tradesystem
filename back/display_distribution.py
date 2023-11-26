import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# 取引量、利益の基本統計量を作成
def display_distribution(data):
    # データがスカラーの場合、リストに変換
    if np.isscalar(data):
        data = [data]
    
    # 基本統計データの計算
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    # mode = np.argmax(np.bincount(data))
    mode = stats.mode(data).mode[0]

    stats_dict = {
    "平均値": [mean],
    "中央値": [median],
    "最頻値": [mode],
    "分散": [variance],
    "標準偏差": [std_dev],
    "最小値": [min_val],
    "最大値": [max_val],
    }
    
    # 3. データフレームに格納する
    stats_df = pd.DataFrame(stats_dict)

    return stats_df

# 購入・売却根拠の統計量、ヒストグラム作成
def display_distribution2(data):
     # 頻度を計算
    value_counts = data.value_counts()

    # 最頻値 (Mode)
    mode = value_counts.idxmax()
    mode_freq = value_counts.max()
    
    value_ratios = data.value_counts(normalize=True)

    # ユニークなカテゴリの数
    num_unique_categories = data.nunique()

    return value_counts, value_ratios