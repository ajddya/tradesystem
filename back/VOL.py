import streamlit as st
import datetime as dt
import numpy as np

# 当日のボラティリティを計算する
def VOL_cal(c_name, date):
    index = st.session_state.c_master.loc[(st.session_state.c_master['企業名']==c_name)].index.values[0]
    target_company = st.session_state.loaded_companies[index]
    rdf_all = target_company.rdf_all

    # 前日の株価データを取得
    pre_buy_day = date - dt.timedelta(days=1)    
    while pre_buy_day not in rdf_all.index:
        pre_buy_day -= dt.timedelta(days=1)
    
    # 各値の抽出
    High = rdf_all[date:date]["High"].values[0]  # 高値
    Low = rdf_all[date:date]["Low"].values[0]  # 安値
    Close = rdf_all[date:date]["Close"].values[0]  # 終値
    pre_Close = rdf_all[pre_buy_day:pre_buy_day]["Close"].values[0]  # 前日の終値

    # 当日のTR
    A = Close - Low       # 当日の高値 - 当日の安値
    B = High - pre_Close  # 当日の高値 - 前日の終値
    C = pre_Close - Low   # 前日の終値 - 当日の安値

    TR = A
    if B > TR:
        TR = B

    if C > TR:
        TR = C

    # 当日のTP
    TP = (High + Low + Close) / 3

    VOL = (TR / TP) * 100

    return round(VOL,2)

# 全銘柄のボラティリティを計算する
def VOL_all(date):
    VOL_list = []
    for i in range(0, len(st.session_state.c_master)):
        VOL_list.append(VOL_cal(st.session_state.c_master["企業名"][i], date)) 
        
    V_mean = np.mean(VOL_list)

    return round(V_mean,2)

