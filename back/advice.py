import streamlit as st
import pandas as pd
import datetime as dt



#投資行動に関するアドバイス生成
def advice(buy_reason_ratios, buy_log, sell_log):
    advice_df = pd.DataFrame(columns=['指摘事項']) 
    advice_df_temp = pd.DataFrame(columns=['指摘事項'])

    #確証バイアス 投資根拠が70%以上同じものなら指摘する
    comf_bias = buy_reason_ratios[buy_reason_ratios > 0.7]
    if not comf_bias.empty:
        # st.write("確証バイアス")
        advice_df_temp['指摘事項'] = ['確証バイアス']
        advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)
    
    #ハロー効果　購入根拠で一番多いのがチャート形状なら指摘
    if "チャート形状" in buy_reason_ratios:
        if buy_reason_ratios.idxmax() == "チャート形状":
            # st.write("ハロー効果")
            advice_df_temp['指摘事項'] = ['ハロー効果']
            advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)

    #自信過剰　主観による判断が50%以上なら指摘
    assertive1, assertive2 = 0, 0
    if "直感" in buy_reason_ratios:
        assertive1 = buy_reason_ratios["直感"]
        
    if "経験から" in buy_reason_ratios:
        assertive2 = buy_reason_ratios["経験から"]
        
    if assertive1 + assertive2 > 0.5:
        # st.write("自信過剰")
        advice_df_temp['指摘事項'] = ['自信過剰']
        advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)

    
    #権威への服従効果　購入根拠でアナリストによる評価が50%以上なら指摘
    if "アナリストによる評価" in buy_reason_ratios:
        specific_category_ratio = buy_reason_ratios["アナリストによる評価"]
        if specific_category_ratio > 0.5:
            # st.write("権威への服従効果")
            advice_df_temp['指摘事項'] = ['権威への服従効果']
            advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)
            
            
    #テンションリダクション　利益最大の銘柄の売却後に２つ以上の銘柄を購入しているときに指摘
    sell_day = sell_log[sell_log["利益"]==sell_log["利益"].max()]["年月"].iloc[0]

    sell_day = dt.datetime.strptime(sell_day, "%Y/%m/%d")

    start_date = sell_day
    end_date = sell_day + pd.Timedelta(days=3)


    buy_log_temp = buy_log.copy()
    for i in range(0, len(buy_log)):
        buy_log_temp['年月'].iloc[i] = dt.datetime.strptime(buy_log_temp['年月'].iloc[i], "%Y/%m/%d")


    df_temp = buy_log_temp[(buy_log_temp['年月'] >= start_date) & (buy_log_temp['年月'] <= end_date)]

    # インデックスをリセット
    df_temp = df_temp.reset_index(drop=True)
    
    if len(df_temp) > 1:
        # st.write("テンションリダクション効果")
        advice_df_temp['指摘事項'] = ['テンションリダクション効果']
        advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)
        
        
    #代表性ヒューリスティクス　１か月以内の取引が全体の70%以上なら指摘
    if not st.session_state.level_id == "LEVEL_1":
        trade_time_df = pd.DataFrame()
        for i in range(0,len(sell_log)):
            c_name = sell_log.iloc[i]['企業名']
            buy_time = buy_log_temp[buy_log_temp['企業名']==c_name]['年月']
            sell_time = sell_log[sell_log['企業名']==c_name]['年月'].iloc[0]
            sell_time = dt.datetime.strptime(sell_time, "%Y/%m/%d")

            delta_time = sell_time - buy_time

            # 31日以内のものだけをフィルタリング
            within_31_days = delta_time[delta_time <= pd.Timedelta(days=31)]

            # 結果をデータフレームに追加
            trade_time_df = pd.concat([trade_time_df,within_31_days], ignore_index=True)

        trade_time_df = trade_time_df.reset_index(drop=True)
        short_trade_rate = len(trade_time_df) / len(sell_log)
        if short_trade_rate > 0.7:
            # st.write("代表性ヒューリスティクス")
            advice_df_temp['指摘事項'] = ['代表性ヒューリスティクス']
            advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)

    return advice_df
