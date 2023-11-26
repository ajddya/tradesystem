import streamlit as st
import pandas as pd
import datetime as dt



# 各取引のアドバイス生成
def some_trade_advice(buy_log, sell_log):
    trade_advice_df = pd.DataFrame(columns=['企業名', '指摘事項', '指摘銘柄数'])
    trade_advice_df_temp = pd.DataFrame(columns=['企業名', '指摘事項', '指摘銘柄数'])
    
    #損失回避性
    loss_df  = sell_log[sell_log['売却根拠'] == '利益確定売り']

    if loss_df.empty == False:
        trade_advice_df_temp['企業名'] = [loss_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['損失回避性']
        trade_advice_df_temp['指摘銘柄数'] = [len(loss_df)]
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
    
    #アンカリング効果　buy_logから投資根拠が「安いと思ったから」のものを持ってくる
    anc_df  = buy_log[buy_log['購入根拠'] == '安いと思ったから']

    if anc_df.empty == False:
        trade_advice_df_temp['企業名'] = [anc_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['アンカリング効果']
        trade_advice_df_temp['指摘銘柄数'] = [len(anc_df)]
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
    
    #感応度逓減性　ブレークイーブン効果などと一緒に表示させる
    # sensory_df1 = buy_log[buy_log["購入金額"] == buy_log["購入金額"].max()]
    # sensory_df2 = buy_log[buy_log["購入金額"] == buy_log["購入金額"].min()]
    # sensory_df = pd.concat([sensory_df1,sensory_df2],ignore_index=True)
    
    #現在志向バイアス　売った後1ヶ月以内に最大値があるなら指摘
    pluss_benef_df = sell_log[sell_log['利益'] > 0]
    pluss_benef_df = pluss_benef_df.reset_index(drop=True)

    present_oriented_df = pd.DataFrame()
    for i in range(0, len(pluss_benef_df)):
        pbd_sell_day = pluss_benef_df.iloc[i]['年月']
        pbd_sell_day = dt.datetime.strptime(pbd_sell_day, "%Y/%m/%d")
        end_date = pbd_sell_day + dt.timedelta(days=30)
        pbd_com_name = pluss_benef_df.iloc[i]['企業名']
        index = st.session_state.c_master.loc[(st.session_state.c_master['企業名']==pbd_com_name)].index.values[0]
        after_sell_day_KK = st.session_state.loaded_companies[index].rdf_all[pbd_sell_day : end_date]

        #sell_dayの株価を取得
        sell_day_KK = after_sell_day_KK['Close'].iloc[0]

        #sell_day後の最大closeの値を取得
        max_close_KK = after_sell_day_KK[after_sell_day_KK['Close']==after_sell_day_KK['Close'].max()]['Close'].iloc[0]

        if sell_day_KK < max_close_KK:
            if (max_close_KK - sell_day_KK) > (pluss_benef_df.iloc[i]['利益']/pluss_benef_df.iloc[i]['売却株式数']):
            # 1つの行をDataFrameとして連結する
                temp_df = pd.DataFrame(pluss_benef_df.iloc[i]).transpose()
                present_oriented_df = pd.concat([present_oriented_df, temp_df], ignore_index=True)
        
    if present_oriented_df.empty == False:
        trade_advice_df_temp['企業名'] = [present_oriented_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['現在志向バイアス']
        trade_advice_df_temp['指摘銘柄数'] = [len(present_oriented_df)]
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
    
    #ブレークイーブン効果　購入日の株価より小さな株価が売却びまで連続で続いている場合に指摘
    minas_benef_df = sell_log[sell_log['利益'] < 0]
    minas_benef_df = minas_benef_df.reset_index(drop=True)

    minas_seqence_df = pd.DataFrame()
    for i in range(0,len(minas_benef_df)):
        mbd_sell_day = minas_benef_df.iloc[i]['年月']
        mbd_sell_day = dt.datetime.strptime(mbd_sell_day, "%Y/%m/%d")
        mbd_com_name = minas_benef_df.iloc[i]['企業名']
        mbd_buy_day = buy_log[buy_log['企業名']==mbd_com_name]['年月'].iloc[-1]
        mbd_buy_day = dt.datetime.strptime(mbd_buy_day, "%Y/%m/%d")

        index = st.session_state.c_master.loc[(st.session_state.c_master['企業名']==mbd_com_name)].index.values[0]
        in_trade_rdf = st.session_state.loaded_companies[index].rdf_all[mbd_buy_day : mbd_sell_day]

        buy_day_KK = in_trade_rdf['Close'].iloc[0]

        # インデックスを日付型としてリセット
        in_trade_rdf = in_trade_rdf.reset_index()

        # Closeのカラムでbuy_day_KKよりも小さい値の場所をTrue、それ以外をFalseとする新しいカラムを作成
        in_trade_rdf['Below_Buy_Day_KK'] = in_trade_rdf['Close'] < buy_day_KK

        # 連続したTrueの数をカウントするための関数
        def count_true_consecutive(s):
            count = max_count = 0
            for v in s:
                if v:
                    count += 1
                    max_count = max(max_count, count)
                else:
                    count = 0
            return max_count

        # 連続してbuy_day_KKよりも小さい値が30日以上続く場所を見つける
        if count_true_consecutive(in_trade_rdf['Below_Buy_Day_KK']) >= 30:
            minas_seqence_df_temp = pd.DataFrame(minas_benef_df.iloc[i]).transpose()
            minas_seqence_df = pd.concat([minas_seqence_df, minas_seqence_df_temp], ignore_index = True)

    if minas_seqence_df.empty == False:
        # print('ブレークイーブン効果')
        # print('感応度逓減性')
        trade_advice_df_temp['企業名'] = [minas_seqence_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['ブレークイーブン効果']
        trade_advice_df_temp['指摘銘柄数'] = [len(minas_seqence_df)]
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
        
    return trade_advice_df
