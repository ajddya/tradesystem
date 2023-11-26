import streamlit as st
import pandas as pd

# ____________________________________________________________分類型の関数作成#___________________________________________________________________________________________________________________________________________________________________________________________________________
# 分類する関数
def classify_action_type(personal, sell_log, buy_reason_ratios, sell_reason_ratios, trade_value, wield_grades):

    classify_type = pd.DataFrame(columns=['分類型'], index=['保守型', 'リサーチ主導型', '積極型', '感情主導型', 'テクニカル'])
    
    Conservative  = 0     #保守型
    Research      = 0     #リサーチ主導型
    Positive      = 0     #積極型
    Emotion       = 0     #感情主導型
    Technical     = 0     #テクニカル型
    
    #個人の性格情報から分類型にポイントを与える
    max_character_list = personal[personal['性格']==personal['性格'].max()].index.values
    min_character_list = personal[personal['性格']==personal['性格'].min()].index.values

    character_count = len(max_character_list) * len(min_character_list)

    for character_max in max_character_list:
        categorize_temp = st.session_state.categorize[st.session_state.categorize['max']==character_max]

        for character_min in min_character_list:
            categorize_temp_temp = categorize_temp[categorize_temp['min']==character_min]
            if not categorize_temp_temp.empty:
                categorize_index = categorize_temp_temp.index.values[0]

                Conservative += st.session_state.categorize["保守型"][categorize_index] * 2
                Research += st.session_state.categorize["リサーチ主導型"][categorize_index]
                Positive += st.session_state.categorize["積極型"][categorize_index]
                Emotion += st.session_state.categorize["感情主導型"][categorize_index]
                Technical += st.session_state.categorize["テクニカル"][categorize_index] * 2

    if not character_count == 0:
        Conservative /= character_count
        Research /= character_count
        Positive /= character_count
        Emotion /= character_count
        Technical /= character_count
    
    
    #取引回数のデータから分類型にポイントを与える
    trade_count = len(sell_log)

    high_line = 100
    low_line = 50

    # LEVEL = 1
    # レベルごとに取引回数の多いライン、少ないラインを定義する
    if st.session_state.level_id == "LEVEL_1":
        high_line /= 12
        low_line /= 12
    elif st.session_state.level_id == "LEVEL_2":
        high_line /= 4
        low_line /= 4
    elif st.session_state.level_id == "LEVEL_3":
        high_line /=2
        low_line /= 2
    elif st.session_state.level_id == "LEVEL_4":
        high_line /= 1
        low_line /= 1


    if trade_count >= high_line :
        Positive += 1 * 2
        Technical += 1
    elif trade_count >= low_line :
        Emotion += 1
    else:
        Conservative += 1
        Research += 1 * 2
        
        
    # 投資根拠のデータから分類型にポイントを与える
    # 各分類型の購入根拠
    buy_reason_Conservative = ["業績が安定している", "リスクが小さい", "配当目当て"]
    buy_reason_Research = ["利回りがいい", "財務データ"]
    buy_reason_Positive = ["全体的な景気"]
    buy_reason_Emotion = ["直感"]
    buy_reason_Technical = ["チャート形状", "過去の経験から"]
    # 各分類型の売却根拠
    sell_reason_Conservative = ["利益確定売り"]
    sell_reason_Research = []
    sell_reason_Positive = ["全体的な景気"]
    sell_reason_Emotion = ["チャート形状", "直感"]
    sell_reason_Technical = ["チャート形状", "過去の経験から"]
    

    for buy_reason in buy_reason_ratios.index.values:
        if buy_reason in buy_reason_Conservative:
            Conservative += (buy_reason_ratios[buy_reason] / 2)
        if buy_reason in buy_reason_Research:
            Research += (buy_reason_ratios[buy_reason] )
        if buy_reason in buy_reason_Positive:
            Positive += (buy_reason_ratios[buy_reason] / 2)
        if buy_reason in buy_reason_Emotion:
            Emotion += (buy_reason_ratios[buy_reason])
        if buy_reason in buy_reason_Technical:
            Technical += (buy_reason_ratios[buy_reason] * 1.5)

    for sell_reason in sell_reason_ratios.index.values:
        if sell_reason in sell_reason_Conservative:
            Conservative += (sell_reason_ratios[sell_reason] / 2)
        if sell_reason in sell_reason_Research:
            Research += (sell_reason_ratios[sell_reason])
        if sell_reason in sell_reason_Positive:
            Positive += (sell_reason_ratios[sell_reason] / 2)
        if sell_reason in sell_reason_Emotion:
            Emotion += (sell_reason_ratios[sell_reason])        
        if sell_reason in sell_reason_Technical:
            Technical += (sell_reason_ratios[sell_reason] * 1.5)
            
    # 運用成績・取引量から分類型にポイントを与える
    high_trade_value = 500000
    high_scattered = 100000
    low_benefit_line = 100000

    # 購入金額の平均が50万以上なら積極型のポイント＋１
    if trade_value['平均値'][0] >= high_trade_value:
        Positive += 1 * 2

    if wield_grades['標準偏差'][0] >= high_scattered:
        # 安定していない
        Emotion += 1 * 2
    else:
        # 安定している
        if wield_grades['平均値'][0] > 0:
            if wield_grades['平均値'][0] < low_benefit_line:
                Conservative += 1 * 2
            else:
                Research += 1
                
    #____________________________________________________          
    classify_type['分類型']['保守型'] = Conservative
    classify_type['分類型']['リサーチ主導型'] = Research
    classify_type['分類型']['積極型'] = Positive
    classify_type['分類型']['感情主導型'] = Emotion
    classify_type['分類型']['テクニカル'] = Technical

    return classify_type    
