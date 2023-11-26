import streamlit as st
import pandas as pd
from back.VOL import VOL_cal, VOL_all

def change_page(num, name=None):
    if name:
        st.session_state.selected_company = name

    st.session_state.page_id = f"page{num}"


def buy(name, rdf_all):

    #保有株式数と１株あたりの株価の初期値
    possess_KK_num = 0
    possess_KK_avg = 0
    benefit = 0
    
    #最新のrdfの株価を取得
    now_data_KK = rdf_all['Close'][st.session_state.now]
    
    #購入金額を計算
    purchace_amount = now_data_KK * st.session_state.buy_num
    
    if purchace_amount > st.session_state.possess_money:
        colored_text = "<span style='font-size:30px'><span style='color:red'> 買付余力が足りません </span> </span> "
        st.markdown(colored_text, unsafe_allow_html=True)

            
    else:
        #選択した企業名が保有株式の中にあるならその数値を取り出す
        if name in st.session_state.possess_KK_df['企業名'].values:
            possess_KK_num = st.session_state.possess_KK_df[st.session_state.possess_KK_df['企業名']==name]['保有株式数'].values[0]
            possess_KK_avg = st.session_state.possess_KK_df[st.session_state.possess_KK_df['企業名']==name]['1株あたりの株価'].values[0]
        
        #1株あたりの株価を算出
        possess_KK_num_one = possess_KK_num / 100
        buy_num_one = st.session_state.buy_num / 100
        possess_KK_avg = (possess_KK_num_one * possess_KK_avg + now_data_KK * buy_num_one) / (possess_KK_num_one + buy_num_one)
        
        #保有株式数を追加
        possess_KK_num += st.session_state.buy_num   
        #この銘柄の合計金額を変数に格納
        # possess_KK = possess_KK_avg * possess_KK_num
    
        benefit = (now_data_KK - possess_KK_avg) * possess_KK_num
    
        #保有株式のデータベース作成
        if name in st.session_state.possess_KK_df['企業名'].values:
            st.session_state.possess_KK_df['1株あたりの株価'] = st.session_state.possess_KK_df['1株あたりの株価'].mask(st.session_state.possess_KK_df['企業名']==name,[possess_KK_avg])
            st.session_state.possess_KK_df['保有株式数'] = st.session_state.possess_KK_df['保有株式数'].mask(st.session_state.possess_KK_df['企業名']==name,[possess_KK_num])
            # st.session_state.possess_KK_df['現在の株価'] = [now_data_KK]
            # st.session_state.possess_KK_df['利益'] = [benefit]
        else:
            possess_KK_df_temp = pd.DataFrame(columns=['企業名', '保有株式数', '現在の株価', '1株あたりの株価', '利益',])
            possess_KK_df_temp['企業名'] = [name]
            possess_KK_df_temp['保有株式数'] = [possess_KK_num]
            possess_KK_df_temp['現在の株価'] = [now_data_KK]
            possess_KK_df_temp['1株あたりの株価'] = [round(possess_KK_avg,1)]
            possess_KK_df_temp['利益'] = [benefit]
            st.session_state.possess_KK_df = pd.concat([st.session_state.possess_KK_df,possess_KK_df_temp],ignore_index=True)

        buy_now_str = st.session_state.now.strftime('%Y/%m/%d')
            
        #データログにデータを追加
        buy_log_temp = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '購入株式数', '購入金額', '購入金額比率', '当日のボラティリティ', '当日のボラティリティ平均', '属性'])
        buy_log_temp['企業名'] = [name]
        buy_log_temp['年月'] = [buy_now_str]
        buy_log_temp['購入根拠'] = [st.session_state.Rationale_for_purchase]
        buy_log_temp['購入株式数'] = [st.session_state.buy_num ]
        buy_amount = now_data_KK * st.session_state.buy_num
        buy_log_temp['購入金額'] = [buy_amount]
        buy_amount_rate = buy_amount / st.session_state.possess_money
        buy_log_temp['購入金額比率'] = [round(buy_amount_rate, 2)]
        buy_log_temp['当日のボラティリティ'] = [VOL_cal(name, st.session_state.now)]
        buy_log_temp['当日のボラティリティ平均'] = [VOL_all(st.session_state.now) ]
        buy_log_temp['属性'] = ['買い']
        st.session_state.buy_log = pd.concat([st.session_state.buy_log,buy_log_temp],ignore_index=True)
    
        st.session_state.possess_money -= purchace_amount

        change_page(2)

def sell(name, rdf_all):

    sell_num = st.session_state.sell_num
    
    #最新のrdfの株価を取得
    now_data_KK = rdf_all['Close'][st.session_state.now]
    
    possess_KK_num = 0
    if name in st.session_state.possess_KK_df['企業名'].values:
        possess_KK_num = st.session_state.possess_KK_df[st.session_state.possess_KK_df['企業名']==name]['保有株式数'].values[0]
        possess_KK_avg = st.session_state.possess_KK_df[st.session_state.possess_KK_df['企業名']==name]['1株あたりの株価'].values[0]
    
    #保有株があるなら、評価損益を計算して利益を表示する
    #エラー分の表示
    if possess_KK_num == 0:
        colored_text2 = "<span style='font-size:30px'><span style='color:red'> あなたは株を持っていません </span> </span> "
        st.markdown(colored_text2, unsafe_allow_html=True)
        # raise Exception('あなたは株を持っていません！')
    else:
        if possess_KK_num < st.session_state.sell_num:
            colored_text = "<span style='font-size:30px'><span style='color:red'> 売却数に対して保有株式数が足りません </span> </span> "
            st.markdown(colored_text, unsafe_allow_html=True)
            # st.write("売却数に対して保有株式数が足りません")
        else:
            #損益を計算し格納
            benefit = (now_data_KK - possess_KK_avg)*st.session_state.sell_num
            
            #保有株式、保有株式数を変更
            # possess_KK -= possess_KK_avg * 100
            possess_KK_num -= st.session_state.sell_num
            
            #保有株式の株価と株式数を更新
            st.session_state.possess_KK_df['1株あたりの株価'] = st.session_state.possess_KK_df['1株あたりの株価'].mask(st.session_state.possess_KK_df['企業名']==name,[possess_KK_avg])
            st.session_state.possess_KK_df['保有株式数'] = st.session_state.possess_KK_df['保有株式数'].mask(st.session_state.possess_KK_df['企業名']==name,[possess_KK_num])
            
            st.session_state.possess_KK_df = st.session_state.possess_KK_df[st.session_state.possess_KK_df['保有株式数']!=0]
            st.session_state.possess_KK_df = st.session_state.possess_KK_df.reset_index(drop=True)
            
            sell_now_str = st.session_state.now.strftime('%Y/%m/%d')
                    
            sell_log_temp = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '売却株式数', '利益','属性'])
            sell_log_temp['企業名'] = [name]
            sell_log_temp['年月'] = [sell_now_str]
            sell_log_temp['売却根拠'] = [st.session_state.basis_for_sale]
            sell_log_temp['売却株式数'] = [st.session_state.sell_num]
            sell_log_temp['利益'] = [benefit]
            sell_log_temp['属性'] = ['売り']
            st.session_state.sell_log = pd.concat([st.session_state.sell_log,sell_log_temp],ignore_index=True)

                
            st.session_state.possess_money += now_data_KK * st.session_state.sell_num

            change_page(2)

