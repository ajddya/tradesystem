import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import pickle
import os
import shutil
from functools import partial
import japanize_matplotlib
import matplotlib.pyplot as plt
import sqlite3
import uuid

# 各種関数ファイルのインポート
from back.simlation_result import Simulation_Results
from back.Company_Data import CompanyData

from back.make_graph import make_graph
from back.create_chose_companies import create_chose_companies
from back.make_simple_graph import make_simple_graph
from back.get_NKX_data import get_NKX_data
from back.VOL import VOL_cal, VOL_all
from back.buy_sell import buy, sell
from back.display_distribution import display_distribution, display_distribution2
from back.some_trade_advice import some_trade_advice
from back.advice import advice
from back.classify_action_type import classify_action_type
from back.reset import reset
from back.database import save_userdata, insert_data_to_db, insert_survey_to_db

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'

#____________________________初期値を代入する関数________________________________________
 #全体の期間を指定
all_range_start = dt.datetime(2020,9,1)
# all_range_end = dt.datetime(2022,3,31)
# now = dt.datetime(2021,1,1)


#変数の初期値
def init():
    # データ読み込み
    if "c_master" not in st.session_state:
        st.session_state.c_master = pd.read_csv('read_file/company_list3.csv')

    if "categorize" not in st.session_state:
        st.session_state.categorize = pd.read_csv('read_file/categorize.csv')

    if "action_type_advice" not in st.session_state:
        st.session_state.action_type_advice = pd.read_csv('read_file/action_type_advice.csv')

    if "Behavioral_Economics" not in st.session_state:
        st.session_state.Behavioral_Economics = pd.read_csv('read_file/Behavioral_Economics.csv')

    if "overall_advice" not in st.session_state:
        st.session_state.overall_advice = pd.read_csv('read_file/overall_advice.csv')

    if "loaded_companies" not in st.session_state:
        with open("read_file/companies.pkl", "rb") as file:
            st.session_state.loaded_companies = pickle.load(file)

    if "all_range_end" not in st.session_state:
        st.session_state.all_range_end = dt.datetime(2021,2,1)  

    #買い・売りの仮ログデータのデータフレーム作成
    if "buy_log_temp" not in st.session_state:
        st.session_state.buy_log_temp = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '購入株式数', '購入金額', '当日のボラティリティ', '当日のボラティリティ平均', '属性'])

    if "sell_log_temp" not in st.session_state:
        st.session_state.sell_log_temp = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '売却株式数', '利益', '属性'])

    if "benef_temp" not in st.session_state:
        st.session_state.benef_temp = 0
    
    if "sales_df" not in st.session_state:
        st.session_state.sales_df = pd.DataFrame(columns=['売上','営業利益','当期利益','基本的1株当たりの当期利益'],index=['2018','2019','2020','2021'])

    if "CF_df" not in st.session_state:
        st.session_state.CF_df = pd.DataFrame(columns=['営業CF','投資CF','財務CF'],index=['2020','2021'])

    if "FS_df" not in st.session_state:
        st.session_state.FS_df = pd.DataFrame(columns=['2020','2021'],index=['1株当たりの当期純利益','PER','1株当たりの純資産','PBR','ROA','ROE','自己資本比率'])

    if "div_df" not in st.session_state:
        st.session_state.div_df = pd.DataFrame(columns=['2020','2021'],index=['配当性向', '配当利回り'])

    if "div_df2" not in st.session_state:
        st.session_state.div_df2 = pd.DataFrame(columns=['中間','期末'],index=['金額', '配当権利付き最終日', "配当基準日"])

    if "trade_advice_df_temp" not in st.session_state:
        st.session_state.trade_advice_df_temp = pd.DataFrame(columns=['企業名', '指摘事項'])

    if "advice_df_temp" not in st.session_state:
        st.session_state.advice_df_temp = pd.DataFrame(columns=['指摘事項'])

    # シミュレーションの実施回数
    if "num" not in st.session_state:
        st.session_state.num = 1

if "init_executed" not in st.session_state:
    init()  # main関数を実行
    st.session_state.init_executed = True

# 後で値が変わる変数の初期値(データベースから取得するデータ)
def changeable_init():
    # 各変数の初期化
    if "account_created" not in st.session_state:
        st.session_state.account_created = False

    if "load_data_bool" not in st.session_state:
        st.session_state.load_data_bool = False

    if "survey_bool" not in st.session_state:
        st.session_state.survey_bool = False

    if "result_bool" not in st.session_state:
        st.session_state.result_bool = False

    if "possess_money_bool" not in st.session_state:
        st.session_state.possess_money_bool = False
    
    
    if "personal" not in st.session_state:
        st.session_state.personal = pd.DataFrame(columns=['性格'], index=['新規性', '誠実性', '外交性', '協調性', '神経症傾向'])

    if "now" not in st.session_state:
        st.session_state.now = dt.datetime(2021,1,4)

    #乱数から企業名をリストに格納する
    if "chose_companies" not in st.session_state:
        st.session_state.chose_companies = []
    if "chose_companies_name_list" not in st.session_state:
        st.session_state.chose_companies_name_list = []

    #買付余力
    if "possess_money" not in st.session_state:
        st.session_state.possess_money = 1000000

    #初期の買付余力
    if "possess_money_init" not in st.session_state:
        st.session_state.possess_money_init = 1000000

    #所有株式のデータフレーム作成
    if "possess_KK_df" not in st.session_state:
        st.session_state.possess_KK_df = pd.DataFrame(columns=['企業名', '保有株式数', '現在の株価', '1株あたりの株価', '利益'])

    #買い・売りのログデータのデータフレーム作成
    if "buy_log" not in st.session_state:
        st.session_state.buy_log = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '購入株式数', '購入金額', '購入金額比率', '当日のボラティリティ', '当日のボラティリティ平均', '属性'])

    if "sell_log" not in st.session_state:
        st.session_state.sell_log = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '売却株式数', '利益', '属性'])

    if "Dividends_df" not in st.session_state:
        st.session_state.Dividends_df = pd.DataFrame(columns=['企業名', '属性', '配当金', "配当基準日", "実施", "種類"])

    if "trade_advice_df" not in st.session_state:
        st.session_state.trade_advice_df = pd.DataFrame(columns=['企業名', '指摘事項', '指摘銘柄数'])

    if "advice_df" not in st.session_state:
        st.session_state.advice_df = pd.DataFrame(columns=['指摘事項'])  

    # 個人情報のデータ
    if "personal_df" not in st.session_state:
        st.session_state.personal_df = pd.DataFrame(columns=['ユーザID', 'ユーザ名', '年齢', '性別', '投資経験の有無', '投資に関する知識の有無', '開放性', '誠実性', '外交性', '協調性', '神経症傾向'])  

    if "result" not in st.session_state:
        st.session_state.result = [] 

    if "n" not in st.session_state:
        st.session_state.n = 1 

changeable_init()

#_____________________________________________________________________________________________________________________________

#create_chose_companiesの状態を保持
if "create_chose_companies_executed" not in st.session_state: 
    create_chose_companies()
    st.session_state.create_chose_companies_executed = True

if "target_company" not in st.session_state: 
    st.session_state.target_company = st.session_state.chose_companies[0]

#companiesからデータを抽出する
name = st.session_state.target_company.name
rdf = st.session_state.target_company._rdf
rdf_all = st.session_state.target_company.rdf_all

#_______________________________1日、１週間進めるボタン作成_____________________________________________________
#nowを更新する関数
def add_next_day(num):

    next_day = st.session_state.now + dt.timedelta(days=num)

    # next_day が rdf_all に存在するか確認し、存在しない場合は次の日付に移動
    while next_day not in rdf_all.index:
        next_day += dt.timedelta(days=1)
    
    st.session_state.now = next_day

#____________________________________________________________#.dbをダウンロードする#______________________________________________________________________________
# ダウンロード用の関数
def download_db():
    db_path = 'my_database.db'

    if os.path.exists(db_path):
        with open(db_path, "rb") as file:
            st.sidebar.write("以下のボタンをクリックするとデータベースファイルがダウンロードできます。")
            st.sidebar.download_button(
                label="Download DB",
                data=file,
                file_name="my_database.db",
                mime="application/octet-stream"
            )
    else:
        st.error('データベースファイルが見つかりません。')

# my_database と new_database を同期させる
new_db_path = 'new_database.db'
existing_db_path = 'my_database.db'

# 新しいファイルが存在するか確認
if os.path.exists(new_db_path):
    shutil.copyfile(existing_db_path, new_db_path)
else:
    shutil.copyfile(existing_db_path, 'new_database.db')


# システム管理_______________________________________________________________________________________________________________________________________________________________________
def start_sym(n):
    # 初めから始めるボタン
    if n == 1:
        reset()
        if st.session_state.account_created==True:
            st.session_state.possess_money_init = st.session_state.possess_money
            st.session_state.show_page = True
        else:
            st.sidebar.write("アカウント情報を入力してください")

    # 続きから始めるボタン
    if n == 2:
        if st.session_state.account_created==True:
            st.session_state.show_page = True
        else:
            st.sidebar.write("アカウント情報を入力してください")

def change_page(num, name=None):
    if name:
        st.session_state.selected_company = name

    st.session_state.page_id = f"page{num}"

def change_page_to_result(buy_log, sell_log, possess_money):
    st.session_state.buy_log = buy_log
    st.session_state.sell_log = sell_log
    st.session_state.possess_money = possess_money
    st.session_state.show_page = True
    st.session_state.page_id = "page5"

def change_page2(num, buy_log=None, sell_log=None, benef=None, advice_df=None):
    if buy_log is not None and not buy_log.empty:
        st.session_state.buy_log_temp = buy_log
    if sell_log is not None and not sell_log.empty:
        st.session_state.sell_log_temp = sell_log
    if benef:
        st.session_state.benef_temp = benef
    if advice_df is not None and not advice_df.empty:
        st.session_state.advice_df_temp = advice_df

    st.session_state.page_id2 = f"page2_{num}"
#_____________________________________________________________________________________________________________________________

# 全体の期間を超えた場合はループを終了
if st.session_state.now > st.session_state.all_range_end:
    change_page(5)

if st.session_state.now > st.session_state.all_range_end:
    # 現在時刻を終了時間に合わせる
    st.session_state.now = st.session_state.all_range_end

if "selected_company" not in st.session_state:
    st.session_state.selected_company = st.session_state.chose_companies_name_list[0]

#各選択可能銘柄の現在値を更新する
for i in range(0,len(st.session_state.chose_companies)):
    target_company_temp = st.session_state.chose_companies[i]
    # now_KK = target_company_temp.rdf_all['Close'][st.session_state.now]

#各保有銘柄の現在値、利益を更新する
for i in range(0, len(st.session_state.possess_KK_df)):
    possess_KK_df_temp2 = st.session_state.possess_KK_df
    index_temp = st.session_state.c_master[st.session_state.c_master['企業名'] == st.session_state.possess_KK_df['企業名'][i]].index.values[0]
    target_company_temp2 = st.session_state.loaded_companies[index_temp]
    st.session_state.possess_KK_df['現在の株価'][i] = target_company_temp2.rdf_all['Close'][st.session_state.now]
    st.session_state.possess_KK_df['利益'][i] = (st.session_state.possess_KK_df['現在の株価'][i] - st.session_state.possess_KK_df['1株あたりの株価'][i]) * st.session_state.possess_KK_df['保有株式数'][i]

    # 配当関係の処理
    target_company_temp3 = st.session_state.c_master.iloc[index_temp]
    half_right_day = dt.datetime.strptime(target_company_temp3["中間配当権利付き最終日"], "%Y/%m/%d")
    half_base_day = dt.datetime.strptime(target_company_temp3["中間配当基準日"], "%Y/%m/%d")
    right_day = dt.datetime.strptime(target_company_temp3["期末配当権利付き最終日"], "%Y/%m/%d")
    base_day = dt.datetime.strptime(target_company_temp3["期末配当基準日"], "%Y/%m/%d")
    # nowが中間配当権利付き最終日の場合の処理
    if st.session_state.now > half_right_day:
        if (target_company_temp3['企業名'] + '_' + '中間')  not in st.session_state.Dividends_df['種類'].values:
            Dividends_df_temp = pd.DataFrame(columns=['企業名', '属性', '配当金', '配当基準日', '実施', '種類'])
            Dividends_df_temp['企業名'] = [target_company_temp3['企業名']]
            Dividends_df_temp['属性'] = ['中間']
            Dividends_df_temp['配当金'] = [target_company_temp3['中間配当'] * st.session_state.possess_KK_df['保有株式数'][i]]
            Dividends_df_temp['配当基準日'] = [half_base_day]
            Dividends_df_temp['実施'] = [False]
            Dividends_df_temp['種類'] = [target_company_temp3['企業名'] + '_' + '中間']
            st.session_state.Dividends_df = pd.concat([st.session_state.Dividends_df, Dividends_df_temp], ignore_index=True)

    # nowが期末配当権利付き最終日の場合の処理
    if st.session_state.now > right_day:
        if (target_company_temp3['企業名'] + '_' + '期末')  not in st.session_state.Dividends_df['種類'].values:
            Dividends_df_temp = pd.DataFrame(columns=['企業名', '属性', '配当金', '配当基準日', '実施', '種類'])
            Dividends_df_temp['企業名'] = [target_company_temp3['企業名']]
            Dividends_df_temp['属性'] = ['期末']
            Dividends_df_temp['配当金'] = [target_company_temp3['期末配当'] * st.session_state.possess_KK_df['保有株式数'][i]]
            Dividends_df_temp['配当基準日'] = [base_day]
            Dividends_df_temp['実施'] = [False]
            Dividends_df_temp['種類'] = [target_company_temp3['企業名'] + '_' + '期末']
            st.session_state.Dividends_df = pd.concat([st.session_state.Dividends_df, Dividends_df_temp], ignore_index=True)

# 配当基準日になったら配当金を加算する
Dividends_df_temp = st.session_state.Dividends_df[st.session_state.Dividends_df["実施"]==False]
Dividends_df_temp.reset_index(drop=True)
for i in range(0, len(Dividends_df_temp)):
    if Dividends_df_temp['配当基準日'][i] < st.session_state.now:
        st.session_state.possess_money += Dividends_df_temp["配当金"][i]
        st.session_state.Dividends_df.loc[st.session_state.Dividends_df['種類']==Dividends_df_temp['種類'][i], '実施'] = True
        


button_css1 = f"""
    <style>
        div.stButton > button:first-child  {{
        color        : white               ;
        padding      : 14px 20px           ;
        margin       : 8px 0               ;
        width        : 100%                ;
        font-weight  : bold                ;/* 文字：太字                   */
        border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
        border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
        background   : #a9a9a9             ;/* 背景色：薄いグレー            */
    }}
    </style>
    """
st.markdown(button_css1, unsafe_allow_html=True)

#_____________________________________________________________トレード画面_________________________________________________________________________________________________________________________________

if 'show_page' not in st.session_state:
    st.session_state.show_page = False


if st.session_state.show_page:
    # 選択可能銘柄一覧
    def page1():
        st.title("選択可能銘柄一覧")

        col1, col2 = st.columns((5, 5))
        with col1:
            st.button("一日進める", on_click=lambda: add_next_day(1))
        with col2:
            st.button("一週間進める", on_click=lambda: add_next_day(7))

        st.write(f"now = {st.session_state.now}")

        st.write("_______________________________________________________________________________________________________")

        # #選択可能銘柄の企業名一覧を作成
        # st.session_state.selected_company = st.selectbox("銘柄", st.session_state.chose_companies_name_list)

        #選択された企業から企業データを復元
        index = st.session_state.c_master[st.session_state.c_master['企業名'] == '日経平均'].index.values[0]
        st.session_state.target_company = st.session_state.loaded_companies[index]

        #companiesからデータを抽出する
        name = st.session_state.target_company.name
        rdf = st.session_state.target_company._rdf
        rdf_all = st.session_state.target_company.rdf_all

        rdf = rdf_all[all_range_start : st.session_state.now]

        now_KK = rdf_all['Close'][-1]
        pre_KK = rdf_all['Close'][-2]
        now_pre = now_KK - pre_KK

        col_a, col_b, col_c = st.columns((2, 2, 4))
        with col_a:
            st.subheader('日経平均')
            st.button("株価を見る", on_click=partial(change_page, 9))
        with col_b:
            st.metric(label='現在値', value=f'{round(now_KK,1)} 円', delta=f'{round(now_pre,1)} 円', delta_color='inverse')
        with col_c:
            buf = make_simple_graph(name, rdf)
            st.image(buf)

        st.write("_______________________________________________________________________________________________________")


        companies_data = []
        for company in st.session_state.chose_companies:
            rdf_all_temp = company.rdf_all[all_range_start:st.session_state.now]
            now_KK = rdf_all_temp['Close'][-1]
            pre_KK = rdf_all_temp['Close'][-2]
            now_pre = now_KK - pre_KK
            companies_data.append((company.name, now_KK, now_pre, rdf_all_temp))

        # 2. ループの最適化
        for i, (name, now_KK, now_pre, rdf) in enumerate(companies_data):
            col3, col4, col5 = st.columns((2, 2, 4))
            with col3:
                st.subheader(name)
                st.button("株価を見る", key=f"chose_companies_key_{i}", on_click=partial(change_page, 2, name))
            with col4:
                st.metric(label='現在値', value=f'{round(now_KK,1)} 円', delta=f'{round(now_pre,1)} 円', delta_color='inverse')
            with col5:
                buf = make_simple_graph(name, rdf)
                st.image(buf)


        st.write("_______________________________________________________________________________________________________")

        st.button("保有株式へ", on_click=lambda: change_page(3))

    # トレード画面
    def page2():
        st.title("トレード画面")

        st.write("_______________________________________________________________________________________________________")

        st.button("選択可能銘柄一覧へ",on_click = lambda: change_page(1))

        #選択可能銘柄の企業名一覧を作成
        # st.session_state.selected_company = st.selectbox("銘柄", st.session_state.chose_companies_name_list)

        st.write("_______________________________________________________________________________________________________")


        #選択された企業から企業データを復元
        index = st.session_state.c_master[st.session_state.c_master['企業名'] == st.session_state.selected_company].index.values[0]
        st.session_state.target_company = st.session_state.loaded_companies[index]


        st.subheader(f'{st.session_state.target_company.name}の株価')

        #companiesからデータを抽出する
        name = st.session_state.target_company.name
        rdf = st.session_state.target_company._rdf
        rdf_all = st.session_state.target_company.rdf_all

        rdf = rdf_all[all_range_start : st.session_state.now]

        #グラフ表示
        # rdf = add_next_available_day(rdf)
        make_graph(name, rdf)

        col3, col4 = st.columns((5, 5))
        with col3:
            action = st.button("買う", on_click=lambda: change_page(6))
        with col4:
            action = st.button("売る", on_click=lambda: change_page(7))

        st.write("_______________________________________________________________________________________________________")

        st.button("企業情報を見る",on_click = lambda: change_page(4))

        st.write("_______________________________________________________________________________________________________")
        if st.session_state.possess_KK_df.empty == True:
            st.write("あなたは現在株を所有していません。") 
        else:
            st.write("現在保有している株式") 
            st.dataframe(st.session_state.possess_KK_df)


    # 保有株式画面
    def page3():
        st.title("保有株式")

        if st.session_state.possess_KK_df.empty == True:
            st.write("あなたは現在株を所有していません。") 
        else:
            st.write("現在保有している株式") 
            st.dataframe(st.session_state.possess_KK_df)

        st.subheader(f"買付余力：{round(st.session_state.possess_money)}")

        st.write("_______________________________________________________________________________________________________")

        if not st.session_state.possess_KK_df.empty:
            companies_data_2_temp = []
            for com_name in st.session_state.possess_KK_df["企業名"]:
                c_master_index = st.session_state.c_master[st.session_state.c_master["企業名"]==com_name].index.values[0]
                companies_data_2_temp.append(st.session_state.loaded_companies[c_master_index])

            companies_data_2 = []
            for company in companies_data_2_temp:
                rdf_all_temp = company.rdf_all[all_range_start:st.session_state.now]
                now_KK = rdf_all_temp['Close'][-1]
                pre_KK = rdf_all_temp['Close'][-2]
                now_pre = now_KK - pre_KK
                companies_data_2.append((company.name, now_KK, now_pre, rdf_all_temp))

            # 2. ループの最適化
            for i, (name, now_KK, now_pre, rdf) in enumerate(companies_data_2):
                col3_2, col4_2, col5_2 = st.columns((2, 2, 4))
                with col3_2:
                    st.subheader(name)
                    st.button("株価を見る", key=f"chose_companies_key_{i}", on_click=partial(change_page, 2, name))
                with col4_2:
                    st.metric(label='現在値', value=f'{round(now_KK,1)} 円', delta=f'{round(now_pre,1)} 円', delta_color='inverse')
                with col5_2:
                    buf = make_simple_graph(name, rdf)
                    st.image(buf)


        st.write("_______________________________________________________________________________________________________")

        st.button("選択可能銘柄一覧へ戻る",on_click=lambda: change_page(1))


    # 企業情報画面
    def page4():
        st.title("企業情報")
        st.write("_______________________________________________________________________________________________________")
        st.button("トレード画面へ戻る",on_click=lambda: change_page(2))
        st.write("_______________________________________________________________________________________________________")

        name = st.session_state.target_company.name
        rdf_all = st.session_state.target_company.rdf_all

        now_KK = rdf_all['Close'][-1]

        index = st.session_state.c_master[st.session_state.c_master['企業名'] == name].index.values[0]

        # pd.set_option('display.max_colwidth', 100)

        Financial_anounce_day = dt.datetime.strptime(st.session_state.c_master["短信決算発表日"][index], "%Y/%m/%d")

        year = 2021
        if Financial_anounce_day <= st.session_state.now:
            year = 2022

        for i in range(2018,year):
            st.session_state.sales_df['売上'].loc[f'{i}'] = st.session_state.c_master[f'{i}売上'][index]
            st.session_state.sales_df['営業利益'].loc[f'{i}'] = st.session_state.c_master[f'{i}営業利益'][index]
            st.session_state.sales_df['当期利益'].loc[f'{i}'] = st.session_state.c_master[f'{i}当期利益'][index]
            st.session_state.sales_df['基本的1株当たりの当期利益'].loc[f'{i}'] = st.session_state.c_master[f'{i}基本的1株当たり当期利益'][index]

        for i in range(2020,year):
            st.session_state.CF_df['営業CF'].loc[f'{i}'] = st.session_state.c_master[f'{i}営業CF'][index]
            st.session_state.CF_df['投資CF'].loc[f'{i}'] = st.session_state.c_master[f'{i}投資CF'][index]
            st.session_state.CF_df['財務CF'].loc[f'{i}'] = st.session_state.c_master[f'{i}財務CF'][index]

            st.session_state.FS_df[f'{i}'].loc['1株当たりの当期純利益'] = st.session_state.c_master[f'{i}1株当たりの当期純利益'][index]
            data_temp1 = st.session_state.c_master[f'{i}1株当たりの当期純利益'][index].replace(',','')
            data_temp1 = data_temp1.replace('△','-')
            st.session_state.FS_df[f'{i}'].loc['PER'] =  round(now_KK / float(data_temp1), 1)
            st.session_state.FS_df[f'{i}'].loc['1株当たりの純資産'] = st.session_state.c_master[f'{i}1株当たりの純資産'][index]
            data_temp2 = st.session_state.c_master[f'{i}1株当たりの純資産'][index].replace(',','')
            data_temp2 = data_temp2.replace('△','-')
            st.session_state.FS_df[f'{i}'].loc['PBR'] = round(now_KK / float(data_temp2) , 1) 
            st.session_state.FS_df[f'{i}'].loc['ROA'] = st.session_state.c_master[f'{i}ROA'][index]
            st.session_state.FS_df[f'{i}'].loc['ROE'] = st.session_state.c_master[f'{i}ROE'][index]
            st.session_state.FS_df[f'{i}'].loc['自己資本比率'] = st.session_state.c_master[f'{i}自己資本比率'][index]

            st.session_state.div_df[f'{i}'].loc['配当性向'] = st.session_state.c_master[f'{i}配当性向'][index]
            st.session_state.div_df[f'{i}'].loc['配当利回り'] = st.session_state.c_master[f'{i}配当利回り'][index]

        st.session_state.div_df2['中間'].loc['金額'] = st.session_state.c_master['中間配当'][index]
        st.session_state.div_df2['期末'].loc['金額'] = st.session_state.c_master['期末配当'][index]
        st.session_state.div_df2['中間'].loc['配当権利付き最終日'] = st.session_state.c_master['中間配当権利付き最終日'][index]
        st.session_state.div_df2['期末'].loc['配当権利付き最終日'] = st.session_state.c_master['期末配当権利付き最終日'][index]
        st.session_state.div_df2['中間'].loc['配当基準日'] = st.session_state.c_master['中間配当基準日'][index]
        st.session_state.div_df2['期末'].loc['配当基準日'] = st.session_state.c_master['期末配当基準日'][index]

        st.subheader(f"{name}の企業情報")

        # セレクトボックスでページを選択
        selected_page = st.selectbox("ページを選択してください", ["業績", "財務情報", "配当"])

        # 選択されたページに基づいて内容を表示
        if selected_page == "業績":
            st.write('売上推移')
            st.write(st.session_state.sales_df)

            for i in range(2018,year):
                st.session_state.sales_df['売上'].loc[f'{i}'] = float(st.session_state.c_master[f'{i}売上'][index].replace(',',''))

            fig, ax1 = plt.subplots(figsize=(10,6))

            # 売上を棒グラフで表示
            ax1.bar(st.session_state.sales_df.index, st.session_state.sales_df['売上'], color='blue', alpha=0.6, label='売上')
            ax1.set_xlabel('年')
            ax1.set_ylabel('売上')
            ax1.tick_params(axis='y', labelcolor='blue')

            # y軸の最小値をデータの最小値より300,000低く設定
            min_value = st.session_state.sales_df['売上'].min()
            ax1.set_ylim([min_value * 0.9, ax1.get_ylim()[1]])

            ax1.legend(loc="upper left")
            plt.title('売上の推移')
            plt.tight_layout()
            st.pyplot(fig)


        elif selected_page == "財務情報":
            st.write('キャッシュフロー')
            st.write(st.session_state.CF_df)
            st.write('財務諸表')
            st.write(st.session_state.FS_df)

            st.write(f"短信決算発表日：{st.session_state.c_master['短信決算発表日'][index]}")

        elif selected_page == "配当":
            st.write('配当')
            st.write(st.session_state.div_df)
            st.write(st.session_state.div_df2)


    # 結果画面
    def page5():
        st.title("結果画面")

        st.header(f"{st.session_state.acount_name}さんの結果")

        # 現在時刻を終了時間に合わせる
        st.session_state.now = st.session_state.all_range_end

        #保有資産に各保有株の現在値*株式数分を加算する
        if st.session_state.possess_money_bool==False:
            for i in range(0,len(st.session_state.possess_KK_df)):
                st.session_state.possess_money += st.session_state.possess_KK_df['現在の株価'][i] * st.session_state.possess_KK_df['保有株式数'][i]
                
            st.session_state.possess_money_bool = True

        st.write(f"保有資産：{st.session_state.possess_money}")

        benef = st.session_state.possess_money - st.session_state.possess_money_init
        
        if benef < 0:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:green'> {round(benef,1)}円</span> </span>の損失を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)
        else:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:red'> +{round(benef,1)}円</span> </span>の利益を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)

        if not st.session_state.sell_log.empty:
            # リスク許容度計算
            VOL_delta_list = []
            for i in range(0, len(st.session_state.buy_log)):
                VOL_delta = st.session_state.buy_log["当日のボラティリティ"][i] - st.session_state.buy_log["当日のボラティリティ平均"][i]
                VOL_delta_list.append(VOL_delta)
                
            V_delta_mean = round(np.mean(VOL_delta_list), 2)
            Risk_tolerance = "中"
            if V_delta_mean < -1.0:
                Risk_tolerance = "低"
            
            if V_delta_mean > 1.0:
                Risk_tolerance = "高"

            # 売買期間計算
            tern_delta_list = []
            for i in range(0, len(st.session_state.sell_log)):
                sell_day = st.session_state.sell_log["年月"][i]
                sell_day = dt.datetime.strptime(sell_day, "%Y/%m/%d")
                c_name = st.session_state.sell_log["企業名"][i]
                buy_day_df = st.session_state.buy_log[st.session_state.buy_log['企業名']==c_name]

                for j in range(0, len(buy_day_df)):
                    buy_day = dt.datetime.strptime(buy_day_df.iloc[j]['年月'], "%Y/%m/%d")
                    tern_delta = sell_day - buy_day
                    tern_delta_day = tern_delta.days
                    if tern_delta_day > 0:
                        tern_delta_list.append(tern_delta_day)
                
            tern_mean = round(np.mean(tern_delta_list), 2)
            tern = "短"                
            if tern_mean > 100:
                tern = "長"

            # 購入金額比率計算
            buy_rate_list = []
            for i in range(0, len(st.session_state.buy_log)):
                buy_rate = st.session_state.buy_log['購入金額比率'][i]
                buy_rate_list.append(buy_rate)

            buy_rate_mean = round(np.mean(buy_rate_list), 2)
            tendency = "集中投資型"
            if buy_rate_mean < 0.2:
                tendency = "分散投資型"
                    

            # ユーザからの情報をデータフレームとして受け取る
            behavioral_sell_data = {
                "取引回数": [len(st.session_state.sell_log)], 
                "投資根拠": [st.session_state.sell_log['売却根拠']],
                "運用成績": [st.session_state.sell_log['利益']],  
                "取引株式数": [st.session_state.sell_log['売却株式数']]
            }
            bdf = pd.DataFrame(behavioral_sell_data)

            # ユーザからの情報をデータフレームとして受け取る
            behavioral_buy_data = {
                "取引回数": [len(st.session_state.buy_log)],  # 1年間の取引回数
                "投資根拠": [st.session_state.buy_log['購入根拠']],
                "取引量": [st.session_state.buy_log['購入金額']]
            }
            bdf2 = pd.DataFrame(behavioral_buy_data)

            trade_value = display_distribution(bdf2['取引量'][0])
            wield_grades = display_distribution(bdf['運用成績'][0])
            
            buy_reason_count, buy_reason_ratios = display_distribution2(bdf2['投資根拠'][0])
            sell_reason_count, sell_reason_ratios = display_distribution2(bdf['投資根拠'][0])

            #個人の性格情報から分類型にポイントを与える
            st.session_state.personal['性格']['新規性'] = st.session_state.Open
            st.session_state.personal['性格']['誠実性'] = st.session_state.Integrity
            st.session_state.personal['性格']['外交性'] = st.session_state.Diplomatic
            st.session_state.personal['性格']['協調性'] = st.session_state.Coordination
            st.session_state.personal['性格']['神経症傾向'] = st.session_state.Neuroticism

            classify_type_df = classify_action_type(st.session_state.personal, st.session_state.sell_log, buy_reason_ratios, sell_reason_ratios, trade_value, wield_grades)

            # 最も高いポイントに分類
            action_type = classify_type_df[classify_type_df['分類型']==classify_type_df['分類型'].max()].index.values[0]
        
            target_action_type = st.session_state.action_type_advice[st.session_state.action_type_advice["行動型"]==action_type]
            target_action_type = target_action_type.reset_index(drop=True)

            feature = target_action_type["特徴"][0]
            weekness = target_action_type["欠点"][0]
            advice_text = target_action_type["アドバイス"][0]
            essay = target_action_type["文章"][0]

            st.write("_______________________________________________________________________________________________________")
            st.subheader("総合評価")
            # 枠線で囲む文章
            overall_advice_temp = st.session_state.overall_advice[st.session_state.overall_advice["リスク許容度"]==Risk_tolerance]
            overall_advice_temp = overall_advice_temp[overall_advice_temp["売買期間"]==tern]
            overall_advice_temp = overall_advice_temp[overall_advice_temp["投資傾向"]==tendency]
            overall_advice_temp = overall_advice_temp[overall_advice_temp["投資型"]==action_type]

            text = overall_advice_temp['アドバイス'].values[0]

            html_code = f"""
            <div style="
                border: 2px solid #000000;
                border-radius: 5px;
                padding: 10px;
            ">
                {text}
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)

            st.write("_______________________________________________________________________________________________________")

            st.subheader("全体の投資傾向について")

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">投資行動型</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:fantasy; color:blue; font-size: 24px;">{action_type}</p>', unsafe_allow_html=True)
            # st.write("特徴：")
            # st.write(feature)
            # st.write("欠点：")
            # st.write(weekness)
            # st.write("アドバイス：")
            # st.write(advice_text)
            html_code = f"""
            <div style="
                border: 2px solid #000000;
                border-radius: 5px;
                padding: 10px;
            ">
                {essay}
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)


            # checkの初期値を設定
            st.session_state.check = False

            st.write("################################################################################")
            check = st.checkbox("投資行動の情報を表示", value = st.session_state.check)
            if check:
                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">リスク許容度</p>', unsafe_allow_html=True)
                st.write(f"ボラティリティ誤差平均：{V_delta_mean}")
                st.write(f"リスク許容度：{Risk_tolerance}")

                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">売買期間</p>', unsafe_allow_html=True)
                st.write(f"売買期間平均：{tern_mean}")
                st.write(f"売買期間：{tern}")

                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">投資傾向</p>', unsafe_allow_html=True)
                st.write(f"購入金額比率平均：{buy_rate_mean}")
                st.write(f"投資傾向：{tendency}")


                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">取引量</p>', unsafe_allow_html=True)
                #各統計量表示
                st.dataframe(trade_value)
                # ヒストグラムを作成
                fig, ax = plt.subplots()
                ax.hist(bdf2['取引量'][0], bins=10, color='blue', alpha=0.7, edgecolor='black')
                ax.legend(loc="upper left")
                plt.title('取引量のヒストグラム')
                ax.set_xlabel('１取引あたりの購入金額')
                ax.set_ylabel('count')
                plt.tight_layout()
                # Streamlitでヒストグラムを表示
                st.pyplot(fig)

                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">運用成績</p>', unsafe_allow_html=True)
                #各統計量表示
                st.dataframe(wield_grades)
                # ヒストグラムを作成
                fig, ax = plt.subplots()
                ax.hist(bdf['運用成績'][0], bins=10, color='blue', alpha=0.7, edgecolor='black')
                ax.legend(loc="upper left")
                plt.title('利益のヒストグラム')
                ax.set_xlabel('利益')
                ax.set_ylabel('count')
                plt.tight_layout()
                # Streamlitでヒストグラムを表示
                st.pyplot(fig)

                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">購入根拠</p>', unsafe_allow_html=True)
                # 統計量を表示
                col_b1, col_b2 = st.columns((4, 6))
                with col_b1:
                    st.write("\n各カテゴリのカウント:")
                    st.write(buy_reason_count)

                    st.write("\n各カテゴリの割合:")
                    st.write(buy_reason_ratios)

                with col_b2:
                    # 円グラフを作成
                    fig, ax = plt.subplots()
                    ax.pie(buy_reason_ratios, labels=buy_reason_ratios.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                    # Streamlitに表示
                    st.pyplot(fig)

                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">売却根拠</p>', unsafe_allow_html=True)
                # 統計量を表示
                col_s1, col_s2 = st.columns((4, 6))
                with col_s1:
                    st.write("\n各カテゴリのカウント:")
                    st.write(sell_reason_count)

                    st.write("\n各カテゴリの割合:")
                    st.write(sell_reason_ratios)
                    
                with col_s2:
                    # 円グラフを作成
                    fig, ax = plt.subplots()
                    ax.pie(sell_reason_ratios, labels=sell_reason_ratios.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                    # Streamlitに表示
                    st.pyplot(fig)


            st.write("################################################################################")

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">行動経済学の指摘事項</p>', unsafe_allow_html=True)
            if "advice" not in st.session_state:
                st.session_state.advice_df = advice(buy_reason_ratios, st.session_state.buy_log, st.session_state.sell_log)
                st.session_state.advice = True

            if st.session_state.advice_df.empty:
                st.write("特になし")
            else:
                for i in range(0, len(st.session_state.advice_df)):
                    st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 24px;">{st.session_state.advice_df["指摘事項"][i]}</p>', unsafe_allow_html=True)

                    target_BE = st.session_state.Behavioral_Economics[st.session_state.Behavioral_Economics['理論']==st.session_state.advice_df['指摘事項'][i]]
                    target_BE = target_BE.reset_index(drop=True)
                    st.write(target_BE['内容'][0])
                    # st.write("アドバイス")
                    st.write(f"　→ {target_BE['アドバイス'][0]}")


            st.write("_______________________________________________________________________________________________________")
            st.subheader("各取引について")

            # st.write("各種投資行動の説明を書く")

            if "some_trade_advice" not in st.session_state:
                st.session_state.trade_advice_df = some_trade_advice(st.session_state.buy_log, st.session_state.sell_log)  
                st.session_state.some_trade_advice = True

            if st.session_state.trade_advice_df.empty:
                st.write("指摘することはありません。")
            else:
                st.write("あなたの売買データから以下のバイアスが見つかりました。")
                #trade_advice_dfからグラフを作成する
                for i in range(0,len(st.session_state.trade_advice_df)):
                    tgt_name = st.session_state.trade_advice_df.iloc[i]['企業名']
                    tgt_sell_day = st.session_state.sell_log[st.session_state.sell_log['企業名']==tgt_name]['年月'].iloc[-1]

                    tgt_buy_day = st.session_state.buy_log[st.session_state.buy_log['企業名']==tgt_name]['年月'].iloc[-1]

                    tgt_buy_day = dt.datetime.strptime(tgt_buy_day, "%Y/%m/%d")
                    tgt_sell_day = dt.datetime.strptime(tgt_sell_day, "%Y/%m/%d")

                    tgt_buy_day_temp = tgt_buy_day + dt.timedelta(days=-30)
                    tgt_sell_day_temp = tgt_sell_day + dt.timedelta(days=30)

                    index = st.session_state.c_master.loc[(st.session_state.c_master['企業名']==tgt_name)].index.values[0]
                    #companiesからデータを抽出する
                    target_company = st.session_state.loaded_companies[index]
                    name = target_company.name
                    rdf = target_company.rdf_all[tgt_buy_day_temp:tgt_sell_day_temp]


                    st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 24px;">{st.session_state.trade_advice_df.iloc[i]["指摘事項"]}</p>', unsafe_allow_html=True)
                    target_BE2 = st.session_state.Behavioral_Economics[st.session_state.Behavioral_Economics['理論']==st.session_state.trade_advice_df.iloc[i]['指摘事項']]
                    target_BE2 = target_BE2.reset_index(drop=True)
                    st.write(target_BE2['内容'][0])

                    if st.session_state.trade_advice_df.iloc[i]['指摘事項'] == '現在志向バイアス':
                        rdf_temp = rdf[tgt_sell_day:tgt_sell_day_temp]
                        max_date = rdf_temp[rdf_temp['Close']==rdf_temp['Close'].max()].index.values[0]
                        make_graph(name, rdf, buy_date=tgt_buy_day, sell_date=tgt_sell_day, now_kk_bool=True, max_date=max_date)
                    else:
                        make_graph(name, rdf, buy_date=tgt_buy_day, sell_date=tgt_sell_day, now_kk_bool=True)

                    tgt_benef = st.session_state.sell_log[st.session_state.sell_log['企業名']==tgt_name]['利益'].iloc[-1]

                    if tgt_benef < 0:
                        colored_text = f"利益：　<span style='font-size:20px'><span style='color:green'> {round(tgt_benef,1)}円</span> </span>"
                        st.markdown(colored_text, unsafe_allow_html=True)
                    else:
                        colored_text = f"利益：　<span style='font-size:20px'><span style='color:red'> +{round(tgt_benef,1)}円</span> </span>"
                        st.markdown(colored_text, unsafe_allow_html=True)

                    # st.write("アドバイス")
                    html_code = f"""
                    <div style="
                        border: 2px solid #000000;
                        border-radius: 5px;
                        padding: 10px;
                    ">
                        {target_BE2['アドバイス'][0]}
                    </div>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                    st.write("")

            st.write("_______________________________________________________________________________________________________")

            #現在時刻情報を取得
            dt_now = dt.datetime.now()
            dt_now_str = dt_now.strftime('%Y/%m/%d')

            Dividends_df_temp = st.session_state.Dividends_df.copy()
            for i in range(0, len(Dividends_df_temp)):
                Dividends_df_temp['配当基準日'].iloc[i] = Dividends_df_temp['配当基準日'].iloc[i].strftime('%Y/%m/%d')

            # ユーザからの情報をデータフレームとして受け取る
            Simulation_Result = {
                "実施日": [dt_now_str],
                "実施回数": [st.session_state.num], 
                "行動型": [action_type],
                "レベル": [st.session_state.level_id],  
                "運用成績": [benef],
                "buy_log": [st.session_state.buy_log],
                "sell_log": [st.session_state.sell_log],
                "Dividends": [Dividends_df_temp],
                "アドバイス": [st.session_state.advice_df],
                "各取引に関するアドバイス": [st.session_state.trade_advice_df]
            }
            Simulation_Results_df = pd.DataFrame(Simulation_Result)

            #実績画面にデータを保存する
            Simulation_Results_instance = Simulation_Results(dates=dt_now_str, num=st.session_state.num, action_type=action_type, LEVEL=st.session_state.level_id, investment_result=benef, buy_log=st.session_state.buy_log, sell_log=st.session_state.sell_log, Dividends=st.session_state.Dividends_df, advice=st.session_state.advice_df, trade_advice=st.session_state.trade_advice_df)

            if st.session_state.result_bool == False:
                # クラスを利用してresultにデータを保存する
                st.session_state.result.append(Simulation_Results_instance)

                #データベースに保存する
                insert_data_to_db(st.session_state.personal_df, Simulation_Results_df)
                st.session_state.num += 1
                st.session_state.result_bool = True


            if st.button(" シミュレーションを終了する "):
                if st.session_state.survey_bool==False:
                    change_page2(7)
                
                st.session_state.show_page = False

        else:
            st.write("株の取引が行われていないため結果を表示できません")

            if st.button(" シミュレーションを終了する "):
                st.session_state.show_page = False


    # 購入画面
    def page6():
        st.title("購入画面") 

        st.button("キャンセル",on_click=lambda: change_page(2))

        #購入株式数
        st.session_state.buy_num = st.slider("購入株式数", 100, 1000, st.session_state.get("buy_num", 100),step=100)

        if "Rationale_for_purchase" not in st.session_state:
            st.session_state.Rationale_for_purchase = "指定なし"

        buy_reason_arrow = [
            "チャート形状",
            "業績が安定している",
            "財務データ",
            "利回りがいい",
            "配当目当て",
            "リスクが小さい",
            "直感",
            "過去の経験から",
            "安いと思ったから",
            "全体的な景気",
            "好きな企業に投資",
            "アナリストによる評価",
            "その他"
        ]

        #購入根拠
        st.session_state.Rationale_for_purchase = st.radio("購入根拠", buy_reason_arrow)

        st.button("購入する",on_click=lambda: buy(name, rdf_all))

            
    # 売却画面
    def page7():
        st.title("売却画面") 

        st.button("キャンセル",on_click=lambda: change_page(2))

        #売却株式数
        st.session_state.sell_num = st.slider("売却株式数", 100, 1000, st.session_state.get("sell_num", 100),step=100)

        if "basis_for_sale" not in st.session_state:
            st.session_state.basis_for_sale = "指定なし"

        sell_reason_arrow = [
            "チャート形状",
            "直感",
            "過去の経験から",
            "全体的な景気",
            "損切り",
            "利益確定売り",
            "その他"
        ]

        #購入根拠
        st.session_state.basis_for_sale = st.radio("売却根拠", sell_reason_arrow)

        st.button("売却する",on_click=lambda: sell(name, rdf_all))


    # ログ画面
    def page8():
        st.title("ログ画面")
        st.subheader("買い・売りログデータ")
        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.dataframe(st.session_state.buy_log)
        with col_sell:
            st.dataframe(st.session_state.sell_log)

        st.subheader("配当に関するデータ")
        st.dataframe(st.session_state.Dividends_df)

        st.button("選択可能銘柄一覧へ戻る",on_click=lambda: change_page(1))

    # 日経平均
    def page9():
        st.title("日経平均株価")
        st.write("_______________________________________________________________________________________________________")
        st.button("選択可能銘柄一覧へ",on_click = lambda: change_page(1))
        st.write("_______________________________________________________________________________________________________")

        index = st.session_state.c_master[st.session_state.c_master['企業名'] == '日経平均'].index.values[0]
        st.session_state.target_company = st.session_state.loaded_companies[index]

        #companiesからデータを抽出する
        name = st.session_state.target_company.name
        rdf = st.session_state.target_company._rdf
        rdf_all = st.session_state.target_company.rdf_all

        rdf = rdf_all[all_range_start : st.session_state.now]

        #グラフ表示
        make_graph(name, rdf)



    if "page_id" not in st.session_state:
        st.session_state.page_id = "page1"

    if st.session_state.page_id == "page1":
        page1()

    if st.session_state.page_id == "page2":
        page2()

    if st.session_state.page_id == "page3":
        page3()

    if st.session_state.page_id == "page4":
        page4()

    if st.session_state.page_id == "page5":
        page5()

    if st.session_state.page_id == "page6":
        page6()   

    if st.session_state.page_id == "page7":
        page7()   

    if st.session_state.page_id == "page8":
        page8()

    if st.session_state.page_id == "page9":
        page9() 

    if st.session_state.page_id == "page10":
        page10() 

    st.sidebar.button("一日進める", key='uniq_key_1',on_click=lambda: add_next_day(1))
    st.sidebar.button("一週間進める", key='uniq_key_2', on_click=lambda: add_next_day(7))
    st.sidebar.write(f"now = {st.session_state.now}")
    st.sidebar.write(f"end = {st.session_state.all_range_end}")

    st.sidebar.header(f"買付余力：{round(st.session_state.possess_money)} 円")
    if st.session_state.possess_KK_df.empty == True:
        st.sidebar.write("あなたは現在株を所有していません。") 
    else:
        st.sidebar.write("現在保有している株式") 
        st.sidebar.dataframe(st.session_state.possess_KK_df)

    st.sidebar.button("保有株式へ", key='uniq_key_3',on_click=lambda: change_page(3))
    st.sidebar.button("売買ログへ", key='uniq_key_4',on_click=lambda: change_page(8))
    

    st.sidebar.write("_______________________________________________________________________________________________________")

    if st.sidebar.button('シミュレーションを中断する'):
        st.session_state.show_page = False

        
#_____________________________________________________________スタート画面_________________________________________________________________________________________________________________________________

else:
    # ホーム画面
    def start_system():
        st.title("投資シミュレーションシステム")
        st.image("image/home_screen.png")
        st.button("このシステムの使い方を見る",on_click=lambda: change_page2(5))
        st.button("スタート画面へ",on_click=lambda: change_page2(1))

    # スタート画面
    def page2_1():
        st.title("投資シミュレーションシステム")
        st.subheader("スタート画面")

        col5, col6 = st.columns((5, 5))
        with col5:
            st.button("投資経験がない方はこちらへ",on_click=lambda: change_page2(4))

        with col6:
            st.button("このシステムの使い方", on_click=lambda: change_page2(5))

        st.button("これまでの実績", on_click=lambda: change_page2(2))
        st.button("アカウント設定", on_click=lambda: change_page2(3))

        st.session_state.level_id = st.selectbox(
            "期間設定",
            ["1ヶ月", "3ヶ月", "6ヶ月", "12ヶ月"],
            key="level-select"
        )

        if st.session_state.level_id == "1ヶ月":
            st.session_state.all_range_end = dt.datetime(2021,2,1) 
            # st.write("１ヶ月で利益を出してください")
        
        if st.session_state.level_id == "3ヶ月":
            st.session_state.all_range_end = dt.datetime(2021,4,1) 
            # st.write("３ヶ月で +5万円の利益を出してください")

        if st.session_state.level_id == "6ヶ月":
            st.session_state.all_range_end = dt.datetime(2021,7,1) 
            # st.write("６ヶ月で +10万円の利益を出してください")

        if st.session_state.level_id == "12ヶ月":
            st.session_state.all_range_end = dt.datetime(2022,1,4) 
            # st.write("１年で +20万円利益を出してください")

        if not st.session_state.personal_df.empty:            
            # データベースに保存
            save_userdata()

            st.write(f"現在の保有資産：{st.session_state.possess_money}")

        st.button('シミュレーションをはじめから始める',on_click=lambda: start_sym(1))
        st.button('シミュレーションを続きから始める',on_click=lambda: start_sym(2))

        if "pass_bool" not in st.session_state:
            st.session_state.pass_bool = False

        if not st.session_state.personal_df.empty:
            if st.session_state.acount_name == "test_1234":
                if st.session_state.pass_bool == False:
                    pw = st.sidebar.text_input("パスワード")
                    st.sidebar.checkbox("")
                    if pw == "311055":
                        st.session_state.pass_bool = True
                    
                else:
                    st.sidebar.button('データベースの確認',on_click=lambda: change_page2(99))
                    st.sidebar.button('データベースのダウンロード',on_click=download_db)
                    #データベースファイルのアップロード
                    uploaded_file = st.sidebar.file_uploader(".dbファイルをアップロードしてください", type=["db"])
                    if uploaded_file is not None:
                        temp_db_path = "temp_uploaded.db"
                        with open(temp_db_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        if os.path.exists(temp_db_path):
                            existing_db_path = 'my_database.db'
                            shutil.copyfile(temp_db_path, existing_db_path)
                            st.sidebar.success("データベースファイルが置き換えられました。")
                        else:
                            st.error("一時ファイルの保存に失敗しました。")


    # 実績
    def page2_2():
        st.title("実績")

        st.write("########################################################################################")

        for i, result in enumerate(st.session_state.result, start=1):
            st.subheader(f"第{i}回")
            result.display()
            st.button("結果を見る", key=f"result_{i}", on_click=partial(change_page2, 6, result.buy_log, result.sell_log, result.investment_result, result.advice))

            st.write("########################################################################################")

        # if "advice_temp" in st.session_state:
        #     del st.session_state.advice_temp

        if "some_trade_advice_temp" in st.session_state:
            del st.session_state.some_trade_advice_temp
            
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

    # 各変数うのデータをpersonal_dfに格納する
    def create_acount():
        st.session_state.account_created = True
        st.session_state.personal_df["ユーザID"] = st.session_state.acount_ID,
        st.session_state.personal_df["ユーザ名"] = st.session_state.acount_name,
        st.session_state.personal_df["年齢"] = st.session_state.acount_age,
        st.session_state.personal_df["性別"] = st.session_state.acount_sex,
        st.session_state.personal_df["投資経験の有無"] = st.session_state.trade_expe,
        st.session_state.personal_df["投資に関する知識の有無"] = st.session_state.trade_knowledge,
        st.session_state.personal_df["開放性"] = st.session_state.Open,
        st.session_state.personal_df["誠実性"] = st.session_state.Integrity,
        st.session_state.personal_df["外交性"] = st.session_state.Diplomatic,
        st.session_state.personal_df["協調性"] = st.session_state.Coordination,
        st.session_state.personal_df["神経症傾向"] = st.session_state.Neuroticism
        change_page2(1)

    def load_data(acount_name):
        # データベースに接続
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()        

        # SQLクエリで該当するデータを取得
        cursor.execute("SELECT * FROM personal_info WHERE ユーザ名=?", (acount_name,))
        rows = cursor.fetchall()

        # データベースに指定のメールアドレスが存在しない場合の処理
        if not rows:
            st.sidebar.write("指定されたアカウント名の情報はデータベースに存在しません。")
            return

        # データの表示
        st.session_state.acount_name = rows[0][0]
        st.session_state.acount_ID = rows[0][1]
        st.session_state.acount_age = rows[0][2]
        st.session_state.acount_sex = rows[0][3]
        st.session_state.trade_expe = rows[0][4]
        st.session_state.trade_knowledge = rows[0][5]
        st.session_state.Open = int(rows[0][6])
        st.session_state.Integrity = int(rows[0][7])
        st.session_state.Diplomatic = int(rows[0][8])
        st.session_state.Coordination = int(rows[0][9])
        st.session_state.Neuroticism = int(rows[0][10])

        # resultデータの取り出し
        st.session_state.result = [] 
        cursor.execute("SELECT data FROM simulation_results WHERE ユーザ名=?", (acount_name,))
        rows = cursor.fetchall()

        if rows:
            # データベースから取得したデータをデシリアライズします
            fetched_data = pickle.loads(rows[0][0])
            
            # fetched_dataがリストであることを確認します
            if isinstance(fetched_data, list):
                st.session_state.result.extend(fetched_data)
            else:
                st.session_state.result.append(fetched_data)

        st.session_state.num = len(st.session_state.result) + 1

        # その他変数データの取り出し
        # ユーザ名が一致するデータの取得
        cursor.execute('SELECT value FROM value_table WHERE ユーザ名 = ?', (acount_name,))
        data = cursor.fetchall()

        # 一致するデータがある場合
        if data:            
            df = pickle.loads(data[0][0])
            
            st.session_state.now = df["now"][0]
            st.session_state.chose_companies_name_list = df["chose_companies_name_list"][0]
            st.session_state.possess_money = df["possess_money"][0]
            st.session_state.possess_money_init = df["possess_money_init"][0]
            st.session_state.possess_KK_df = pd.DataFrame(df["possess_KK_df"][0])
            st.session_state.buy_log = pd.DataFrame(df["buy_log"][0])
            st.session_state.sell_log = pd.DataFrame(df["sell_log"][0])
            st.session_state.Dividends_df = pd.DataFrame(df["Dividends_df"][0])
            st.session_state.advice_df = pd.DataFrame(df["advice_df"][0])
            st.session_state.trade_advice_df = pd.DataFrame(df["trade_advice_df"][0])
            st.session_state.create_chose_companies_executed = df["create_chose_companies_executed"][0]
            st.session_state.selected_company = df["selected_company"][0]
            st.session_state.result_bool = df["result_bool"][0]
            st.session_state.survey_bool = df["survey_bool"][0]
            st.session_state.possess_money_bool = df["possess_money_bool"][0]

            st.session_state.chose_companies = []
            for i in range(0,len(st.session_state.chose_companies_name_list)):
                company_name = st.session_state.chose_companies_name_list[i]
                index = st.session_state.c_master[st.session_state.c_master['企業名'] == company_name].index.values[0]
                st.session_state.chose_companies.append(st.session_state.loaded_companies[index])

        # データベース接続のクローズ
        conn.close()

        create_acount()

    # アカウント画面
    def page2_3():
        st.title("アカウント設定")

        st.write("アカウントを持っていない場合はこちらから作成してください。")
        st.button("アカウントを作成",on_click=lambda: change_page2("3_a"))

        st.write("アカウント名を入力して以前までのデータを取得することができます。")
        
        if not st.session_state.load_data_bool:
            if st.button("以前のデータを取得する"):
                st.session_state.load_data_bool = True

        # アカウント名の情報によりデータを取得する
        if st.session_state.load_data_bool:
            col7, col8 = st.columns((7, 3))
            with col7:
                st.session_state.acount_name = st.text_input("アカウント名を入力してください", value=st.session_state.get("acount_name", ""))
            with col8:
                st.button("データを取得する",on_click=lambda: load_data(st.session_state.acount_name))

        if st.session_state.account_created:
            st.write("_______________________________________________________________________________________________________")
            st.subheader("アカウント情報")
            st.write(f"ユーザID：{st.session_state.personal_df['ユーザID'][0]}")
            st.write(f"ユーザ名：{st.session_state.personal_df['ユーザ名'][0]}")
            st.write(f"年齢：{st.session_state.personal_df['年齢'][0]}")
            st.write(f"性別：{st.session_state.personal_df['性別'][0]}")
            st.write(f"投資経験の有無：{st.session_state.personal_df['投資経験の有無'][0]}")
            st.write(f"投資に関する知識の有無：{st.session_state.personal_df['投資に関する知識の有無'][0]}")
            st.write(f"開放性：{st.session_state.personal_df['開放性'][0]}")
            st.write(f"誠実性：{st.session_state.personal_df['誠実性'][0]}")
            st.write(f"外交性：{st.session_state.personal_df['外交性'][0]}")
            st.write(f"協調性：{st.session_state.personal_df['協調性'][0]}")
            st.write(f"神経症傾向：{st.session_state.personal_df['神経症傾向'][0]}")
            st.write(f"現在の保有資産：{st.session_state.possess_money}")
        st.write("_______________________________________________________________________________________________________")
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

    # アカウント名がデータベースにあるかをチェックする
    def check_acount_name(acount_name):
        # データベースに接続
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()

        # SQLクエリで指定されたユーザ名が存在するか確認
        cursor.execute("SELECT 1 FROM personal_info WHERE ユーザ名=?", (acount_name,))
        result = cursor.fetchone()

        # データベース接続のクローズ
        conn.close()

        # ユーザ名が存在する場合、resultには(1,)のようなタプルが返され、存在しない場合はNoneが返されます。
        return result is not None        

    # アカウントを新規作成する
    def page2_3_a():

        if "acount_ID" not in st.session_state:
            st.session_state.acount_ID = str(uuid.uuid4())

        st.write(f"ユーザID：{st.session_state.acount_ID}")
        st.session_state.acount_name = st.text_input("アカウント名を入力してください", value=st.session_state.get("acount_name", ""))

        # アカウント名がすでに使われている場合はメッセージを出力してキャンセルする
        if check_acount_name(st.session_state.acount_name ):
            st.error("そのアカウント名はすでに使われているため使用できません")
            st.session_state.acount_name = ""

        st.session_state.acount_age = st.text_input("年齢を入力してください", value=st.session_state.get("acount_age", ""))
        st.session_state.acount_sex = st.selectbox("性別を入力してください", ("男", "女"), index=0 if st.session_state.get("acount_sex", "男") == "男" else 1)
        
        #投資経験の有無
        trade_expe_arrow = [
            "投資経験がない",
            "少しだけある",
            "1年未満",
            "3年未満",
            "3年以上"
        ]
        st.session_state.trade_expe = st.radio("投資経験の有無", trade_expe_arrow)

        #投資に関する知識の有無
        trade_knowledge_arrow = [
            "ない",
            "少しだけある",
            "ある方だと思う",
            "十分にある"
        ]
        st.session_state.trade_knowledge = st.radio("投資に関する知識の有無", trade_knowledge_arrow)


        st.write("以下のどちらかのURLから個人の性格についてのテストを実施して情報を入力してください。")
        st.write("① https://commutest.com/bigfive")
        st.write("② https://questi.jp/diagnoses/commons/bigfive/new")
        st.session_state.Open = st.slider("開放性", 0, 6, st.session_state.get("Open", 6))
        st.session_state.Integrity = st.slider("誠実性", 0, 6, st.session_state.get("Integrity", 6))
        st.session_state.Diplomatic = st.slider("外交性", 0, 6, st.session_state.get("Diplomatic", 6))
        st.session_state.Coordination = st.slider("協調性", 0, 6, st.session_state.get("Coordination", 6))
        st.session_state.Neuroticism = st.slider("神経症傾向", 0, 6, st.session_state.get("Neuroticism", 6))

        if not check_acount_name(st.session_state.acount_name) and (st.session_state.acount_name != ""):
            st.button("アカウントを作成する",on_click=lambda: create_acount())

        st.button("戻る",on_click=lambda: change_page2(3))

   # 投資について 
    def page2_4():
        st.title("投資について")
        st.write("_______________________________________________________________________________________________________")
        check1 = st.checkbox("株式投資に関して")
        if check1:
            st.subheader("株式投資の基本的な仕組み")
            st.markdown('<p style="font-family:monospace; color:black ; font-size: 18px;　text-decoration: underline;">・　株式とは何か、株式を購入することで何が得られるのか　</p>', unsafe_allow_html=True)
            st.write("株式とは、企業の一部を所有する権利のことです。株式を購入すると、その企業の利益に応じて配当を受け取ることができるほか、株価が上がれば売却時に利益を得ることができます。")

            st.markdown('<p style="font-family:monospace; color:black ; font-size: 18px;　text-decoration: underline;">・　株価の変動の原因とその影響　</p>', unsafe_allow_html=True)
            st.write("株価は、企業の業績や経済状況、市場の需給バランスなど様々な要因によって変動します。株価が上がれば利益を得ることができますが、下がれば損失を被ることもあります。")

            st.markdown('<p style="font-family:monospace; color:black ; font-size: 18px;　text-decoration: underline;">・　配当とは何か　</p>', unsafe_allow_html=True)
            st.write("配当とは、企業が利益を出した際に、その一部を株主に分配することです。配当は通常、現金で受け取ることができます。各銘柄の配当基準日にその銘柄の株を持っていた場合に配当金を受け取ることができます。")


            st.subheader("リスクとリターン")
            st.markdown('<p style="font-family:monospace; color:black ; font-size: 18px;　text-decoration: underline;">・　株式投資におけるリスクの種類とその対処方法　</p>', unsafe_allow_html=True)
            st.write("株式投資には、株価の変動によるリスクや企業の倒産リスクなどがあります。これらのリスクを軽減するためには、複数の企業の株式を購入してリスクを分散することや、十分な情報収集を行うことが大切です。")

            st.markdown('<p style="font-family:monospace; color:black ; font-size: 18px;　text-decoration: underline;">・　リターンの期待値とその実現の可能性　</p>', unsafe_allow_html=True)
            st.write("株式投資によるリターンは、株価の上昇や配当によって得られます。しかし、リターンの期待値は常に変動するため、投資の際には慎重な判断が必要です。")

        st.write("_______________________________________________________________________________________________________")
        check1_2 = st.checkbox("配当に関して")
        if check1_2:
            st.markdown('<p style="font-family:monospace; color:black ; font-size: 18px;　text-decoration: underline;">・　配当とは何か　</p>', unsafe_allow_html=True)
            st.write("配当とは、企業が利益を出した際に、その一部を株主に分配することです。配当は通常、現金で受け取ることができます。各銘柄の配当基準日にその銘柄の株を持っていた場合に配当金を受け取ることができます。")

            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">配当性向</p>', unsafe_allow_html=True)
            st.write("当期純利益のうち企業が株主にどれだけの配当金を還元しているかを表している数値")
            st.write("企業が成長するには投資する必要があるため、配当性向が低いからといって悪いわけではない。")
            st.write("")

            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">配当利回り</p>', unsafe_allow_html=True)
            st.write("株価に対する年間配当金の割合を示す指標")
            st.write("例えば、配当金が1000円で株価が30万円の株式と配当金が800円で株価が24万円の株式では、どちらの利回りがいいかわからない。")
            st.write("そんな時に統一して測る物差しとして配当利回りは利用される。")
            st.write("")


        st.write("_______________________________________________________________________________________________________")
        check2 = st.checkbox("チャートグラフの見方")
        if check2:
            st.image("image/image_elem_1.png")
            st.image("image/image_elem_2.png")

            st.write("出来高は、その日の取引量の多さを示します。")
            st.write("ローソク足チャートは、特定の期間における株価の始値、終値、高値、安値を「ローソク」と呼ばれる形で表したものです。ローソクの色や形によって、株価の上昇や下降が一目でわかります。")
            st.write("始値より終値が高くなったものを陽線、その逆を陰線と呼びます。陽線、陰線の見方は以下のようになります。")
            st.write("始値：その日の最初の株価の値を示します。")
            st.write("終値：その日の最後の株価の値を示します。")
            st.write("高値：その日の株価が一番高くなった時点の値を示します。")
            st.write("安値：その日の株価が一番低くなった時点の値を示します。")

        st.write("_______________________________________________________________________________________________________")
        check3 = st.checkbox("企業情報の見方")
        if check3:
            st.subheader("[1] 企業の安定性を見る指標")
            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">①　自己資本比率</p>', unsafe_allow_html=True)
            st.write("返済不要の自己資本が全体の資本調達の何%あるかを示す数値")
            st.write("自己資本比率が高いほど経営は安定し、倒産しにくい会社だと考えられます。一般に自己資本比率が50%なら優良企業、40%以上なら倒産しにくい会社と言われています。")
            st.write("")

            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">② 流動比率</p>', unsafe_allow_html=True)
            st.write("企業の短期間の支払い能力を示す指標。高い方が良いとされている。")
            st.write("流動比率は、資金繰りが怪しくなっている企業を簡単に調べる目安になります。")
            st.write("")

            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">③ キャッシュフロー</p>', unsafe_allow_html=True)
            st.write("企業活動などによって得られた収入から、外部への支出を引いて、手元に残る資金の流れを示したもの")
            st.write("一般に優良な会社の場合以下のようになっているのが良いとされる。")
            st.write("・営業キャッシュフローがプラス　　　（営業成績好調）")
            st.write("・投資キャッシュフローがマイナス　　（積極的な設備投資）")
            st.write("・財務キャッシュフローがマイナス　　（借入金の返済）")
            st.write("")


            st.subheader("[2] 企業の成長性、収益性を見る指標")
            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">① 売上高の推移</p>', unsafe_allow_html=True)
            st.write("売上高は、はっきりと、企業の成長性を示し、株価を長期的に押し上げる要因になっている")
            st.write("")


            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">② EPS</p>', unsafe_allow_html=True)
            st.write("１株あたりのその年１年間の純利益のこと")
            st.write("企業の純利益が上がる、もしくは発行済み株式数が減ることでEPSは増加する")
            st.write("")


            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">③ ROE（株主資本利益率）</p>', unsafe_allow_html=True)
            st.write("株主資本が企業の収益にどれだけ関係しているかを示す")
            st.write("ROEが高い企業は、積極的な経営が行われるという見方になる")
            st.write("")

            
            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">④ ROA（総資産利益率）</p>', unsafe_allow_html=True)
            st.write("事業に投下されている資産によって得られる利益")
            st.write("株主重視の経営を考えると、ROAよりROEの方が重視されることになる")
            st.write("")


            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">⑤ 売上利益率</p>', unsafe_allow_html=True)
            st.write("売上高における売上総利益の割合")        
            st.write("売上総利益が高ければ、高付加価値の商品を販売していることになり、競争力の高い企業という評価になる")
            st.write("")


            st.subheader("[3] 株価の妥当な値位置をみる指標")
            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">① PER（株価収益率）</p>', unsafe_allow_html=True)
            st.write("１株あたりの利益に対して、株価が何倍まで買われているかを示している指標")
            st.write("過去の株価の高値、安値をつけたときのPERの数値と現在のPERを比較して、現在の株価が割安か否かを判断することができる")
            st.write("")


            st.markdown('<p style="font-family:monospace; color:mediumblue ; font-size: 24px;">② PBR（株価純資産倍率）</p>', unsafe_allow_html=True)
            st.write("１株あたりの純資産に対し、株価が何倍まで買われているかを示している指標")
            st.write("PRR<1ならば、株価が下値を支えられる可能性があるという見方になる")
            st.write("")


        # st.write("_______________________________________________________________________________________________________")
        # check4 = st.checkbox("行動経済学に関して")
        # if check4:
        #     st.write("各種行動経済学に関する理論")

        st.write("_______________________________________________________________________________________________________")
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

    def display_image(num):
        return f"image/image_{num}.png" 

    def up_n():
        if st.session_state.n < 12:
            st.session_state.n += 1

    def down_n():
        if st.session_state.n > 1:
            st.session_state.n -= 1

    # 注意事項
    def page2_5_b():
        st.title("注意事項")
        st.write("_______________________________________________________________________________________________________")
        st.subheader("1. 個人情報の取り扱いについて")
        st.write("本システムにおいて登録されるすべての情報は、厳重に保管・管理されます。無許可での第三者への提供や公開は行いません。\n 個人を特定できる情報は、システムの適切な運用とサポートの目的のみで使用され、それ以外の目的での利用や分析は行いません。")

        st.subheader("2. 売買データの取り扱いについて")
        st.write("ユーザが本システムで行う売買のデータは、デモトレードの結果の分析や改善のために利用される場合があります。\n すべてのトレードデータは匿名化され、個人を特定する情報と合わせて解析されることはありません")

        st.subheader("3. デモトレードの結果に関する免責")
        st.write("本システムでのデモトレードの結果は、実際の金融市場での取引結果を保証するものではありません。\n ユーザが本システムの情報を元に実際の金融市場での取引を行った場合、その結果に対する一切の責任を当方は負いません。")

        st.subheader("4. 本システムに関する免責")
        st.write("本システムに記載されている情報は投資勧誘・提案を目的としたものではありません。")

        st.subheader("5. システム内通貨に関して")
        st.write("本システム内の資産に該当するシステム内通貨は、金銭または金銭的な価値を持つもの及び経済上の利益に該当するものと交換することはできません。")

        st.write("_______________________________________________________________________________________________________")
        st.button("戻る",on_click=lambda: change_page2(5))

    # システムの使い方
    def page2_5():
        st.title("システムの使い方")
        st.write("########################################################################################")

        st.write("_______________________________________________________________________________________________________")

        col7, col8, col9 = st.columns((1, 8, 1))
        with col7:
            for i in range(10):
                st.write("")

            st.button(" ＜ ",on_click=down_n)

        with col8:
            st.image(display_image(st.session_state.n), use_column_width=True)

        with col9:
            for i in range(10):
                st.write("")

            st.button(" ＞ ",on_click=up_n)
        st.write("_______________________________________________________________________________________________________")

        if st.session_state.n == 1:
            st.write("本システムでは、ユーザにデモトレードを実施してもらいそこから得られた運用成績や売買データなどからアドバイスを提供します。")

        if st.session_state.n == 2:
            st.write("ホーム画面にある各ボタンでは以下のことができます。")
            colored_text1 = f"<span style='font-size:15px'><span style='color:red'> 投資経験がない方はこちらへ </span> </span>　：　株式投資に関する基本的な情報を提供しています。"
            st.markdown(colored_text1, unsafe_allow_html=True)
            colored_text2 = f"<span style='font-size:15px'><span style='color:red'> このシミュレーションについて </span> </span>　：　"
            st.markdown(colored_text2, unsafe_allow_html=True)
            st.write("このシステムの使い方やシステムを使う上での注意事項などがわかります。")
            colored_text3 = f"<span style='font-size:15px'><span style='color:red'> これまでの実績 </span> </span>　：　これまでのシミュレーションの結果をここから見ることができます。"
            st.markdown(colored_text3, unsafe_allow_html=True)
            colored_text4 = f"<span style='font-size:15px'><span style='color:red'> 期間設定 </span> </span>　：　1ヶ月、3ヶ月、6ヶ月、12ヶ月の中からシミュレーションで使用するデータの期間を設定できます。"
            st.markdown(colored_text4, unsafe_allow_html=True)


        if st.session_state.n == 3:
            st.write("アカウントがない場合は、「アカウント設定」から作成できます。アカウントを設定及びレベルを選択するとシミュレーションを始められます。")
            for i in range(5):
                st.write("")


        if st.session_state.n == 4:
            st.write("シミュレーションが始まると日経平均株価及び購入可能銘柄が20個表示されます。")
            st.write("銘柄には、「企業名」「現在の株価」「直近20日間のチャート」の情報が表示されます。")
            st.write("（選択可能銘柄は2023年10月時点の日経平均採用銘柄からランダムに20銘柄が表示されます。）")
            for i in range(3):
                st.write("")

        if st.session_state.n == 5:
            st.write("各銘柄の詳細な情報を見るもしくは購入/売却がしたい場合は、株価を見るというボタンを押してください。")
            for i in range(5):
                st.write("")

        if st.session_state.n == 6:
            st.write("ここでは詳細な株価の情報を見ることができます。")
            for i in range(5):
                st.write("")

        if st.session_state.n == 7:
            st.write("下の方にスクロールしていくと、この銘柄の現在の株価がわかります。")
            st.write("また、「企業情報を見る」ボタンを押すとこの企業の業績など様々な情報を見ることができます。")
            for i in range(4):
                st.write("")

        if st.session_state.n == 8:
            st.write("さらに、ここでこの銘柄の購入/売却ができます。")
            for i in range(5):
                st.write("")

        if st.session_state.n == 9:
            st.write("購入画面では、購入株式数や購入根拠を選択できます。")
            st.write("購入根拠は後の分析で使用しますので、その銘柄を購入する際の素直な理由を選択してください。")
            for i in range(4):
                st.write("")

        if st.session_state.n == 10:
            st.write("購入した銘柄はサイドバーもしくは「保有株式へ」から見ることができます。")
            for i in range(5):
                st.write("")

        if st.session_state.n == 11:
            st.write("「1日進める」もしくは「１週間進める」を押すと時間が進み各銘柄の株価が更新されます。　（更新には少し時間がかかります）")
            for i in range(5):
                st.write("")

        if st.session_state.n == 12:
            st.write("時間はサイドバーから確認することができます。nowが現在時間、endが終了時間を表しており、nowがendを超えた時システムが終了し結果が表示されます。")
            for i in range(5):
                st.write("")

        st.write("_______________________________________________________________________________________________________")

        st.button("注意事項",on_click=lambda: change_page2("5_b"))

        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

    # 簡易結果画面表示
    def page2_6():
        st.title("結果画面")
        st.header(f"{st.session_state.acount_name}さんの結果")
        if st.session_state.benef_temp < 0:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:green'> {round(st.session_state.benef_temp,1)}円</span> </span>の損失を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)
        else:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:red'> +{round(st.session_state.benef_temp,1)}円</span> </span>の利益を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)

        # リスク許容度計算
        VOL_delta_list = []
        for i in range(0, len(st.session_state.buy_log_temp)):
            VOL_delta = st.session_state.buy_log_temp["当日のボラティリティ"][i] - st.session_state.buy_log_temp["当日のボラティリティ平均"][i]
            VOL_delta_list.append(VOL_delta)
            
        V_delta_mean = round(np.mean(VOL_delta_list), 2)
        Risk_tolerance = "中"
        if V_delta_mean < -1.0:
            Risk_tolerance = "低"
        
        if V_delta_mean > 1.0:
            Risk_tolerance = "高"

        # 売買期間計算
        tern_delta_list = []
        for i in range(0, len(st.session_state.sell_log_temp)):
            sell_day = st.session_state.sell_log_temp["年月"][i]
            sell_day = dt.datetime.strptime(sell_day, "%Y/%m/%d")
            c_name = st.session_state.sell_log_temp["企業名"][i]
            buy_day_df = st.session_state.buy_log_temp[st.session_state.buy_log_temp['企業名']==c_name]
            for j in range(0, len(buy_day_df)):
                buy_day = dt.datetime.strptime(buy_day_df.iloc[j]['年月'], "%Y/%m/%d")
                tern_delta = sell_day - buy_day
                tern_delta_day = tern_delta.days
                if tern_delta_day > 0:
                    tern_delta_list.append(tern_delta_day)
            
        tern_mean = round(np.mean(tern_delta_list), 2)
        tern = "短"                
        if tern_mean > 100:
            tern = "長"

        # 購入金額比率計算
        buy_rate_list = []
        for i in range(0, len(st.session_state.buy_log_temp)):
            buy_rate = st.session_state.buy_log_temp['購入金額比率'][i]
            buy_rate_list.append(buy_rate)

        buy_rate_mean = round(np.mean(buy_rate_list), 2)
        tendency = "集中投資型"
        if buy_rate_mean < 0.2:
            tendency = "分散投資型"

        behavioral_sell_data = {
            "取引回数": [len(st.session_state.sell_log_temp)], 
            "投資根拠": [st.session_state.sell_log_temp['売却根拠']],
            "運用成績": [st.session_state.sell_log_temp['利益']],  
            "取引株式数": [st.session_state.sell_log_temp['売却株式数']]
        }
        bdf = pd.DataFrame(behavioral_sell_data)

        # ユーザからの情報をデータフレームとして受け取る
        behavioral_buy_data = {
            "取引回数": [len(st.session_state.buy_log_temp)],  # 1年間の取引回数
            "投資根拠": [st.session_state.buy_log_temp['購入根拠']],
            "取引量": [st.session_state.buy_log_temp['購入金額']]
        }
        bdf2 = pd.DataFrame(behavioral_buy_data)

        trade_value = display_distribution(bdf2['取引量'][0])
        wield_grades = display_distribution(bdf['運用成績'][0])
        
        buy_reason_count, buy_reason_ratios = display_distribution2(bdf2['投資根拠'][0])
        sell_reason_count, sell_reason_ratios = display_distribution2(bdf['投資根拠'][0])

        #個人の性格情報から分類型にポイントを与える
        st.session_state.personal['性格']['新規性'] = st.session_state.Open
        st.session_state.personal['性格']['誠実性'] = st.session_state.Integrity
        st.session_state.personal['性格']['外交性'] = st.session_state.Diplomatic
        st.session_state.personal['性格']['協調性'] = st.session_state.Coordination
        st.session_state.personal['性格']['神経症傾向'] = st.session_state.Neuroticism

        classify_type_df = classify_action_type(st.session_state.personal, st.session_state.sell_log_temp, buy_reason_ratios, sell_reason_ratios, trade_value, wield_grades)

        # 最も高いポイントに分類
        action_type = classify_type_df[classify_type_df['分類型']==classify_type_df['分類型'].max()].index.values[0]

        target_action_type = st.session_state.action_type_advice[st.session_state.action_type_advice["行動型"]==action_type]
        target_action_type = target_action_type.reset_index(drop=True)

        feature = target_action_type["特徴"][0]
        weekness = target_action_type["欠点"][0]
        advice_text = target_action_type["アドバイス"][0]
        essay = target_action_type["文章"][0]

        st.write("_______________________________________________________________________________________________________")
        st.subheader("総合評価")
        # 枠線で囲む文章
        overall_advice_temp = st.session_state.overall_advice[st.session_state.overall_advice["リスク許容度"]==Risk_tolerance]
        overall_advice_temp = overall_advice_temp[overall_advice_temp["売買期間"]==tern]
        overall_advice_temp = overall_advice_temp[overall_advice_temp["投資傾向"]==tendency]
        overall_advice_temp = overall_advice_temp[overall_advice_temp["投資型"]==action_type]

        text = overall_advice_temp['アドバイス'].values[0]

        html_code = f"""
        <div style="
            border: 2px solid #000000;
            border-radius: 5px;
            padding: 10px;
        ">
            {text}
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)

        st.write("_______________________________________________________________________________________________________")

        st.subheader("全体の投資傾向について")

        st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">投資行動型</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:fantasy; color:blue; font-size: 24px;">{action_type}</p>', unsafe_allow_html=True)
        # st.write("特徴：")
        # st.write(feature)
        # st.write("欠点：")
        # st.write(weekness)
        # st.write("アドバイス：")
        # st.write(advice_text)
        html_code = f"""
        <div style="
            border: 2px solid #000000;
            border-radius: 5px;
            padding: 10px;
        ">
            {essay}
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)


        st.session_state.check = False

        st.write("################################################################################")
        check = st.checkbox("投資行動の情報を表示", value = st.session_state.check)
        if check:
            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">リスク許容度</p>', unsafe_allow_html=True)
            st.write(f"ボラティリティ誤差平均：{V_delta_mean}")
            st.write(f"リスク許容度：{Risk_tolerance}")

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">売買期間</p>', unsafe_allow_html=True)
            st.write(f"売買期間平均：{tern_mean}")
            st.write(f"売買期間：{tern}")

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">投資傾向</p>', unsafe_allow_html=True)
            st.write(f"購入金額比率平均：{buy_rate_mean}")
            st.write(f"投資傾向：{tendency}")
            
            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">取引量</p>', unsafe_allow_html=True)
            #各統計量表示
            st.dataframe(trade_value)
            # ヒストグラムを作成
            fig, ax = plt.subplots()
            ax.hist(bdf2['取引量'][0], bins=10, color='blue', alpha=0.7, edgecolor='black')
            ax.set_title('取引量のヒストグラム')
            ax.set_xlabel('１取引あたりの購入金額')
            ax.set_ylabel('count')
            # Streamlitでヒストグラムを表示
            st.pyplot(fig)

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">運用成績</p>', unsafe_allow_html=True)
            #各統計量表示
            st.dataframe(wield_grades)
            # ヒストグラムを作成
            fig, ax = plt.subplots()
            ax.hist(bdf['運用成績'][0], bins=10, color='blue', alpha=0.7, edgecolor='black')
            ax.set_title('利益のヒストグラム')
            ax.set_xlabel('利益')
            ax.set_ylabel('count')
            # Streamlitでヒストグラムを表示
            st.pyplot(fig)

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">購入根拠</p>', unsafe_allow_html=True)
            # 統計量を表示
            col_b1, col_b2 = st.columns((4, 6))
            with col_b1:
                st.write("\n各カテゴリのカウント:")
                st.write(buy_reason_count)

                st.write("\n各カテゴリの割合:")
                st.write(buy_reason_ratios)

            with col_b2:
                # 円グラフを作成
                fig, ax = plt.subplots()
                ax.pie(buy_reason_ratios, labels=buy_reason_ratios.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                # Streamlitに表示
                st.pyplot(fig)


            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">売却根拠</p>', unsafe_allow_html=True)
            # 統計量を表示
            col_s1, col_s2 = st.columns((4, 6))
            with col_s1:
                st.write("\n各カテゴリのカウント:")
                st.write(sell_reason_count)

                st.write("\n各カテゴリの割合:")
                st.write(sell_reason_ratios)

            with col_s2:
                # 円グラフを作成
                fig, ax = plt.subplots()
                ax.pie(sell_reason_ratios, labels=sell_reason_ratios.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

                # Streamlitに表示
                st.pyplot(fig)

        st.write("################################################################################")

        st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">行動経済学の指摘事項</p>', unsafe_allow_html=True)
        # if "advice_temp" not in st.session_state:
        #     st.session_state.advice_df_temp = advice(buy_reason_ratios, st.session_state.buy_log_temp, st.session_state.sell_log_temp)
        #     st.session_state.advice_temp = True

        if st.session_state.advice_df_temp.empty:
            st.write("特になし")
        else:
            for i in range(0, len(st.session_state.advice_df_temp)):
                # st.write(st.session_state.advice_df_temp['指摘事項'][i])
                st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 24px;">{st.session_state.advice_df_temp["指摘事項"][i]}</p>', unsafe_allow_html=True)

                target_BE = st.session_state.Behavioral_Economics[st.session_state.Behavioral_Economics['理論']==st.session_state.advice_df_temp['指摘事項'][i]]
                target_BE = target_BE.reset_index(drop=True)
                st.write(target_BE['内容'][0])
                # st.write("アドバイス")
                st.write(f"　→ {target_BE['アドバイス'][0]}")


        st.write("_______________________________________________________________________________________________________")

        st.subheader("各取引について")

        # st.write("各種投資行動の説明を書く")

        if "some_trade_advice_temp" not in st.session_state:
            st.session_state.trade_advice_df_temp = some_trade_advice(st.session_state.buy_log_temp, st.session_state.sell_log_temp)  
            st.session_state.some_trade_advice_temp = True


        #trade_advice_dfからグラフを作成する
        if st.session_state.trade_advice_df_temp.empty:
            st.write("指摘することはありません。")
        else:
            st.write("あなたの売買データから以下のバイアスが見つかりました。")
            for i in range(0,len(st.session_state.trade_advice_df_temp)):
                tgt_name = st.session_state.trade_advice_df_temp.iloc[i]['企業名']
                tgt_sell_day = st.session_state.sell_log_temp[st.session_state.sell_log_temp['企業名']==tgt_name]['年月'].iloc[-1]

                tgt_buy_day = st.session_state.buy_log_temp[st.session_state.buy_log_temp['企業名']==tgt_name]['年月'].iloc[-1]

                tgt_buy_day = dt.datetime.strptime(tgt_buy_day, "%Y/%m/%d")
                tgt_sell_day = dt.datetime.strptime(tgt_sell_day, "%Y/%m/%d")

                tgt_buy_day_temp = tgt_buy_day + dt.timedelta(days=-30)
                tgt_sell_day_temp = tgt_sell_day + dt.timedelta(days=30)

                index = st.session_state.c_master.loc[(st.session_state.c_master['企業名']==tgt_name)].index.values[0]
                #companiesからデータを抽出する
                target_company = st.session_state.loaded_companies[index]
                name = target_company.name
                rdf = target_company.rdf_all[tgt_buy_day_temp:tgt_sell_day_temp]

                # st.write(st.session_state.trade_advice_df_temp.iloc[i]['指摘事項'])
                st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 24px;">{st.session_state.trade_advice_df_temp.iloc[i]["指摘事項"]}</p>', unsafe_allow_html=True)

                target_BE2 = st.session_state.Behavioral_Economics[st.session_state.Behavioral_Economics['理論']==st.session_state.trade_advice_df_temp.iloc[i]['指摘事項']]
                target_BE2 = target_BE2.reset_index(drop=True)
                st.write(target_BE2['内容'][0])

                if st.session_state.trade_advice_df_temp.iloc[i]['指摘事項'] == '現在志向バイアス':
                    rdf_temp = rdf[tgt_sell_day:tgt_sell_day_temp]
                    max_date = rdf_temp[rdf_temp['Close']==rdf_temp['Close'].max()].index.values[0]
                    make_graph(name, rdf, buy_date=tgt_buy_day, sell_date=tgt_sell_day, now_kk_bool=True, max_date=max_date)
                else:
                    make_graph(name, rdf, buy_date=tgt_buy_day, sell_date=tgt_sell_day, now_kk_bool=True)

                tgt_benef = st.session_state.sell_log_temp[st.session_state.sell_log_temp['企業名']==tgt_name]['利益'].iloc[-1]

                if tgt_benef < 0:
                    colored_text = f"利益：　<span style='font-size:20px'><span style='color:green'> {round(tgt_benef,1)}円</span> </span>"
                    st.markdown(colored_text, unsafe_allow_html=True)
                else:
                    colored_text = f"利益：　<span style='font-size:20px'><span style='color:red'> +{round(tgt_benef,1)}円</span> </span>"
                    st.markdown(colored_text, unsafe_allow_html=True)

                html_code = f"""
                <div style="
                    border: 2px solid #000000;
                    border-radius: 5px;
                    padding: 10px;
                ">
                    {target_BE2['アドバイス'][0]}
                </div>
                """
                st.markdown(html_code, unsafe_allow_html=True)
                st.write("")

        st.write("_______________________________________________________________________________________________________")

        st.button("戻る", key='uniq_key_6',on_click=lambda: change_page2(2))

    # システムの評価アンケート
    def page2_7():
        st.title("システムの評価")
        st.write("このシステムの利用に関してアンケートを実施しております。ご協力お願いします。")


        # システムの満足度
        satisfaction_arrow = [
            "満足している",
            "まあまあ満足している",
            "あまり満足していない",
            "満足していない"
        ]
        st.session_state.satisfaction = st.radio("１. このシステムに満足していますか。", satisfaction_arrow)

        # システムの使いやすさ
        satisfaction_arrow2 = [
            "使いやすかった",
            "使いにくかった",
            "わからない"
        ]
        st.session_state.satisfaction2 = st.radio("２. システムは使いやすかったですか。", satisfaction_arrow2)

        # システムの正確性(分類)に関して
        accurate_classify_arrow = [
            "合っている",
            "合っていない",
            "わからない"
        ]
        st.session_state.accurate_classify = st.radio("３. 投資行動型の分類は合っていると思いますか。", accurate_classify_arrow)

        # システムの正確性(行動経済学)に関して
        accurate_instruction_arrow = [
            "合っている",
            "合っていない",
            "わからない"
        ]
        st.session_state.accurate_instruction = st.radio("４. 行動経済学の指摘は合っていると思いますか。", accurate_instruction_arrow)

        # システムの有用性に関して
        usefulness_arrow1 = [
            "思う",
            "思わない",
            "わからない"
        ]
        st.session_state.usefulness1 = st.radio("５. このシステムのアドバイスは役にたつと思いますか。", usefulness_arrow1)

        # システムの有用性に関して
        usefulness_arrow2 = [
            "思う",
            "思わない",
            "わからない"
        ]
        st.session_state.usefulness2 = st.radio("５. このシステムのアドバイスを今後役立てようと思いますか。", usefulness_arrow2)

        st.session_state.opinion = st.text_input("６. このシステムに関してご意見があればお聞かせください。", value=st.session_state.get("opinion", ""))

        st.button("システムの評価を送る",on_click=insert_survey_to_db)

    # データベースの確認
    def page2_99():
        # データベースの中身を確認する
        st.write("Trade_Simulate.db")
        # データベースに接続
        conn = sqlite3.connect('my_database.db')
        c = conn.cursor()

        c.execute('SELECT * FROM user_data ')
        # data = c.fetchone()

        for row in c:
            serialized_data = row[0]

            deserialized_data = pd.read_json(serialized_data)
            st.write(deserialized_data)

        # カーソルをクローズ（オプション）
        c.close()

        # データベースの接続をクローズ
        conn.close()


        st.write("_______________________________________________________________________________________________________")
        # データベースに接続
        conn = sqlite3.connect('my_database.db')

        # personal_info テーブルからすべてのデータを取得
        query = "SELECT * FROM personal_info"
        df = pd.read_sql_query(query, conn)
        query = "SELECT * FROM simulation_results"
        df2 = pd.read_sql_query(query, conn)
        query = "SELECT * FROM value_table"
        df3 = pd.read_sql_query(query, conn)

        # Streamlit でデータを表示
        st.write("my_database.db")
        st.write(df)
        st.write(df2)
        st.write(df3)

        # データベース接続をクローズ
        conn.close()

        st.write("_______________________________________________________________________________________________________")
        # データベースに接続
        conn = sqlite3.connect('my_database.db')

        # personal_info テーブルからすべてのデータを取得
        query = "SELECT * FROM survey_info"
        df3 = pd.read_sql_query(query, conn)

        # データベース接続をクローズ
        conn.close()

        st.write("survey.db")
        st.write(df3)

        st.write("_______________________________________________________________________________________________________")
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))


    if "page_id2" not in st.session_state:
        st.session_state.page_id2 = "start_system"

    if st.session_state.page_id2 == "start_system":
        start_system()

    if st.session_state.page_id2 == "page2_1":
        page2_1()

    if st.session_state.page_id2 == "page2_2":
        page2_2()

    if st.session_state.page_id2 == "page2_3":
        page2_3()

    if st.session_state.page_id2 == "page2_3_a":
        page2_3_a()

    if st.session_state.page_id2 == "page2_4":
        page2_4()

    if st.session_state.page_id2 == "page2_5":
        page2_5()

    if st.session_state.page_id2 == "page2_5_b":
        page2_5_b()

    if st.session_state.page_id2 == "page2_6":
        page2_6()

    if st.session_state.page_id2 == "page2_7":
        page2_7()

    if st.session_state.page_id2 == "page2_99":
        page2_99()

