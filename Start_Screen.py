import streamlit as st

import pandas as pd

import pandas_datareader.data as pdr

import mplfinance as mpf
import plotly.graph_objs as go
import datetime as dt

import numpy as np

import random

import json

import pickle
import io

from functools import partial

# import japanize_matplotlib

import matplotlib.pyplot as plt
from scipy import stats

import sqlite3

# 日本語フォントの設定
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'  # または 'Hiragino Kaku Gothic Pro'
# plt.rcParams['font.family'] = 'IPAexGothic'



#企業データを格納するクラス
class CompanyData:
    def __init__(self, code, name, rdf_all, _rdf):
        self.code = code
        self.name = name
        self.rdf_all = rdf_all
        self._rdf = _rdf

    def display(self):
        # ここに、このクラスのデータを表示するためのコードを追加できます
        print(f"Code: {self.code}")
        print(f"Name: {self.name}")
        print("RDF All:")
        print(self.rdf_all)
        print("RDF:")
        print(self._rdf)
        
    def to_list(self):
        return [self.code, self.name, self.rdf_all, self._rdf]

#シミュレーション結果を保存するクラス
class Simulation_Results:
    def __init__(self, dates, action_type, LEVEL, investment_result, buy_log, sell_log, Dividends, advice, trade_advice):
        self.dates = dates                             #実施日
        self.action_type = action_type                 #行動型
        self.LEVEL = LEVEL                             #保有金額
        self.investment_result = investment_result     #投資結果（利益）
        self.buy_log = buy_log                         #購入ログ
        self.sell_log = sell_log                       #売却ログ
        self.Dividends = Dividends                     #配当に関するデータ
        self.advice = advice                           #行動経済学の指摘事項
        self.trade_advice = trade_advice               #各取引の指摘事項
        self._observers = []

    def display(self):
        # ここに、このクラスのデータを表示するためのコードを追加できます
        st.write(f"実施日: {self.dates}")
        st.write(f"レベル：{self.LEVEL}")
        st.write(f"分類型: {self.action_type}")
        st.write(f"利益: {self.investment_result}")      
        st.write("全体のアドバイス:")
        st.write(self.advice)
        st.write("各取引のアドバイス:")
        st.write(self.trade_advice)

#____________________________初期値を代入する関数________________________________________
 #全体の期間を指定
all_range_start = dt.datetime(2020,9,1)
# all_range_end = dt.datetime(2022,3,31)
now_range = dt.datetime(2021,1,1)
# now = dt.datetime(2021,1,1)

i_max = 20

#変数の初期値
def main():
    # データ読み込み
    if "c_master" not in st.session_state:
        st.session_state.c_master = pd.read_csv('company_list3.csv')

    if "categorize" not in st.session_state:
        st.session_state.categorize = pd.read_csv('categorize.csv')

    if "action_type_advice" not in st.session_state:
        st.session_state.action_type_advice = pd.read_csv('action_type_advice.csv')

    if "Behavioral_Economics" not in st.session_state:
        st.session_state.Behavioral_Economics = pd.read_csv('Behavioral_Economics.csv')

    if "loaded_companies" not in st.session_state:
        with open("companies.pkl", "rb") as file:
            st.session_state.loaded_companies = pickle.load(file)

    if "account_created" not in st.session_state:
        st.session_state.account_created = False

    if "personal" not in st.session_state:
        st.session_state.personal = pd.DataFrame(columns=['性格'], index=['新規性', '誠実性', '外交性', '協調性', '神経症傾向'])

    if "now" not in st.session_state:
        st.session_state.now = dt.datetime(2021,1,4)

    if "all_range_end" not in st.session_state:
        st.session_state.all_range_end = dt.datetime(2021,2,1)  

    #乱数から企業名をリストに格納する
    if "chose_companies" not in st.session_state:
        st.session_state.chose_companies = []
    if "chose_companies_name_list" not in st.session_state:
        st.session_state.chose_companies_name_list = []

    #買付余力
    if "possess_money" not in st.session_state:
        st.session_state.possess_money = 10000000

    #所有株式のデータフレーム作成
    if "possess_KK_df" not in st.session_state:
        st.session_state.possess_KK_df = pd.DataFrame(columns=['企業名', '保有株式数', '現在の株価', '1株あたりの株価', '利益'])

    #買い・売りのログデータのデータフレーム作成
    if "buy_log" not in st.session_state:
        st.session_state.buy_log = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '購入株式数', '購入金額', '属性'])

    if "sell_log" not in st.session_state:
        st.session_state.sell_log = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '売却株式数', '利益', '属性'])

    #買い・売りの仮ログデータのデータフレーム作成
    if "buy_log_temp" not in st.session_state:
        st.session_state.buy_log_temp = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '購入株式数', '購入金額', '属性'])

    if "sell_log_temp" not in st.session_state:
        st.session_state.sell_log_temp = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '売却株式数', '利益', '属性'])

    if "Dividends_df" not in st.session_state:
        st.session_state.Dividends_df = pd.DataFrame(columns=['企業名', '属性', '配当金', "配当基準日", "実施"])

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

    if "trade_advice_df" not in st.session_state:
        st.session_state.trade_advice_df = pd.DataFrame(columns=['企業名', '指摘事項'])

    if "advice_df" not in st.session_state:
        st.session_state.advice_df = pd.DataFrame(columns=['指摘事項'])  

    if "trade_advice_df_temp" not in st.session_state:
        st.session_state.trade_advice_df_temp = pd.DataFrame(columns=['企業名', '指摘事項'])

    if "advice_df_temp" not in st.session_state:
        st.session_state.advice_df_temp = pd.DataFrame(columns=['指摘事項'])  

    # 個人情報のデータ
    if "personal_df" not in st.session_state:
        st.session_state.personal_df = pd.DataFrame(columns=['ユーザ名', '年齢', '性別', '投資経験の有無', '投資に関する知識の有無', '開放性', '誠実性', '外交性', '協調性', '神経症傾向'])  

    if "result" not in st.session_state:
        st.session_state.result = [] 

if "main_executed" not in st.session_state:
    main()  # main関数を実行
    st.session_state.main_executed = True

#_______________________________________________________________________________

#日経平均株価を取得
def get_NKX_data():
    NKX = pdr.DataReader("^NKX","stooq",all_range_start,st.session_state.now).sort_index()

    NKX["ma5"]   = NKX["Close"].rolling(window=5).mean()
    NKX["ma25"]  = NKX["Close"].rolling(window=25).mean()
    NKX["ma75"]  = NKX["Close"].rolling(window=75).mean()

    return NKX

#rdfからアクティブでないグラフを作る
@st.cache_data
def make_simple_graph(name, rdf):
    rdf = rdf[-20:]
    fig, axes = mpf.plot(rdf, type='candle',figsize=(6, 3), volume=True, returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=75)
    buf.seek(0)
    return buf

#rdfからグラフを表示する関数
def make_graph(name, rdf, buy_date=None, sell_date=None, now_kk_bool=False, max_date=False):
    #初期の表示期間指定
    rdf.index = pd.to_datetime(rdf.index)
    start_temp = rdf.tail(100)
    start = start_temp.index[0]
    end = rdf.index[-1]

    code = st.session_state.c_master[st.session_state.c_master['企業名']==name]['企業コード'].iloc[-1]

    layout = {
            "height":800,
            "title"  : { "text": "{} {}".format(code, name), "x": 0.5 }, 
            "xaxis" : { "rangeslider": { "visible": True }},
            #グラフの表示場所指定
            "yaxis1" : {"domain": [.30, 1.0], "title": "価格（円）", "side": "left", "tickformat": "," },
            #出来高の表示場所指定
            "yaxis2" : {"domain": [.00, .20], "title": "Volume", "side": "right"},
            "plot_bgcolor":"light blue"
              }

    data =  [
            go.Candlestick(yaxis="y1",x=rdf.index, open=rdf["Open"], high=rdf["High"], low=rdf["Low"], close=rdf["Close"], name="株価",
                           increasing_line_color="red", decreasing_line_color="gray"),
            #５日平均線追加
            go.Scatter(yaxis="y1",x=rdf.index, y=rdf["ma5"], name="5日平均線",
                line={ "color": "royalblue", "width":1.2}),
            #25日平均線追加
            go.Scatter(yaxis="y1",x=rdf.index, y=rdf["ma25"], name="25日平均線",
                line={ "color": "lightseagreen", "width":1.2}),
            #出来高追加
            go.Bar(yaxis="y2", x=rdf.index, y=rdf["Volume"], name="出来高",
                marker={ "color": "slategray"})
            ]

    if buy_date:
        data.append(
            go.Scatter(x=[buy_date, buy_date], y=[rdf["Low"].min(), rdf["High"].max()],
                       mode="lines", line=dict(color="red", width=2), name="購入日")
        )
            
    if sell_date:
        data.append(
            go.Scatter(x=[sell_date, sell_date], y=[rdf["Low"].min(), rdf["High"].max()],
                       mode="lines", line=dict(color="green", width=2), name="売却日")
        )

    if max_date:
        data.append(
            go.Scatter(x=[max_date, max_date], y=[rdf["Low"].min(), rdf["High"].max()],
                       mode="lines", line=dict(color="black", width=2), name="株価の最大値")
        )

    fig = go.Figure(data = data, layout = go.Layout(layout))

    # レイアウトを更新
    fig.update_layout(height=700, width=800, hovermode="x unified", dragmode="pan",margin_b=10)

    fig.update_xaxes(range=[start, end], tickformat='%m-%d-%Y')
    
    # fig.show()
    st.plotly_chart(fig)
    
    now_close = rdf['Close'][-1]
    pre_close = rdf['Close'][-2]
    now_pre = now_close - pre_close
    
    if now_kk_bool==False:
        st.metric(label='現在値', value=f'{round(now_close,1)} 円', delta=f'{round(now_pre,1)} 円', delta_color='inverse')

def create_chose_companies():
    chosen_indices = random.sample(range(222), i_max)
    
    for idx in chosen_indices:
        com_temp2 = st.session_state.loaded_companies[idx]
        st.session_state.chose_companies.append(com_temp2)
        st.session_state.chose_companies_name_list.append(com_temp2.name)

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
#_____________________________________________________________________________________________________________________________

#_______________________________1日、１週間進めるボタン作成_____________________________________________________
#nowを更新する関数
def add_next_day(num):

    next_day = st.session_state.now + dt.timedelta(days=num)

    # next_day が rdf_all に存在するか確認し、存在しない場合は次の日付に移動
    while next_day not in rdf_all.index:
        next_day += dt.timedelta(days=1)
    
    st.session_state.now = next_day

#_______________________________買い・売りボタンの設定___________________________________
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
        buy_log_temp = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '購入株式数', '購入金額', '属性'])
        buy_log_temp['企業名'] = [name]
        buy_log_temp['年月'] = [buy_now_str]
        buy_log_temp['購入根拠'] = [st.session_state.Rationale_for_purchase]
        buy_log_temp['購入株式数'] = [st.session_state.buy_num ]
        buy_amount = now_data_KK * st.session_state.buy_num
        buy_log_temp['購入金額'] = [buy_amount]
        buy_log_temp['属性'] = ['買い']
        st.session_state.buy_log = pd.concat([st.session_state.buy_log,buy_log_temp],ignore_index=True)
    
      
        st.session_state.possess_money -= purchace_amount

        change_page(2)
    
    #_______________________________________

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

#_______________________________評価用基本統計・ヒストグラムの設定___________________________________
# 取引量、利益の基本統計量を作成
def display_distribution(data):
    # 日本語フォントの設定
    font_path = "/Users/tatematsukenichirou/Desktop/my_page/卒研/研究（株価）/デモトレード/program/ipaexg00401/ipaexg.ttf"
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [font_path]

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
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'IPAexGothic'

    # 頻度を計算
    value_counts = data.value_counts()

    # 最頻値 (Mode)
    mode = value_counts.idxmax()
    mode_freq = value_counts.max()
    
    value_ratios = data.value_counts(normalize=True)

    # ユニークなカテゴリの数
    num_unique_categories = data.nunique()

    return value_counts, value_ratios

# 各取引のアドバイス生成#_____________________________________________________________________________________________________________________________
def some_trade_advice(buy_log, sell_log):
    trade_advice_df = pd.DataFrame(columns=['企業名', '指摘事項'])
    trade_advice_df_temp = pd.DataFrame(columns=['企業名', '指摘事項'])
    
    #損失回避性
    loss_df  = sell_log[sell_log['売却根拠'] == '利益確定売り']

    if loss_df.empty == False:
        trade_advice_df_temp['企業名'] = [loss_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['損失回避性']
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
    
    #アンカリング効果　buy_logから投資根拠が「安いと思ったから」のものを持ってくる
    anc_df  = buy_log[buy_log['購入根拠'] == '安いと思ったから']

    if anc_df.empty == False:
        trade_advice_df_temp['企業名'] = [anc_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['アンカリング効果']
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
    
    #感応度逓減性　ブレークイーブン効果などと一緒に表示させる
    # sensory_df1 = buy_log[buy_log["購入金額"] == buy_log["購入金額"].max()]
    # sensory_df2 = buy_log[buy_log["購入金額"] == buy_log["購入金額"].min()]
    # sensory_df = pd.concat([sensory_df1,sensory_df2],ignore_index=True)
    
    #現在思考バイアス　売った後1ヶ月以内に最大値があるなら指摘
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
            # 1つの行をDataFrameとして連結する
            temp_df = pd.DataFrame(pluss_benef_df.iloc[i]).transpose()
            present_oriented_df = pd.concat([present_oriented_df, temp_df], ignore_index=True)
        
    if present_oriented_df.empty == False:
        trade_advice_df_temp['企業名'] = [present_oriented_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['現在志向バイアス']
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

        # 連続してbuy_day_KKよりも小さい値が7日以上続く場所を見つける
        if count_true_consecutive(in_trade_rdf['Below_Buy_Day_KK']) >= 30:
            # この条件を満たす場合にDataFrameに追加する処理をここに書く
            minas_seqence_df = pd.concat([minas_seqence_df, minas_benef_df.iloc[i]], ignore_index = True)

    if minas_seqence_df.empty == False:
        # print('ブレークイーブン効果')
        # print('感応度逓減性')
        trade_advice_df_temp['企業名'] = [minas_seqence_df.iloc[0]['企業名']]
        trade_advice_df_temp['指摘事項'] = ['ブレークイーブン効果']
        trade_advice_df = pd.concat([trade_advice_df,trade_advice_df_temp], ignore_index=True)
        
    return trade_advice_df

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
            buy_time = buy_log[buy_log['企業名']==c_name]['年月']
            sell_time = sell_log[sell_log['企業名']==c_name]['年月']

            delta_time = sell_time - buy_time

            # 31日以内のものだけをフィルタリング
            within_31_days = delta_time[delta_time <= pd.Timedelta(days=31)]

            # 結果をデータフレームに追加
            trade_time_df = trade_time_df.append(within_31_days)

        trade_time_df = trade_time_df.reset_index(drop=True)
        short_trade_rate = len(trade_time_df) / len(sell_log)
        if short_trade_rate > 0.7:
            # st.write("代表性ヒューリスティクス")
            advice_df_temp['指摘事項'] = ['代表性ヒューリスティクス']
            advice_df = pd.concat([advice_df,advice_df_temp], ignore_index=True)

    return advice_df

# 分類型の関数作成#___________________________________________________________________________________________________________________________________________________________________________________________________________
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

    Conservative /= character_count
    Research /= character_count
    Positive /= character_count
    Emotion /= character_count
    Technical /= character_count
    
    
    #取引回数のデータから分類型にポイントを与える
    trade_count = len(sell_log)

    high_line = 100
    low_line = 50

    LEVEL = 1

    if LEVEL == 1:
        high_line /= 12
        low_line /= 12
    elif LEVEL == 2:
        high_line /= 4
        low_line /= 4
    elif LEVEL == 3:
        high_line /=2
        low_line /= 2
    elif LEVEL == 4:
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
            Emotion += (buy_reason_ratios[buy_reason] / 2)
        if buy_reason in buy_reason_Technical:
            Technical += (buy_reason_ratios[buy_reason])

    for sell_reason in sell_reason_ratios.index.values:
        if sell_reason in sell_reason_Conservative:
            Conservative += (sell_reason_ratios[sell_reason] / 2)
        if sell_reason in sell_reason_Research:
            Research += (sell_reason_ratios[sell_reason])
        if sell_reason in sell_reason_Positive:
            Positive += (sell_reason_ratios[sell_reason] / 2)
        if sell_reason in sell_reason_Emotion:
            Emotion += (sell_reason_ratios[sell_reason] / 2)        
        if sell_reason in sell_reason_Technical:
            Technical += (sell_reason_ratios[sell_reason])
            
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

# データベースに保存する関数作成_______________________________________________________________________________________________________________________________________________________________________________________________________
# データの挿入
def insert_data_to_db(private_data, result_data):
    # データベースに接続
    conn = sqlite3.connect('Trade_Simulate.db')
    c = conn.cursor()

    # テーブルの削除
    # c.execute("drop table user_data")

    # テーブルの作成（初回のみ）
    c.execute('CREATE TABLE IF NOT EXISTS user_data(private_data, result_data)')

    private_data_serialized = private_data.to_json()
    result_data_serialized = result_data.to_json()

    # データの挿入
    c.execute('INSERT INTO user_data (private_data, result_data) VALUES (?, ?)', (private_data_serialized, result_data_serialized))
    conn.commit()

    # カーソルをクローズ（オプション）
    c.close()

    # データベースの接続をクローズ
    conn.close()

# データの確認
def check_db():
        # データベースに接続
        conn = sqlite3.connect('Trade_Simulate.db')
        c = conn.cursor()

        c.execute('SELECT * FROM user_data ')
        # data = c.fetchone()

        for row in c:
            serialized_data = row[0]
            serialized_data_1 = row[1]

            deserialized_data = pd.read_json(serialized_data)
            st.write(deserialized_data)

            deserialized_data_1 = pd.read_json(serialized_data_1)
            st.write(deserialized_data_1)

        # カーソルをクローズ（オプション）
        c.close()

        # データベースの接続をクローズ
        conn.close()

# システム管理_______________________________________________________________________________________________________________________________________________________________________

def reset():
    if "main_executed" in st.session_state:
        del st.session_state.main_executed

    if "create_chose_companies_executed" in st.session_state:
        del st.session_state.create_chose_companies_executed
    
    if "target_company" in st.session_state:
        del st.session_state.target_company

    if "selected_company" in st.session_state:
        del st.session_state.selected_company

    if "chose_companies" in st.session_state:
        del st.session_state.chose_companies

    if "chose_companies_name_list" in st.session_state:
        del st.session_state.chose_companies_name_list

    if "now" in st.session_state:
        del st.session_state.now

    if "possess_money" in st.session_state:
        del st.session_state.possess_money

    if "possess_KK_df" in st.session_state:
        del st.session_state.possess_KK_df

    if "buy_log" in st.session_state:
        del st.session_state.buy_log

    if "sell_log" in st.session_state:
        del st.session_state.sell_log

    if "Dividends_df" in st.session_state:
        del st.session_state.Dividends_df

    if "page_id" in st.session_state:
        del st.session_state.page_id

    if "trade_advice_df" in st.session_state:
        del st.session_state.trade_advice_df

    if "advice_df" in st.session_state:
        del st.session_state.advice_df

    if "advice" in st.session_state:
        del st.session_state.advice

    if "some_trade_advice" in st.session_state:
        del st.session_state.some_trade_advice

    if "result_bool" in st.session_state:
        del st.session_state.result_bool
    
def end_sym():
    st.session_state.show_page = False

def start_sym(n):
    # 初めから始めるボタン
    if n == 1:
        reset()
        if st.session_state.account_created==True:
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

def change_page2(num, buy_log=None, sell_log=None, benef=None):
    if buy_log is not None and not buy_log.empty:
        st.session_state.buy_log_temp = buy_log
    if sell_log is not None and not sell_log.empty:
        st.session_state.sell_log_temp = sell_log
    if benef:
        st.session_state.benef_temp = benef

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
    if st.session_state.now == half_right_day:
        Dividends_df_temp = pd.DataFrame(columns=['企業名', '属性', '配当金', '配当基準日', '実施'])
        Dividends_df_temp['企業名'] = target_company_temp3['企業名']
        Dividends_df_temp['属性'] = '中間'
        Dividends_df_temp['配当金'] = target_company_temp3['中間配当'] * st.session_state.possess_KK_df['保有株式数'][i]
        Dividends_df_temp['配当基準日'] = half_base_day
        Dividends_df_temp['実施'] = False
        st.session_state.Dividends_df = pd.concat([st.session_state.Dividends_df, Dividends_df_temp], ignore_index=True)

    # nowが期末配当権利付き最終日の場合の処理
    if st.session_state.now == right_day:
        Dividends_df_temp = pd.DataFrame(columns=['企業名', '属性', '配当金', '配当基準日', '実施'])
        Dividends_df_temp['企業名'] = target_company_temp3['企業名']
        Dividends_df_temp['属性'] = '期末'
        Dividends_df_temp['配当金'] = target_company_temp3['期末配当'] * st.session_state.possess_KK_df['保有株式数'][i]
        Dividends_df_temp['配当基準日'] = base_day
        Dividends_df_temp['実施'] = False
        st.session_state.Dividends_df = pd.concat([st.session_state.Dividends_df, Dividends_df_temp], ignore_index=True)

if st.session_state.now in st.session_state.Dividends_df["配当基準日"]:
    st.session_state.possess_money += st.session_state.Dividends_df[st.session_state.Dividends_df["配当基準日"]==st.session_state.now]["配当金"].sum()
    st.session_state.Dividends_df['実施'] = True


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
        selected_page = st.selectbox("ページを選択してください", ["業績", "財務情報", "配当", "アナリスト分析"])

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

        elif selected_page == "アナリスト分析":
            st.write('アナリスト分析の内容')
            # ここに該当するデータや情報を表示


    # 結果画面
    def page5():
        st.title("結果画面")

        st.header(f"{st.session_state.acount_name}さんの結果")
        # 現在時刻を終了時間に合わせる
        st.session_state.now = st.session_state.all_range_end
        #保有資産に各保有株の現在値*株式数分を加算する
        possess_money = st.session_state.possess_money
        for i in range(0,len(st.session_state.possess_KK_df)):
            possess_money += st.session_state.possess_KK_df['現在の株価'][i] * st.session_state.possess_KK_df['保有株式数'][i]
        st.write(f"保有資産：{possess_money}")
        benef = possess_money - 10000000
        if benef < 0:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:green'> {round(benef,1)}円</span> </span>の損失を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)
        else:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:red'> +{round(benef,1)}円</span> </span>の利益を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)

        if not st.session_state.sell_log.empty:
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

            st.subheader("全体の投資傾向について")
            st.write("################################################################################")

            # st.write("投資傾向分類結果を書く")
            # st.write(f"運用成績：{benef}")

            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">投資行動型</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:fantasy; color:blue; font-size: 24px;">{action_type}</p>', unsafe_allow_html=True)
            st.write("特徴：")
            st.write(feature)
            st.write("欠点：")
            st.write(weekness)
            st.write("アドバイス：")
            st.write(advice_text)

            # checkの初期値を設定
            st.session_state.check = False

            check = st.checkbox("詳細な内容を表示", value = st.session_state.check)
            if check:
                st.write("################################################################################")
                st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">個人の性格</p>', unsafe_allow_html=True)
                st.write(st.session_state.personal)

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
                    st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 18px;">{st.session_state.advice_df["指摘事項"][i]}</p>', unsafe_allow_html=True)

                    target_BE = st.session_state.Behavioral_Economics[st.session_state.Behavioral_Economics['理論']==st.session_state.advice_df['指摘事項'][i]]
                    target_BE = target_BE.reset_index(drop=True)
                    st.write(target_BE['内容'][0])
                    # st.write("アドバイス")
                    st.write(f"　→ {target_BE['アドバイス'][0]}")

            st.write("################################################################################")

            st.subheader("各取引について")
            st.write("################################################################################")

            # st.write("各種投資行動の説明を書く")

            if "some_trade_advice" not in st.session_state:
                st.session_state.trade_advice_df = some_trade_advice(st.session_state.buy_log, st.session_state.sell_log)  
                st.session_state.some_trade_advice = True

            if len(st.session_state.trade_advice_df) == 0:
                st.write("アドバイスすることはありません")
            else:
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


                    st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 18px;">{st.session_state.trade_advice_df.iloc[i]["指摘事項"]}</p>', unsafe_allow_html=True)
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
                    st.write(target_BE2['アドバイス'][0])

            st.write("################################################################################")

            #現在時刻情報を取得
            dt_now = dt.datetime.now()
            dt_now_str = dt_now.strftime('%Y/%m/%d')

            Dividends_df_temp = st.session_state.Dividends_df.copy()
            for i in range(0, len(Dividends_df_temp)):
                Dividends_df_temp['配当基準日'].iloc[i] = Dividends_df_temp['配当基準日'].iloc[i].strftime('%Y/%m/%d')

            # ユーザからの情報をデータフレームとして受け取る
            Simulation_Result = {
                "実施日": [dt_now_str], 
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
            Simulation_Results_instance = Simulation_Results(dates=dt_now_str, action_type=action_type, LEVEL=st.session_state.level_id, investment_result=benef, buy_log=st.session_state.buy_log, sell_log=st.session_state.sell_log, Dividends=st.session_state.Dividends_df, advice=st.session_state.advice_df, trade_advice=st.session_state.trade_advice_df)

            if "result_bool" not in st.session_state:
                st.session_state.result.append(Simulation_Results_instance)
                #データベースに保存する
                insert_data_to_db(st.session_state.personal_df, Simulation_Results_df)
                st.session_state.result_bool = True


            # st.button("スタート画面に戻る",on_click=end_sym())
        else:
            st.write("株の取引が行われていないため結果を表示できません")

        if st.button("スタート画面に戻る"):
            st.session_state.show_page = False

    # 購入画面
    def page6():
        st.title("購入画面") 

        st.button("キャンセル",on_click=lambda: change_page(2))

        #購入株式数
        st.session_state.buy_num = st.slider("売却株式数", 100, 1000, st.session_state.get("buy_num", 100),step=100)

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
        st.title("テスト画面")
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

    # st.sidebar.button('シミュレーションを終了する',key='uniq_key_5', on_click=end_sym()):
    if st.sidebar.button('シミュレーションを終了する'):
        st.session_state.show_page = False

        
#_____________________________________________________________スタート画面_________________________________________________________________________________________________________________________________

else:
    # スタート画面
    def page2_1():
        st.title("スタート画面")

        st.write("########################################")

        st.write("ここにシミュレーションの説明を書く")

        st.write("########################################")

        col5, col6 = st.columns((5, 5))
        with col5:
            st.button("投資経験がない方はこちらへ",on_click=lambda: change_page2(4))

        with col6:
            st.button("このシミュレーションシステムについて", on_click=lambda: change_page2(5))

        st.button("これまでの実績", on_click=lambda: change_page2(2))
        st.button("アカウント設定", on_click=lambda: change_page2(3))

        st.session_state.level_id = st.selectbox(
            "レベルセレクト",
            ["LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4"],
            key="level-select"
        )

        if st.session_state.level_id == "LEVEL_1":
            st.session_state.all_range_end = dt.datetime(2021,2,1) 
            st.write("１ヶ月で利益を出してください")
        
        if st.session_state.level_id == "LEVEL_2":
            st.session_state.all_range_end = dt.datetime(2021,4,1) 
            st.write("３ヶ月で +5万円の利益を出してください")

        if st.session_state.level_id == "LEVEL_3":
            st.session_state.all_range_end = dt.datetime(2021,7,1) 
            st.write("６ヶ月で +10万円の利益を出してください")

        if st.session_state.level_id == "LEVEL_4":
            st.session_state.all_range_end = dt.datetime(2022,1,1) 
            st.write("１年で +20万円利益を出してください")

        st.button('シミュレーションをはじめから始める',on_click=lambda: start_sym(1))


        st.button('シミュレーションを続きから始める',on_click=lambda: start_sym(2))

    # 実績
    def page2_2():
        st.title("実績")

        st.write("########################################################################################")

        for i, result in enumerate(st.session_state.result, start=1):
            st.subheader(f"第{i}回")
            result.display()
            st.button("結果を見る", key=f"result_{i}", on_click=partial(change_page2, 6, result.buy_log, result.sell_log, result.investment_result))

            st.write("########################################################################################")

        if "advice_temp" in st.session_state:
            del st.session_state.advice_temp

        if "some_trade_advice_temp" in st.session_state:
            del st.session_state.some_trade_advice_temp

        # データベースの中身を確認する
        # check_db()
            
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

    def create_acount():
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

    # アカウント画面
    def page2_3():
        st.title("アカウント情報")
        st.write("########################################")

        st.write("ここに個人情報の利用について書く")

        st.write("########################################")


        if not st.session_state.account_created:
            if st.button("アカウントを作成する"):
                st.session_state.account_created = True

        if st.session_state.account_created:
            st.session_state.acount_name = st.text_input("アカウント名を入力してください", value=st.session_state.get("acount_name", ""))
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


            st.write("以下のURLから個人の性格についてのテストを実施して情報を入力してください。")
            st.write("https://commutest.com/bigfive")
            st.session_state.Open = st.slider("開放性", 0, 6, st.session_state.get("Open", 6))
            st.session_state.Integrity = st.slider("誠実性", 0, 6, st.session_state.get("Integrity", 6))
            st.session_state.Diplomatic = st.slider("外交性", 0, 6, st.session_state.get("Diplomatic", 6))
            st.session_state.Coordination = st.slider("協調性", 0, 6, st.session_state.get("Coordination", 6))
            st.session_state.Neuroticism = st.slider("神経症傾向", 0, 6, st.session_state.get("Neuroticism", 6))

            st.button("スタート画面に戻る",on_click=lambda: create_acount())

        else:
            st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

   # 投資について 
    def page2_4():
        st.title("投資について")

        # st.write("########################################")

        # st.write("ここに投資についての説明を書く")

        # st.write("########################################")


        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))

    # シミュレーションについて
    def page2_5():
        st.title("このシミュレーションについて")

        st.write("########################################")

        st.write("ここにシミュレーションの説明を書く")

        st.write("########################################")

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

        classify_type_df = classify_action_type(st.session_state.personal, st.session_state.sell_log_temp, buy_reason_ratios, sell_reason_ratios, trade_value, wield_grades)

        # 最も高いポイントに分類
        action_type = classify_type_df[classify_type_df['分類型']==classify_type_df['分類型'].max()].index.values[0]

        target_action_type = st.session_state.action_type_advice[st.session_state.action_type_advice["行動型"]==action_type]
        target_action_type = target_action_type.reset_index(drop=True)

        feature = target_action_type["特徴"][0]
        weekness = target_action_type["欠点"][0]
        advice_text = target_action_type["アドバイス"][0]


        st.subheader("全体の投資傾向について")
        st.write("################################################################################")

        # st.write("投資傾向分類結果を書く")
        # st.write(f"運用成績：{st.session_state.benef_temp}")

        # action_type = "テクニカル分析型"

        st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">投資行動型</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:fantasy; color:blue; font-size: 24px;">{action_type}</p>', unsafe_allow_html=True)
        st.write("特徴：")
        st.write(feature)
        st.write("欠点：")
        st.write(weekness)
        st.write("アドバイス：")
        st.write(advice_text)
        st.session_state.check = False

        check = st.checkbox("詳細な内容を表示", value = st.session_state.check)
        if check:
            st.write("################################################################################")
            st.markdown('<p style="font-family:fantasy; color:salmon; font-size: 24px;">個人の性格</p>', unsafe_allow_html=True)
            st.write(st.session_state.personal)
            
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
        if "advice_temp" not in st.session_state:
            st.session_state.advice_df_temp = advice(buy_reason_ratios, st.session_state.buy_log_temp, st.session_state.sell_log_temp)
            st.session_state.advice_temp = True

        if st.session_state.advice_df_temp.empty:
            st.write("特になし")
        else:
            for i in range(0, len(st.session_state.advice_df_temp)):
                # st.write(st.session_state.advice_df_temp['指摘事項'][i])
                st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 18px;">{st.session_state.advice_df_temp["指摘事項"][i]}</p>', unsafe_allow_html=True)

                target_BE = st.session_state.Behavioral_Economics[st.session_state.Behavioral_Economics['理論']==st.session_state.advice_df_temp['指摘事項'][i]]
                target_BE = target_BE.reset_index(drop=True)
                st.write(target_BE['内容'][0])
                # st.write("アドバイス")
                st.write(f"　→ {target_BE['アドバイス'][0]}")


        st.write("################################################################################")

        st.subheader("各取引について")
        st.write("################################################################################")

        # st.write("各種投資行動の説明を書く")

        if "some_trade_advice_temp" not in st.session_state:
            st.session_state.trade_advice_df_temp = some_trade_advice(st.session_state.buy_log_temp, st.session_state.sell_log_temp)  
            st.session_state.some_trade_advice_temp = True


        #trade_advice_dfからグラフを作成する
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
            st.markdown(f'<p style="font-family:fantasy; color:green; font-size: 18px;">{st.session_state.trade_advice_df_temp.iloc[i]["指摘事項"]}</p>', unsafe_allow_html=True)

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

            st.write(f"{target_BE2['アドバイス'][0]}")

        st.write("################################################################################")

        st.button("戻る", key='uniq_key_6',on_click=lambda: change_page2(2))

    if "page_id2" not in st.session_state:
        st.session_state.page_id2 = "page2_1"

    if st.session_state.page_id2 == "page2_1":
        page2_1()

    if st.session_state.page_id2 == "page2_2":
        page2_2()

    if st.session_state.page_id2 == "page2_3":
        page2_3()

    if st.session_state.page_id2 == "page2_4":
        page2_4()

    if st.session_state.page_id2 == "page2_5":
        page2_5()

    if st.session_state.page_id2 == "page2_6":
        page2_6()

