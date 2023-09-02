import streamlit as st

import pandas as pd

import pandas_datareader.data as pdr

import mplfinance as mpf
import plotly.graph_objs as go
import datetime as dt

import schedule
from time import sleep
import threading

import numpy as np

import random

import pickle
import io

from functools import partial

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

#____________________________初期値を代入する関数________________________________________
 #全体の期間を指定
all_range_start = dt.datetime(2020,9,1)
# all_range_end = dt.datetime(2021,3,31)
now_range = dt.datetime(2021,1,1)
# now = dt.datetime(2021,1,1)

i_max = 20

s = 2
hread = None  # スレッドを格納するための変数


#再定義したくない変数
def main():
    # データ読み込み
    if "c_master" not in st.session_state:
        st.session_state.c_master = pd.read_csv('company_list2.csv')

    if "loaded_companies" not in st.session_state:
        with open("companies.pkl", "rb") as file:
            st.session_state.loaded_companies = pickle.load(file)

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
        st.session_state.buy_log = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '属性'])

    if "sell_log" not in st.session_state:
        st.session_state.sell_log = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '利益', '属性'])

    if "sales_df" not in st.session_state:
        st.session_state.sales_df = pd.DataFrame(columns=['売上','営業利益','当期利益','基本的1株当たりの当期利益'],index=['2018','2019','2020','2021'])

    if "CF_df" not in st.session_state:
        st.session_state.CF_df = pd.DataFrame(columns=['営業CF','投資CF','財務CF'],index=['2020','2021'])

    if "FS_df" not in st.session_state:
        st.session_state.FS_df = pd.DataFrame(columns=['2020','2021'],index=['1株当たりの当期純利益','PER','1株当たりの純資産','PBR','ROA','ROE','自己資本比率','配当性向'])

    if "system_end" not in st.session_state:
        st.session_state.system_end = False

if "main_executed" not in st.session_state:
    main()  # main関数を実行
    st.session_state.main_executed = True

#_______________________________________________________________________________

# def get_stock_data(code):
#     df = pdr.DataReader("{}.JP".format(code),"stooq").sort_index()
#     df = df['2020-09-01':'2022-03-01']
#     return df

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
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


#rdfからグラフを表示する関数
def make_graph(name, rdf, buy_date=None, sell_date=None, now_kk_bool=False):
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

    fig = go.Figure(data = data, layout = go.Layout(layout))

    # レイアウトを更新
    fig.update_layout(height=700, width=800, hovermode="x unified", dragmode="pan",margin_b=10)

    fig.update_xaxes(range=[start, end], tickformat='%m-%d-%Y')
    
    # fig.show()
    st.plotly_chart(fig)
    
    now_close = rdf['Close'][-1]
    pre_close = rdf['Close'][-2]
    now_pre = now_close - pre_close
    
    # if now_pre < 0:
    #     colored_text = f"<span style='font-size:40px'><span style='color:gray'> {round(now_close,1)}</span> ( <span style='color:gray'>{round(now_pre,1)}</span> ) </span> "
    #     st.markdown(colored_text, unsafe_allow_html=True)
    # else:
    #     colored_text = f"<span style='font-size:40px'><span style='color:red'> {round(now_close,1)}</span> ( <span style='color:red'>{round(now_pre,1)}</span> ) </span> "
    #     st.markdown(colored_text, unsafe_allow_html=True)
    if now_kk_bool==False:
        st.metric(label='現在値', value=f'{round(now_close,1)} 円', delta=f'{round(now_pre,1)} 円', delta_color='inverse')

def create_chose_companies():
    for i in range(0,i_max):
        random_num = random.randrange(222)
        com_temp2 = st.session_state.loaded_companies[random_num]
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



#________________________________再生・停止ボタンに関するコード___________________________________________
##############################################################################################################################
# #nowを更新する関数
# def add_next_available_day(rdf):
# #     Parameters:
# #     - rdf: 現在のデータフレーム
# #     - rdf_all: 全体のデータフレーム
# #     - current_day: 現在の日付
# #     - end_date: 全体の期間の終了日

#     # clear_output(True)
#     # display(hbox, output)
    
#     global now
#     now = now + dt.timedelta(days=1)

#     # next_day が rdf_all に存在するか確認し、存在しない場合は次の日付に移動
#     while now not in rdf_all.index:
#         now += dt.timedelta(days=1)

#         # 全体の期間を超えた場合はループを終了
#         if now > all_range_end:
#             print("シミュレーション終了")
#             return rdf  # 変更なしの rdf を返す

#     # now を更新
#     subject.now = now

#     # print(f"now = {now}")
#     st.write(f"now = {now}")

#     # 存在する日付が見つかったら、その行を rdf に追加
#     next_day_data = rdf_all.loc[[now]]
#     subject.rdf = pd.concat([subject.rdf, next_day_data])
    
#     #####################
#     st.session_state.target_company._rdf = subject.rdf
#     ########################
    
#     return subject.rdf
##############################################################################################################################
##############################################################################################################################
# #再生ボタン関数
# def play(c):
#     global Bool, thread, s
#     s = 2
#     Bool = True
    
#     # 既にスレッドが存在しない、またはスレッドが停止している場合に新しいスレッドを作成
#     if thread is None or not thread.is_alive():
#         thread = threading.Thread(target=run_schedule)
#         thread.start()

# #停止ボタン関数
# def stop(d):
#     global Bool
#     Bool = False

# #早送りボタン関数
# def play_fast(e):
#     global Bool, thread, s
    
#     if s > 1:
#         s /= 2
#     else:
#         s = 1

#     Bool = True
    
#     # 既にスレッドが存在しない、またはスレッドが停止している場合に新しいスレッドを作成
#     if thread is None or not thread.is_alive():
#         thread = threading.Thread(target=run_schedule)
#         thread.start()


# # グローバル変数の初期化
# Bool = False

# def reset_schedule():
#     for job in list(schedule.get_jobs()):  # スケジュールされたすべてのジョブを取得
#         schedule.cancel_job(job)  # ジョブをキャンセル（削除）

# # スケジュールをリセット
# reset_schedule()
    
# #　s秒ごとにデータを更新
# schedule.every(s).seconds.do(add_next_available_day,rdf = rdf)

# # イベント実行
# # 新しい関数を定義して、その中で while ループを実行
# def run_schedule():
#     global Bool
#     while True:
#         if Bool:
#             schedule.run_pending()
#             sleep(1)
#         else:
#             print("停止中")
#             break
            
# # 新しいスレッドで関数を実行
# thread = threading.Thread(target=run_schedule)
# #_____________________________________________________________________________________________________________________________

# #　rdfのデータが更新されるたびにグラフを再描画する
# # Observer パターンの実装
# class DataSubject:
#     def __init__(self, rdf_initial):
#         self._observers = []
#         self._now = None
#         self.rdf = rdf_initial

#     def attach(self, observer):
#         if observer not in self._observers:
#             self._observers.append(observer)

#     def detach(self, observer):
#         self._observers.remove(observer)

#     def notify(self):
#         for observer in self._observers:
#             observer.update(self, rdf)

#     @property
#     def now(self):
#         return self._now

#     @now.setter
#     def now(self, value):
#         self._now = value
#         self.notify()

# #　データ更新時の対応
# class DataObserver:
#     def update(self, subject, rdf):
#         #rdfを更新
#         next_day_data = rdf_all.loc[[subject.now]]
#         subject.rdf = pd.concat([subject.rdf, next_day_data])
        
#         #グラフを再描画
#         make_graph(name, subject.rdf)
        
#         return subject.rdf

# # Observer インスタンスの作成と登録
# subject = DataSubject(rdf_initial=rdf)
# observer = DataObserver()
# subject.attach(observer)
##############################################################################################################################

#_______________________________買い・売りボタンの設定___________________________________
#買いボタン
def buy(name, rdf_all):
    # clear_output(True)
    # #買い・売りボタン表示
    # display(hbox2, output)
    
    # global possess_money, possess_KK_df
    
    #保有株式数と１株あたりの株価の初期値
    possess_KK_num = 0
    possess_KK_avg = 0
    benefit = 0
    
    #buy_num = input("何株購入しますか？")
    # buy_num = st.session_state.buy_num
    
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
            
        #データログにデータを追加
        buy_log_temp = pd.DataFrame(columns=['企業名', '年月', '購入根拠', '属性'])
        buy_log_temp['企業名'] = [name]
        buy_log_temp['年月'] = [st.session_state.now]
        buy_log_temp['購入根拠'] = [st.session_state.Rationale_for_purchase]
        buy_log_temp['属性'] = ['買い']
        st.session_state.buy_log = pd.concat([st.session_state.buy_log,buy_log_temp],ignore_index=True)
    
      
        st.session_state.possess_money -= purchace_amount

        
    # if st.session_state.possess_KK_df.empty == True:
    #     print("あなたは現在株を所有していません。")
    # else:
    #     print("現在保有している株式")
    #     display(st.session_state.possess_KK_df)
    
    # print(f"買付余力：{round(possess_money)}")
        #possess_KK_list[0].display()
    
    #_______________________________________

def sell(name, rdf_all):
    # clear_output(True)
    # display(hbox2, output)
    
    # global possess_money, possess_KK_df

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
            
                    
            sell_log_temp = pd.DataFrame(columns=['企業名', '年月', '売却根拠', '利益','属性'])
            sell_log_temp['企業名'] = [name]
            sell_log_temp['年月'] = [st.session_state.now]
            sell_log_temp['売却根拠'] = [st.session_state.basis_for_sale]
            sell_log_temp['利益'] = [benefit]
            sell_log_temp['属性'] = ['売り']
            st.session_state.sell_log = pd.concat([st.session_state.sell_log,sell_log_temp],ignore_index=True)

                
            st.session_state.possess_money += now_data_KK * st.session_state.sell_num

    #_______________________________________
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

    if "all_range_end" in st.session_state:
        del st.session_state.all_range_end

    if "possess_money" in st.session_state:
        del st.session_state.possess_money

    if "possess_KK_df" in st.session_state:
        del st.session_state.possess_KK_df

    if "buy_log" in st.session_state:
        del st.session_state.buy_log

    if "sell_log" in st.session_state:
        del st.session_state.sell_log

    if "system_end" in st.session_state:
        del st.session_state.system_end

    if "page_id" in st.session_state:
        del st.session_state.page_id
    

def change_page_and_chose_company(num, name):
    st.session_state.selected_company = name
    st.session_state.page_id = f"page{num}"

def change_page(num):
    st.session_state.page_id = f"page{num}"

# def change_page2(num):
#     st.session_state["page-select2"] = f"page2_{num}"

def change_page2(num):
    st.session_state.page_id2 = f"page2_{num}"
#_____________________________________________________________________________________________________________________________

# 全体の期間を超えた場合はループを終了
if st.session_state.now > st.session_state.all_range_end:
    st.session_state.system_end = True
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

# if "reset" not in st.session_state:
#     st.session_state.reset = False

if st.session_state.show_page:

    def page1():
        st.title("選択可能銘柄一覧")

        col1, col2 = st.columns((5, 5))
        with col1:
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
            st.button("一日進める", on_click=lambda: add_next_day(1))
        with col2:
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
            if not rdf_all_temp.empty:
                now_KK = rdf_all_temp['Close'][-1]
                pre_KK = rdf_all_temp['Close'][-2]
                now_pre = now_KK - pre_KK
                companies_data.append((company.name, now_KK, now_pre, rdf_all_temp))

        # 2. ループの最適化
        for i, (name, now_KK, now_pre, rdf) in enumerate(companies_data):
            col3, col4, col5 = st.columns((2, 2, 4))
            with col3:
                st.subheader(name)
                st.button("株価を見る", key=f"chose_companies_key_{i}", on_click=partial(change_page_and_chose_company, 2, name))
            with col4:
                st.metric(label='現在値', value=f'{round(now_KK,1)} 円', delta=f'{round(now_pre,1)} 円', delta_color='inverse')
            with col5:
                buf = make_simple_graph(name, rdf)
                st.image(buf)


        st.button("保有株式へ", on_click=lambda: change_page(3))


    def page2():
        st.title("トレード画面")

        # col1, col2 = st.columns((5, 5))
        # with col1:
        #     # st.markdown('<div id="button1"></div>', unsafe_allow_html=True)
        #     button_css1 = f"""
        #     <style>
        #         div.stButton > button:first-child  {{
        #         color        : white               ;
        #         width        : 100%                ;
        #         font-weight  : bold                ;/* 文字：太字                   */
        #         border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
        #         border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
        #         background   : #a9a9a9             ;/* 背景色：薄いグレー            */
        #     }}
        #     </style>
        #     """
        #     st.markdown(button_css1, unsafe_allow_html=True)
        #     action = st.button("一日進める", on_click=lambda: add_next_day(1))

        # with col2:
        #     # st.markdown('<div id="button2"></div>', unsafe_allow_html=True)
        #     button_css2 = f"""
        #     <style>
        #         div.stButton > button:first-child  {{
        #         color        : white               ;
        #         padding      : 14px 20px           ;
        #         margin       : 8px 0               ;
        #         width        : 100%                ;
        #         cursor       : pointer             ;
        #         font-weight  : bold                ;/* 文字：太字                   */
        #         border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
        #         border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
        #         background   : #a9a9a9            ;/* 背景色：薄いグレー            */
        #     }}
        #     </style>
        #     """
        #     st.markdown(button_css2, unsafe_allow_html=True)
        #     action2 = st.button("一週間進める", on_click=lambda: add_next_day(7))

        # st.write(f"now = {st.session_state.now}")

        st.write("_______________________________________________________________________________________________________")

        st.button("選択可能銘柄一覧へ",on_click = lambda: change_page(1))

        #選択可能銘柄の企業名一覧を作成
        # st.session_state.selected_company = st.selectbox("銘柄", st.session_state.chose_companies_name_list)

        st.write("_______________________________________________________________________________________________________")

        # if "target_company" not in st.session_state: 
        #     st.session_state.target_company = st.session_state.create_chose_companies[0]

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
            st.markdown('<div id="button3"></div>', unsafe_allow_html=True)
            button_css3 = f"""
            <style>
                div.stButton > button:first-child  {{
                color        : white               ;
                padding      : 14px 20px           ;
                margin       : 8px 0               ;
                width        : 100%                ;
                cursor       : pointer             ;
                font-weight  : bold                ;/* 文字：太字                   */
                border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
                border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
                background   : #a9a9a9            ;/* 背景色：薄いグレー            */
            }}
            </style>
            """
            st.markdown(button_css3, unsafe_allow_html=True)
            action = st.button("買う", on_click=lambda: change_page(6))
        with col4:
            st.markdown('<div id="button4"></div>', unsafe_allow_html=True)
            button_css4 = f"""
            <style>
                #button4 + div.stButton > button:first-child  {{
                color        : white               ;
                width        : 100%   !important             ;
                font-weight  : bold                ;/* 文字：太字                   */
                border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
                border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
                background   : #2e8b57            ;/* 背景色：薄いグレー            */
                }}
            </style>
            """
            st.markdown(button_css4, unsafe_allow_html=True)
            action = st.button("売る", on_click=lambda: change_page(7))

        st.write("_______________________________________________________________________________________________________")

        st.button("企業情報を見る",on_click = lambda: change_page(4))

        st.write("_______________________________________________________________________________________________________")
        if st.session_state.possess_KK_df.empty == True:
            st.write("あなたは現在株を所有していません。") 
        else:
            st.write("現在保有している株式") 
            st.dataframe(st.session_state.possess_KK_df)


    def page3():
        st.title("保有株式")

        if st.session_state.possess_KK_df.empty == True:
            st.write("あなたは現在株を所有していません。") 
        else:
            st.write("現在保有している株式") 
            st.dataframe(st.session_state.possess_KK_df)

        st.subheader(f"買付余力：{round(st.session_state.possess_money)}")

        st.button("選択可能銘柄一覧へ戻る",on_click=lambda: change_page(1))


    def page4():
        st.title("企業情報")
        st.write("_______________________________________________________________________________________________________")
        st.button("トレード画面へ戻る",on_click=lambda: change_page(2))
        st.write("_______________________________________________________________________________________________________")

        name = st.session_state.target_company.name
        rdf_all = st.session_state.target_company.rdf_all

        now_KK = rdf_all['Close'][-1]

        index = st.session_state.c_master[st.session_state.c_master['企業名'] == name].index.values[0]

        for i in range(2018,2022):
            st.session_state.sales_df['売上'].loc[f'{i}'] = st.session_state.c_master[f'{i}売上'][index]
            st.session_state.sales_df['営業利益'].loc[f'{i}'] = st.session_state.c_master[f'{i}営業利益'][index]
            st.session_state.sales_df['当期利益'].loc[f'{i}'] = st.session_state.c_master[f'{i}当期利益'][index]
            st.session_state.sales_df['基本的1株当たりの当期利益'].loc[f'{i}'] = st.session_state.c_master[f'{i}基本的1株当たり当期利益'][index]

        for i in range(2020,2022):
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
            st.session_state.FS_df[f'{i}'].loc['配当性向'] = st.session_state.c_master[f'{i}配当性向'][index]

        st.subheader(f"{name}の企業情報")

        st.write('売上推移')
        st.dataframe(st.session_state.sales_df)

        st.write('キャッシュフロー')
        st.dataframe(st.session_state.CF_df)

        st.write('財務諸表')
        st.dataframe(st.session_state.FS_df)




    def page5():
        st.title("結果画面")
        st.header(f"{st.session_state.acount_name}さんの結果")
        # 現在時刻を終了時間に合わせる
        st.session_state.now = st.session_state.all_range_end
        #保有資産に各保有株の現在値*株式数分を加算する
        possess_money = 0
        for i in range(0,len(st.session_state.possess_KK_df)):
            possess_money += st.session_state.possess_money + st.session_state.possess_KK_df['現在の株価'][i] * st.session_state.possess_KK_df['保有株式数'][i]
        st.write(f"保有資産：{possess_money}")
        benef = possess_money - 10000000
        if benef < 0:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:green'> {round(benef,1)}円</span> </span>の損失を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)
        else:
            colored_text = f"あなたは　<span style='font-size:30px'><span style='color:red'> +{round(benef,1)}円</span> </span>の利益を出しました。"
            st.markdown(colored_text, unsafe_allow_html=True)
        # st.write(f"あなたは　{benef}の利益を出しました")

        max_row = st.session_state.sell_log[st.session_state.sell_log['利益']==st.session_state.sell_log['利益'].max()]
        min_row = st.session_state.sell_log[st.session_state.sell_log['利益']==st.session_state.sell_log['利益'].min()]
        new_log_df = pd.concat([max_row,min_row],ignore_index=True)
        
        st.subheader("全体の投資傾向について")
        st.write("########################################")

        st.write("投資傾向分類結果を書く")
        st.write(f"開放性：{st.session_state.Open}")
        st.write(f"誠実性：{st.session_state.Integrity}")
        st.write(f"外交性：{st.session_state.Diplomatic}")
        st.write(f"協調性：{st.session_state.Coordination}")
        st.write(f"神経症傾向：{st.session_state.Neuroticism}")

        st.write(f"運用成績：{benef}")
        # st.write(f"主な投資根拠：{}")

        st.write("########################################")

        st.subheader("改善案")
        st.write("########################################")

        st.write("各種投資行動の説明を書く")
        #new_log_dfからグラフを作成する
        for i in range(0,len(new_log_df)):
            tgt = new_log_df.iloc[i]
            tgt_name = tgt['企業名']
            tgt_sell_day = tgt['年月']

            tgt_buy_day = st.session_state.buy_log[st.session_state.buy_log['企業名']==tgt_name]['年月'].iloc[-1]

            tgt_buy_day_temp = tgt_buy_day + dt.timedelta(days=-30)
            tgt_sell_day_temp = tgt_sell_day + dt.timedelta(days=30)

            index = st.session_state.c_master.loc[(st.session_state.c_master['企業名']==tgt_name)].index.values[0]
            #companiesからデータを抽出する
            target_company = st.session_state.loaded_companies[index]
            name = target_company.name
            rdf = target_company.rdf_all[tgt_buy_day_temp:tgt_sell_day_temp]

            make_graph(name, rdf, buy_date=tgt_buy_day, sell_date=tgt_sell_day, now_kk_bool=True)

        st.write("########################################")

        if st.button("スタート画面に戻る"):
            st.session_state.show_page = False


    def page6():
        st.title("購入画面") 

        st.button("キャンセル",on_click=lambda: change_page(2))

        #購入株式数
        st.session_state.buy_num = st.slider("売却株式数", 100, 1000, st.session_state.get("buy_num", 100),step=100)

        if "Rationale_for_purchase" not in st.session_state:
            st.session_state.Rationale_for_purchase = "指定なし"

        buy_reason_arrow = [
            "企業の業績がいいから",
            "株価が急に動いたから",
            "財務データがいいから"
            "直感"
        ]

        #購入根拠
        st.session_state.Rationale_for_purchase = st.radio("購入根拠", buy_reason_arrow)

        if st.button("購入する"):
            buy(name, rdf_all)
            change_page(2)

    def page7():
        st.title("売却画面") 

        st.button("キャンセル",on_click=lambda: change_page(2))

        #売却株式数
        st.session_state.sell_num = st.slider("売却株式数", 100, 1000, st.session_state.get("sell_num", 100),step=100)

        if "basis_for_sale" not in st.session_state:
            st.session_state.basis_for_sale = "指定なし"

        sell_reason_arrow = [
            "企業の業績がいいから",
            "株価が急に動いたから",
            "財務データがいいから",
            "直感"
        ]

        #購入根拠
        st.session_state.basis_for_sale = st.radio("売却根拠", sell_reason_arrow)

        if st.button("売却する"):
            sell(name, rdf_all)
            change_page(2)

    
    def page8():
        st.title("テスト画面")
        st.subheader("買い・売りログデータ")
        col_buy, col_sell = st.columns(2)
        with col_buy:
            st.dataframe(st.session_state.buy_log)
        with col_sell:
            st.dataframe(st.session_state.sell_log)

        st.button("選択可能銘柄一覧へ戻る",on_click=lambda: change_page(1))

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



    # pages = dict(
    #     page1="選択可能銘柄一覧",
    #     page2="トレード画面",
    #     page3="保有資産",
    #     page4="企業情報",
    #     page5="結果",
    #     page6="テスト画面"
    # )

    # page_id = st.sidebar.selectbox( # st.sidebar.*でサイドバーに表示する
    #     "ページ名",
    #     ["page1", "page2", "page3", "page4","page5","page6"],
    #     format_func=lambda page_id: pages[page_id], # 描画する項目を日本語に変換
    #     key="page-select"
    # )

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

    # st.sidebar.write("_______________________________________________________________________________________________________")

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

    if st.sidebar.button('シミュレーションを終了する'):
        st.session_state.show_page = False
        # col_yes, col_no = st.columns((5, 5))
        # with col_yes:
        #     if st.sidebar.button("はい"):
        #         st.session_state.show_page = False
        # with col_no:
        #     st.sidebar.button("いいえ")
        



#_____________________________________________________________スタート画面_________________________________________________________________________________________________________________________________

else:

    def page2_1():
        st.title("スタート画面")

        st.write("########################################")

        st.write("ここにシミュレーションの説明を書く")

        st.write("########################################")


        button_css = f"""
            <style>
                div.stButton > button:first-child  {{
                color        : white               ;
                width        : 100%                ;
                font-weight  : bold                ;/* 文字：太字                   */
                border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
                border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
                background   : #a9a9a9             ;/* 背景色：薄いグレー            */
            }}
            </style>
            """
        st.markdown(button_css, unsafe_allow_html=True)

        col5, col6 = st.columns((5, 5))
        with col5:
            st.button("投資経験がない方はこちらへ",on_click=lambda: change_page2(4))

        with col6:
            st.button("このシミュレーションシステムについて", on_click=lambda: change_page2(5))

        st.button("これまでの実績", on_click=lambda: change_page2(2))
        st.button("アカウント設定", on_click=lambda: change_page2(3))

        level_id = st.selectbox(
            "レベルセレクト",
            ["チュートリアル", "LEVEL_1", "LEVEL_2", "LEVEL_3"],
            key="level-select"
        )

        if level_id == "チュートリアル":
            st.session_state.all_range_end = dt.datetime(2021,2,1) 
            st.write("１ヶ月で利益を出してください")
        
        if level_id == "LEVEL_1":
            st.session_state.all_range_end = dt.datetime(2021,4,1) 
            st.write("３ヶ月で +5万円の利益を出してください")

        if level_id == "LEVEL_2":
            st.session_state.all_range_end = dt.datetime(2021,7,1) 
            st.write("６ヶ月で +10万円の利益を出してください")

        if level_id == "LEVEL_3":
            st.session_state.all_range_end = dt.datetime(2022,1,1) 
            st.write("１年で +20万円利益を出してください")    

        if st.button('シミュレーションをはじめから始める'):
            reset()
            if st.session_state.account_created==True:
                st.session_state.show_page = True
                # st.session_state.reset = True
            else:
                st.write("アカウント情報を入力してください")

        if st.button('シミュレーションを続きから始める'):
            if st.session_state.account_created==True:
                st.session_state.show_page = True
                # st.session_state.reset = True
            else:
                st.write("アカウント情報を入力してください")


    def page2_2():
        st.title("実績")

        st.write("########################################")

        st.write("ここにこれまでの実績を持ってくる")

        st.write("########################################")

        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))


    def page2_3():
        st.title("アカウント情報")
        st.write("########################################")

        st.write("ここに個人情報の利用について書く")

        st.write("########################################")

        if "account_created" not in st.session_state:
            st.session_state.account_created = False

        if not st.session_state.account_created:
            if st.button("アカウントを作成する"):
                st.session_state.account_created = True

        if st.session_state.account_created:
            st.session_state.acount_name = st.text_input("アカウント名を入力してください", value=st.session_state.get("acount_name", ""))
            st.session_state.acount_age = st.text_input("年齢を入力してください", value=st.session_state.get("acount_age", ""))
            st.session_state.acount_sex = st.selectbox("性別を入力してください", ("男", "女"), index=0 if st.session_state.get("acount_sex", "男") == "男" else 1)

            st.write("以下のURLから個人の性格についてのテストを実施して情報を入力してください。")
            st.write("https://commutest.com/bigfive")
            st.session_state.Open = st.slider("開放性", 0, 6, st.session_state.get("Open", 6))
            st.session_state.Integrity = st.slider("誠実性", 0, 6, st.session_state.get("Integrity", 6))
            st.session_state.Diplomatic = st.slider("外交性", 0, 6, st.session_state.get("Diplomatic", 6))
            st.session_state.Coordination = st.slider("協調性", 0, 6, st.session_state.get("Coordination", 6))
            st.session_state.Neuroticism = st.slider("神経症傾向", 0, 6, st.session_state.get("Neuroticism", 6))


        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))



    
    def page2_4():
        st.title("投資について")

        st.write("########################################")

        st.write("ここに投資についての説明を書く")

        st.write("########################################")

        button_css = f"""
            <style>
                div.stButton > button:first-child  {{
                color        : white               ;
                width        : 100%                ;
                font-weight  : bold                ;/* 文字：太字                   */
                border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
                border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
                background   : #a9a9a9             ;/* 背景色：薄いグレー            */
            }}
            </style>
            """
        st.markdown(button_css, unsafe_allow_html=True)
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))


    def page2_5():
        st.title("このシミュレーションについて")

        st.write("########################################")

        st.write("ここにシミュレーションの説明を書く")

        st.write("########################################")

        button_css = f"""
            <style>
                div.stButton > button:first-child  {{
                color        : white               ;
                width        : 100%                ;
                font-weight  : bold                ;/* 文字：太字                   */
                border       : 1px solid #000      ;/* 枠線：ピンク色で5ピクセルの実線 */
                border-radius: 1px 1px 1px 1px     ;/* 枠線：半径10ピクセルの角丸     */
                background   : #a9a9a9             ;/* 背景色：薄いグレー            */
            }}
            </style>
            """
        st.markdown(button_css, unsafe_allow_html=True)
        st.button("スタート画面に戻る",on_click=lambda: change_page2(1))


    # pages2 = dict(
    #     page2_1="スタート画面",
    #     page2_2="実績",
    #     page2_3="アカウント情報",
    #     page2_4="投資について",
    #     page2_5="このシミュレーションについて"
    # )

    # page_id2 = st.sidebar.selectbox( # st.sidebar.*でサイドバーに表示する
    #     "ページ名",
    #     ["page2_1", "page2_2", "page2_3", "page2_4", "page2_5"],
    #     format_func=lambda page_id2: pages2[page_id2], # 描画する項目を日本語に変換
    #     key="page-select2"
    # )

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

# if st.session_state.reset:
#     st.session_state.counter = 0
#     st.session_state.reset = False
