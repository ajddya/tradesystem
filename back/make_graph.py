import streamlit as st

import pandas as pd

import pandas_datareader.data as pdr

import mplfinance as mpf
import plotly.graph_objs as go
import datetime as dt

import numpy as np

import matplotlib.pyplot as plt

# rdfからグラフを作る関数
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
