import streamlit as st
import pandas_datareader.data as pdr

#日経平均株価を取得
def get_NKX_data():
    NKX = pdr.DataReader("^NKX","stooq",all_range_start,st.session_state.now).sort_index()

    NKX["ma5"]   = NKX["Close"].rolling(window=5).mean()
    NKX["ma25"]  = NKX["Close"].rolling(window=25).mean()
    NKX["ma75"]  = NKX["Close"].rolling(window=75).mean()

    return NKX

