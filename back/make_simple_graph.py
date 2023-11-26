import streamlit as st
import io
import mplfinance as mpf
import plotly.graph_objs as go


#rdfからアクティブでないグラフを作る
@st.cache_data
def make_simple_graph(name, rdf):
    rdf = rdf[-20:]
    fig, axes = mpf.plot(rdf, type='candle',figsize=(6, 3), volume=True, returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=75)
    buf.seek(0)
    return buf

