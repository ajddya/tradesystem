#app.py
import streamlit as st

import os

import json

import pandas as pd

import sqlite3

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

def insert_data_to_db():
    # データベースに接続
    conn = sqlite3.connect('my_database.db')
    c = conn.cursor()

    # テーブルの削除
    # c.execute("drop table users")

    c.execute('CREATE TABLE IF NOT EXISTS users(name TEXT)')

    name = st.session_state.user_name
    
    c.execute("INSERT INTO users (name) VALUES (?)", (name,))

    # テーブルの削除
    # c.execute("drop table user_data")

    conn.commit()

    # カーソルをクローズ（オプション）
    c.close()

    # データベースの接続をクローズ
    conn.close()

def check_db():
        # データベースに接続
        conn = sqlite3.connect('my_database.db')
        c = conn.cursor()

        c.execute('SELECT * FROM users ')
        data = c.fetchall()

        for row in data:
            st.write(row)


        # カーソルをクローズ（オプション）
        c.close()

        # データベースの接続をクローズ
        conn.close()


# Streamlitアプリケーションのタイトル
st.title('データベース操作のデモ')

# ユーザー入力の取得
st.session_state.user_name = st.text_input('名前を入力してください')

# データベースへのデータ挿入
st.button('データを送信',on_click = insert_data_to_db)

# データベースからデータを表示
st.button('データを表示',on_click=check_db)

st.button('データをダウンロード',on_click=download_db)
