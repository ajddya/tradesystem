#app.py
import streamlit as st

import os
import sqlalchemy

# 環境変数からデータベース接続情報を取得
db_user = os.environ.get("DB_USER")
db_pass = os.environ.get("DB_PASS")
db_name = os.environ.get("DB_NAME")
cloud_sql_connection_name = os.environ.get("CLOUD_SQL_CONNECTION_NAME")

# SQLAlchemyエンジンを使用してデータベースに接続
db = sqlalchemy.create_engine(
    sqlalchemy.engine.url.URL(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_pass,
        database=db_name,
        query={"unix_socket": f"/cloudsql/{cloud_sql_connection_name}"},
        # または "host" と "port" を使用
        host="vocal-well-405800:asia-northeast1:result",
        port="3306" #(通常は 3306)
    )
)

# Streamlitアプリケーションのタイトル
st.title('データベース操作のデモ')

# ユーザー入力の取得
name = st.text_input('名前を入力してください')

# データベースへのデータ挿入
if st.button('データを送信'):
    with db.connect() as conn:
        conn.execute(sqlalchemy.text(f"INSERT INTO users (name) VALUES ('{name}')"))
        st.success('データがデータベースに挿入されました')

# データベースからデータを表示
if st.button('データを表示'):
    with db.connect() as conn:
        result = conn.execute("SELECT * FROM users").fetchall()
        for row in result:
            st.write(f"ID: {row[0]}, 名前: {row[1]}")
