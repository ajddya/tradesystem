#app.py
import streamlit as st

import os

import sqlalchemy

def connect_tcp_socket() -> sqlalchemy.engine.base.Engine:
    """Initializes a TCP connection pool for a Cloud SQL instance of MySQL."""
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.
    db_host = os.environ[
        "INSTANCE_HOST"
    ]  # e.g. '127.0.0.1' ('172.17.0.1' if deployed to GAE Flex)
    db_user = os.environ["DB_USER"]  # e.g. 'my-db-user'
    db_pass = os.environ["DB_PASS"]  # e.g. 'my-db-password'
    db_name = os.environ["DB_NAME"]  # e.g. 'my-database'
    db_port = os.environ["DB_PORT"]  # e.g. 3306

    pool = sqlalchemy.create_engine(
        # Equivalent URL:
        # mysql+pymysql://<db_user>:<db_pass>@<db_host>:<db_port>/<db_name>
        sqlalchemy.engine.url.URL.create(
            drivername="mysql+pymysql",
            username=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
            database=db_name,
        ),
        # ...
    )
    return pool

db = connect_tcp_socket()

# Streamlitアプリケーションのタイトル
st.title('データベース操作のデモ')

# ユーザー入力の取得
name = st.text_input('名前を入力してください')

# データベースへのデータ挿入
if st.button('データを送信'):
    with db.connect() as conn:
        # テーブルの作成（初回のみ）
        conn.execute(sqlalchemy.text(f'CREATE TABLE IF NOT EXISTS users(name TEXT)'))
        conn.execute(sqlalchemy.text(f"INSERT INTO users (name) VALUES ('{name}')"))
        st.success('データがデータベースに挿入されました')

# データベースからデータを表示
if st.button('データを表示'):
    with db.connect() as conn:
        result = conn.execute("SELECT * FROM users").fetchall()
        for row in result:
            st.write(f"名前: {row[0]}")
