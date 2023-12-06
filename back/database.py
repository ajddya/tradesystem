import streamlit as st
import sqlite3
import pickle
import json
import pandas as pd
import os
import shutil
import time

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

# new_datebase を my_database にコピーする
def save_db():
    # 新しいデータベースファイルのパス
    new_db_path = 'new_database.db'

    # 置き換える既存のデータベースファイルのパス
    existing_db_path = 'my_database.db'

    # 新しいファイルが存在するか確認
    if os.path.exists(new_db_path):
        # 新しいデータベースファイルで既存のファイルを置き換え
        shutil.copyfile(new_db_path, existing_db_path)

    else:
        st.sidebar.write("データの保存に失敗しました。")


# データをデータベースに保存する
def save_userdata():
    # データベースに接続
    conn = sqlite3.connect('new_database.db')
    cursor = conn.cursor()

    # テーブルを削除
    # cursor.execute("drop table personal_info")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS personal_info (
        ユーザ名 TEXT PRIMARY KEY,
        ユーザID TEXT,
        年齢 INTEGER,
        性別 TEXT,
        投資経験の有無 TEXT,
        投資に関する知識の有無 TEXT,
        開放性 FLOAT,
        誠実性 FLOAT,
        外交性 FLOAT,
        協調性 FLOAT,
        神経症傾向 FLOAT
    )
    """)

    # データの挿入または更新
    for _, row in st.session_state.personal_df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO personal_info (ユーザ名, ユーザID, 年齢, 性別, 投資経験の有無, 投資に関する知識の有無, 開放性, 誠実性, 外交性, 協調性, 神経症傾向)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (row["ユーザ名"], row["ユーザID"], row["年齢"], row["性別"], row["投資経験の有無"], row["投資に関する知識の有無"], row["開放性"], row["誠実性"], row["外交性"], row["協調性"], row["神経症傾向"]))
    
    # # result __________________________________________________________________________________________________________

    # テーブルを削除
    # cursor.execute("drop table simulation_results")

    # テーブルの作成
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_results (
        ユーザ名 TEXT PRIMARY KEY,
        data BLOB
    )
    """)

    row = st.session_state.personal_df.iloc[0]
    current_result = st.session_state.result

    # データベースから前回のデータを取得
    cursor.execute("SELECT data FROM simulation_results WHERE ユーザ名=?", (row["ユーザ名"],))
    pre_data = cursor.fetchone()

    change_data_bool = False
    if pre_data:  # 前回のデータが存在する場合
        previous_result = pickle.loads(pre_data[0])
        if previous_result != current_result:
            change_data_bool = True
    else:  # 前回のデータが存在しない場合
        change_data_bool = True


    # st.session_state.resultが変更されている場合のみデータベースを更新
    if change_data_bool:
        serialized_data = pickle.dumps(st.session_state.result)
        cursor.execute("INSERT OR REPLACE INTO simulation_results (ユーザ名, data) VALUES (?, ?)", (row["ユーザ名"], serialized_data))


    # その他変数をデータベースに格納_______________________________________________________________________________________
    save_values = {
            "now": [st.session_state.now], 
            "chose_companies_name_list": [st.session_state.chose_companies_name_list],  
            "possess_money": [st.session_state.possess_money],
            "possess_money_init": [st.session_state.possess_money_init],
            "possess_KK_df": [st.session_state.possess_KK_df],
            "buy_log": [st.session_state.buy_log],
            "sell_log": [st.session_state.sell_log],
            "Dividends_df": [st.session_state.Dividends_df],
            "advice_df": [st.session_state.advice_df],
            "trade_advice_df": [st.session_state.trade_advice_df],
            "create_chose_companies_executed": [st.session_state.create_chose_companies_executed],
            "selected_company": [st.session_state.selected_company],
            "result_bool": [st.session_state.result_bool],
            "survey_bool": [st.session_state.survey_bool],
            "possess_money_bool": [st.session_state.possess_money_bool],
    }
    save_values_df = pd.DataFrame(save_values)

    df_binary = pickle.dumps(save_values_df)

    # テーブルを削除
    # cursor.execute("drop table value_table")

    # テーブルの作成
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS value_table (
        ユーザ名 TEXT PRIMARY KEY,
        value BLOB
    )
    """)

    cursor.execute('INSERT OR REPLACE INTO value_table (ユーザ名, value) VALUES (?, ?)', (st.session_state.personal_df["ユーザ名"][0], df_binary))



    # データベースの変更をコミット
    conn.commit()

    # データベース接続のクローズ
    conn.close()

    save_db()

    time.sleep(1)

# データの挿入
def insert_data_to_db(private_data, result_data):
    # データベースに接続
    conn = sqlite3.connect('new_database.db')
    c = conn.cursor()

    # テーブルの削除
    # c.execute("drop table user_data")

    # テーブルの作成（初回のみ）
    c.execute('CREATE TABLE IF NOT EXISTS user_data(result_data)')

    # private_dataとresult_dataを１つに結合
    result_data = pd.concat([private_data, result_data],axis=1)
    result_data_serialized = result_data.to_json()

    c.execute('INSERT INTO user_data (result_data) VALUES (?)', (result_data_serialized,))

    conn.commit()

    # カーソルをクローズ（オプション）
    c.close()

    # データベースの接続をクローズ
    conn.close()

    save_db()

def insert_survey_to_db():

    # データベースに接続
    conn = sqlite3.connect('new_database.db')
    cursor = conn.cursor()

    # テーブルを削除
    # cursor.execute("drop table survey_info")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS survey_info (
        ユーザ名 TEXT,
        ユーザID TEXT,
        実施回数 INTEGER,
        システムの満足度1 TEXT,
        システムの満足度2 TEXT,
        システムの満足度3 TEXT,
        システムの正確性1 TEXT,
        システムの正確性2 TEXT,
        システムの有用性1 TEXT,
        システムの有用性2 TEXT,
        意見 TEXT
    )
    """)

    # データの挿入または更新
    cursor.execute("""
    INSERT INTO survey_info (ユーザ名, ユーザID, 実施回数, システムの満足度1, システムの満足度2, システムの満足度3, システムの正確性1, システムの正確性2, システムの有用性1, システムの有用性2, 意見)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (st.session_state.acount_name, st.session_state.acount_ID, st.session_state.num - 1, st.session_state.satisfaction, st.session_state.satisfaction2, st.session_state.satisfaction3, st.session_state.accurate_classify, st.session_state.accurate_instruction, st.session_state.usefulness1, st.session_state.usefulness2, st.session_state.opinion))

    # データベースの変更をコミット
    conn.commit()

    # データベース接続のクローズ
    conn.close()

    st.session_state.survey_bool = True
    change_page2(1)

    save_db()
