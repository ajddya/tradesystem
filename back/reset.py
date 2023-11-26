import streamlit as st

def reset():
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

    if "survey_bool" in st.session_state:
        del st.session_state.survey_bool

    if "possess_money_bool" in st.session_state:
        del st.session_state.possess_money_bool
