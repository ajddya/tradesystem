import streamlit as st
import random

#　選択可能銘柄の最大表示数
i_max = 20

# 選択可能銘柄を選定する
def create_chose_companies():
    chosen_indices = random.sample(range(222), i_max)
    
    for idx in chosen_indices:
        com_temp2 = st.session_state.loaded_companies[idx]
        st.session_state.chose_companies.append(com_temp2)
        st.session_state.chose_companies_name_list.append(com_temp2.name)
