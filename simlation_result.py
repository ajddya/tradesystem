import streamlit as st

#シミュレーション結果を保存するクラス
class Simulation_Results:
    def __init__(self, dates, num, action_type, LEVEL, investment_result, buy_log, sell_log, Dividends, advice, trade_advice):
        self.dates = dates                             #実施日
        self.num = num                                 #実施回数
        self.action_type = action_type                 #行動型
        self.LEVEL = LEVEL                             #保有金額
        self.investment_result = investment_result     #投資結果（利益）
        self.buy_log = buy_log                         #購入ログ
        self.sell_log = sell_log                       #売却ログ
        self.Dividends = Dividends                     #配当に関するデータ
        self.advice = advice                           #行動経済学の指摘事項
        self.trade_advice = trade_advice               #各取引の指摘事項
        self._observers = []

    def display(self):
        # ここに、このクラスのデータを表示するためのコードを追加できます
        st.write(f"実施日: {self.dates}")
        st.write(f"レベル：{self.LEVEL}")
        st.write(f"分類型: {self.action_type}")
        st.write(f"利益: {self.investment_result}")      
        st.write("全体のアドバイス:")
        st.write(self.advice)
        st.write("各取引のアドバイス:")
        st.write(self.trade_advice)