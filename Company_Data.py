import streamlit as st

#企業データを格納するクラス
class CompanyData:
    def __init__(self, code, name, rdf_all, _rdf):
        self.code = code
        self.name = name
        self.rdf_all = rdf_all
        self._rdf = _rdf

    def display(self):
        # ここに、このクラスのデータを表示するためのコードを追加できます
        print(f"Code: {self.code}")
        print(f"Name: {self.name}")
        print("RDF All:")
        print(self.rdf_all)
        print("RDF:")
        print(self._rdf)
        
    def to_list(self):
        return [self.code, self.name, self.rdf_all, self._rdf]
