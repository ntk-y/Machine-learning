# 明日の平均気温予測
"""
直前6日間の平均気温から線形回帰モデルで翌日の気温を予測する。
気象庁のサイト(http://www.data.jma.go.jp/gmd/risk/obsdl/index.php)から
過去の10年分の気象データをダウンロードして予測可能。
"""

import csv
import datetime
import pandas as pd

# csvファイルを読み込む
with open('data.csv', 'r', encoding='Shift_JIS') as fr:
    # lines = fr.read()
    lines = fr.readlines() # 読み込んで文字列のリスト化

# ヘッダーを付け替える
lines = ["年,月,日,気温,品質,均質\n"] + lines[5:] # ヘッダー作成 5行目以降追加
lines = map(lambda v: v.replace('/', ','), lines) # / → , 置き換え
result = "".join(lines).strip()
# print(result)

# 結果をファイルへ出力
with open('temp10y.csv', 'w', encoding='utf-8') as fw:
  fw.write(result)
  print("saved.")

# 気温データの読み込み
df = pd.read_csv("temp10y.csv", encoding="utf-8")

# データを学習用と予測用に分離する
train_data = (df["年"] <= 2019) # 学習用
pre_data = (df["年"] >= 2020) # 予測用
interval = 6 # 直前6日のデータから予測

# 直前6日分を学習するためのデータ整理
def make_data(data):
    x=[]
    y=[]
    temps = list(data["気温"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for j in range(interval):
            d = i + j - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

train_x, train_y = make_data(df[train_data])

# print(train_x)
# print(train_y)

# 直線回帰分析を行う
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y)
coef = list(lr.coef_)
intercept = lr.intercept_

# print(coef)
# print(intercept)

# 回帰分析の結果から明日の平均気温を予測する
last6d = (df[pre_data].iloc[-6:])
last6d_temps = list(last6d["気温"])
pre_temp = sum([x * y for (x, y) in zip(coef, last6d_temps)]) + intercept

# 明日の日付
year = (df["年"].iloc[-1])
month = (df["月"].iloc[-1])
day = (df["日"].iloc[-1])
last_date = datetime.date(year, month, day) + datetime.timedelta(days=1)

print("明日(" + str(last_date) + ")の平均気温は " + str(pre_temp) + " 度です。")
