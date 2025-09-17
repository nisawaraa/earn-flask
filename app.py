# app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import io
import base64
import folium
import matplotlib
from branca.colormap import LinearColormap   # ✅ ใช้ LinearColormap ตรง ๆ
from branca.colormap import LinearColormap

matplotlib.rcParams['font.family'] = 'Tahoma'   # หรือ 'TH Sarabun New'

app = Flask(__name__)

# ------------------ โหลดข้อมูล ------------------
df = pd.read_excel('df_final_clean.xlsx')
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# ซีรีส์รายเดือน
monthly = (
    df.groupby(['year', 'month'], as_index=False)['cases']
      .sum()
      .sort_values(['year', 'month'])
      .reset_index(drop=True)
)

# ------------------ โหลดโมเดล ------------------
xgb_model = joblib.load("xgb_forecast_model.pkl")

with open('sarima_model.pkl', 'rb') as f:
    sarima_model = pickle.load(f)

# ------------------ Helper ------------------
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def ym_to_label(y, m):
    return f"{int(y)}-{int(m):02d}"

def next_ym(y, m):
    """คืนค่า (year, month) ของเดือนถัดไป"""
    y = int(y); m = int(m)
    if m == 12:
        return y + 1, 1
    return y, m + 1

# ------------------ XGB forecast ------------------
def xgb_forecast_steps(series_cases, last_year, last_month, steps):
    forecasts = []
    s = list(map(float, series_cases))  # copy
    cur_y, cur_m = int(last_year), int(last_month)

    def month_sin(m): return np.sin(2*np.pi*m/12)
    def month_cos(m): return np.cos(2*np.pi*m/12)

    for _ in range(steps):
        if cur_m == 12:
            cur_y, cur_m = cur_y + 1, 1
        else:
            cur_m += 1

        def lag(k):  return s[-k] if len(s) >= k else s[0]
        def roll(k): return np.mean(s[-k-1:-1]) if len(s) >= (k+1) else np.mean(s)

        x = {
            "lag1":  lag(1), "lag2":  lag(2), "lag3":  lag(3),
            "roll3":  roll(3), "roll6":  roll(6),
            "month": cur_m, "year": cur_y,
            "month_sin": month_sin(cur_m), "month_cos": month_cos(cur_m)
        }
        X_next = pd.DataFrame([x])
        yhat = float(xgb_model.predict(X_next)[0])
        forecasts.append({"date": f"{cur_y}-{cur_m:02d}", "prediction": int(round(yhat))})

        s.append(yhat)

    return forecasts

# ------------------ Routes ------------------
@app.route('/')
def index():
    last_year = monthly.iloc[-1]['year']
    last_month = monthly.iloc[-1]['month']
    latest_cases = int(monthly.iloc[-1]['cases'])

    if last_month == 1:
        prev_y, prev_m = last_year - 1, 12
    else:
        prev_y, prev_m = last_year, last_month - 1

    prev_cases_row = monthly[(monthly['year'] == prev_y) & (monthly['month'] == prev_m)]
    prev_cases = int(prev_cases_row['cases'].values[0]) if len(prev_cases_row) else 0
    diff = latest_cases - prev_cases

    # ข้อมูลจริง 12 เดือนล่าสุด
    recent = monthly.tail(12).copy()
    history_labels = [ym_to_label(r['year'], r['month']) for _, r in recent.iterrows()]
    history_values = [int(r['cases']) for _, r in recent.iterrows()]

    return render_template(
        'index.html',
        latest_cases=latest_cases,
        diff=diff,
        trend_up=(diff > 0),
        history_labels=history_labels,
        history_values=history_values
    )

@app.route('/api/forecast')
def api_forecast():
    steps = int(request.args.get('steps', 12))
    steps = max(1, min(12, steps))

    last_year = int(monthly.iloc[-1]['year'])
    last_month = int(monthly.iloc[-1]['month'])
    series_cases = monthly['cases'].astype(float).tolist()

    # XGB
    fc_xgb = xgb_forecast_steps(series_cases, last_year, last_month, steps)

    # SARIMA
    fc_sarima = []
    sarima_pred = sarima_model.get_forecast(steps=steps).predicted_mean
    for dt, val in sarima_pred.items():
        fc_sarima.append({"date": dt.strftime("%Y-%m"), "prediction": int(round(val))})

    return jsonify({
        "steps": steps,
        "xgb": fc_xgb,
        "sarima": fc_sarima
    })

@app.route('/yearly')
def yearly():
    yearly_sum = (
        df.groupby('year')['cases']
          .sum()
          .reset_index()
          .sort_values('year')
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly_sum['year'].astype(str), yearly_sum['cases'])
    ax.set_title('จำนวนผู้ป่วยรายปี')
    ax.set_xlabel('ปี')
    ax.set_ylabel('จำนวนผู้ป่วย')
    img_yearly = plot_to_img(fig)

    # ✅ แปลงเป็น list[dict] เช่น [{'year': 2017, 'cases': 75}, ...]
    yearly_data = yearly_sum.to_dict(orient='records')

    return render_template(
        'yearly.html',
        yearly=yearly_data,
        yearly_img=img_yearly
    )

@app.route("/phayao")
def phayao():
    # ดึงค่าปีจาก query string (ถ้าไม่มีจะเป็น None)
    year = request.args.get("year", type=int)

    # กรองข้อมูลตามปี
    tambon_cases = df.copy()
    if year:
        tambon_cases = tambon_cases[tambon_cases['year'] == year]

    tambon_cases = (
        tambon_cases.groupby('ตำบล', as_index=False)['cases']
                    .sum()
                    .rename(columns={'cases': 'count'})
    )

    # โหลดพิกัด
    coords = pd.read_csv("phayao_tambon_coordinates.csv")
    coords.columns = coords.columns.str.strip()

    # รวมเข้ากับจำนวนผู้ป่วย
    merged = coords.merge(tambon_cases, on='ตำบล', how='left')
    merged['count'] = merged['count'].fillna(0)
    merged = merged.dropna(subset=['lat', 'lon'])

    max_count = float(merged['count'].max()) if len(merged) else 1.0
    center_lat = merged['lat'].mean() if len(merged) else 19.25
    center_lon = merged['lon'].mean() if len(merged) else 99.9

    # colormap ไล่สี เขียว → เหลือง → ส้ม → แดง
    colormap = LinearColormap(
    colors=['green', 'yellow', 'orange', 'red'],
    vmin=merged['count'].min(),
    vmax=merged['count'].max()
)

    colormap.caption = f"จำนวนผู้ป่วย{' ปี '+str(year) if year else ' (รวมทุกปี)'}"

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="cartodbpositron")

    for _, row in merged.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6 + (row["count"] / max_count) * 14,
            color=colormap(row["count"]),
            fill=True,
            fill_color=colormap(row["count"]),
            fill_opacity=0.85,
            weight=1,
            popup=f"<b>{row['ตำบล']}</b><br>ผู้ป่วย: {int(row['count'])} คน"
        ).add_to(m)

    colormap.add_to(m)

    return render_template(
        "phayao.html",
        map_html=m._repr_html_(),
        year=year,
        years=sorted(df['year'].unique())
    )

@app.route('/sarima_forecast')
def sarima_forecast_api():
    forecast = sarima_model.get_forecast(steps=12).predicted_mean
    pred_df = pd.DataFrame({'date': forecast.index, 'prediction': forecast.values})
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    return jsonify(pred_df.to_dict(orient='records'))

# ------------------ Run ------------------
if __name__ == '__main__':
    app.run(debug=True)
