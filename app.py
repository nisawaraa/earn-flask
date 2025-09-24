# app.py
from flask import Flask, render_template, jsonify, request, Response
import pandas as pd
import numpy as np
import joblib, pickle, io, base64
import matplotlib
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap
import folium

matplotlib.rcParams['font.family'] = 'Tahoma'
app = Flask(__name__)

# ---------------- Load data ----------------
df = pd.read_excel("df_final_clean.xlsx")
df.columns = df.columns.str.strip()
df["date"]  = pd.to_datetime(df["date"])
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month

monthly = (
    df.groupby(["year","month"], as_index=False)["cases"]
      .sum().sort_values(["year","month"]).reset_index(drop=True)
)
available_years = sorted(df["year"].unique().tolist())

# ---------------- Load model + features + encoders ----------------
xgb_model = joblib.load("xgb_regressor.pkl")
try:
    expected_features = joblib.load("xgb_features.pkl")
except Exception:
    # เผื่อไม่มีไฟล์ฟีเจอร์
    try:
        expected_features = list(getattr(xgb_model, "feature_names_in_", []))
    except Exception:
        expected_features = None

try:
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
except Exception:
    label_encoders = {}

# ---------------- Helpers ----------------
def month_to_season(m: int) -> int:
    if m in [3,4,5]: return 1
    elif m in [6,7,8,9,10]: return 2
    else: return 3

def ym_label(y, m): return f"{int(y)}-{int(m):02d}"

def plot_to_img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0); b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig); return b64

def safe_le_transform(le, series: pd.Series) -> pd.Series:
    known = set(le.classes_.tolist()); out = []
    for v in series.astype(str):
        if v in known: out.append(int(le.transform([v])[0]))
        else: out.append(-1)
    return pd.Series(out, index=series.index, dtype="int32")

def build_feature_row(row: pd.Series, year: int, month: int) -> pd.DataFrame:
    # ใช้เฉพาะตัวแปรอิสระ (ไม่ใส่ cases)
    base = {
        "temp_15d_avg": float(row.get("temp_15d_avg", np.nan)),
        "rain_15d_avg": float(row.get("rain_15d_avg", np.nan)),
        "อาชีพ": str(row.get("อาชีพ","")),
        "ตำบล": str(row.get("ตำบล","")),
        "อำเภอ": str(row.get("อำเภอ","")),
        "เดือน": int(month),
        "ปี": int(year),
        "ฤดูกาล": int(month_to_season(month)),
    }
    X = pd.DataFrame([base])

    # encode หมวดหมู่ให้ตรงกับตอนเทรน
    for col in ["อาชีพ","ตำบล","อำเภอ"]:
        if col in label_encoders:
            X[col] = safe_le_transform(label_encoders[col], X[col])
        else:
            X[col] = -1

    # จัดคอลัมน์ให้ตรงกับ expected features
    if expected_features:
        X = X.reindex(columns=expected_features, fill_value=0)

    # บังคับเป็นตัวเลข
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        X[c] = X[c].astype("float32", errors="ignore")
    return X

def compare_real_vs_xgb_for_year(target_year: int):
    out = []
    for m in range(1, 12 + 1):
        sub = df[(df["year"] == target_year) & (df["month"] == m)]
        if sub.empty:
            continue

        # เลือกแถวแรกของเดือนเป็นตัวแทน
        row = sub.iloc[0]

        X = build_feature_row(row, target_year, m)
        yhat = float(xgb_model.predict(X)[0])

        out.append({
            "date": ym_label(target_year, m),
            "real": int(row["cases"]),
            "pred": int(round(yhat))   # 🔥 ปัดเป็นจำนวนเต็ม
        })
    return out


# ---------------- Routes ----------------
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/dashboard")
def dashboard():
    latest_cases = int(monthly.iloc[-1]["cases"])
    last_y = int(monthly.iloc[-1]["year"])
    last_m = int(monthly.iloc[-1]["month"])
    prev_y, prev_m = (last_y-1, 12) if last_m==1 else (last_y, last_m-1)
    prev_row = monthly[(monthly["year"]==prev_y) & (monthly["month"]==prev_m)]
    prev_cases = int(prev_row["cases"].values[0]) if not prev_row.empty else 0
    diff = latest_cases - prev_cases

    default_year = max(available_years)
    compare = compare_real_vs_xgb_for_year(default_year)

    recent = monthly.tail(12).copy()
    history_labels = [ym_label(r["year"], r["month"]) for _, r in recent.iterrows()]
    history_values = [int(r["cases"]) for _, r in recent.iterrows()]

    return render_template(
        "index.html",
        latest_cases=latest_cases, diff=diff, trend_up=(diff>0),
        years=available_years, default_year=default_year,
        compare_data=compare,
        history_labels=history_labels, history_values=history_values
    )

@app.route("/api/compare")
def api_compare():
    year = request.args.get("year", type=int, default=max(available_years))
    data = compare_real_vs_xgb_for_year(year)
    return jsonify({
        "year": year,
        "rows": data,
        "labels": [r["date"] for r in data],
        "real":   [r["real"] for r in data],
        "pred":   [r["pred"] for r in data],
    })

@app.route("/api/compare_csv")
def api_compare_csv():
    year = request.args.get("year", type=int, default=max(available_years))
    data = compare_real_vs_xgb_for_year(year)
    df_out = pd.DataFrame(data)
    csv = df_out.to_csv(index=False, encoding="utf-8-sig")
    return Response(csv, mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=compare_{year}.csv"})

# ---------- Yearly ----------
@app.route("/yearly")
def yearly():
    yearly_sum = df.groupby("year")["cases"].sum().reset_index().sort_values("year")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(yearly_sum["year"].astype(str), yearly_sum["cases"])
    ax.set_title("จำนวนผู้ป่วยรายปี"); ax.set_xlabel("ปี"); ax.set_ylabel("จำนวนผู้ป่วย")
    yearly_img = plot_to_img(fig)
    return render_template("yearly.html",
        yearly=yearly_sum.to_dict(orient="records"), yearly_img=yearly_img)

# ---------- Phayao Map ----------
@app.route("/phayao")
def phayao():
    year = request.args.get("year", type=int)
    tambon_cases = df.copy()
    if year: tambon_cases = tambon_cases[tambon_cases["year"]==year]
    tambon_cases = (tambon_cases.groupby("ตำบล", as_index=False)["cases"]
                    .sum().rename(columns={"cases":"count"}))
    coords = pd.read_csv("phayao_tambon_coordinates.csv"); coords.columns = coords.columns.str.strip()
    merged = coords.merge(tambon_cases, on="ตำบล", how="left")
    merged["count"] = merged["count"].fillna(0); merged = merged.dropna(subset=["lat","lon"])
    max_count = float(merged["count"].max()) if len(merged) else 1.0
    center_lat = merged["lat"].mean() if len(merged) else 19.25
    center_lon = merged["lon"].mean() if len(merged) else 99.9

    colormap = LinearColormap(colors=["green","yellow","orange","red"],
                              vmin=merged["count"].min(), vmax=merged["count"].max())
    colormap.caption = f"จำนวนผู้ป่วย{' ปี ' + str(year) if year else ' (รวมทุกปี)'}"
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="cartodbpositron")
    for _, r in merged.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + (r["count"]/max_count)*14,
            color=colormap(r["count"]), fill=True, fill_color=colormap(r["count"]),
            fill_opacity=0.85, weight=1,
            popup=f"<b>{r['ตำบล']}</b><br>ผู้ป่วย: {int(r['count'])} คน"
        ).add_to(m)
    colormap.add_to(m)
    return render_template("phayao.html", map_html=m._repr_html_(),
                           year=year, years=available_years)

if __name__ == "__main__":
    app.run(debug=True)
