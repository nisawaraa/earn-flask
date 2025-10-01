from flask import Flask, render_template, request
import pandas as pd
import joblib, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap
import folium

matplotlib.rcParams['font.family'] = 'Tahoma'
app = Flask(__name__)

# ---------------- Load Data ----------------
df = pd.read_excel("df_final_clean.xlsx")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

available_years = sorted(df["year"].unique().tolist())

tambon_list = sorted(df["ตำบล"].dropna().astype(str).unique().tolist())
ampur_list  = sorted(df["อำเภอ"].dropna().astype(str).unique().tolist())
occ_list    = sorted(df["อาชีพ"].dropna().astype(str).unique().tolist())
season_list = sorted(df["ฤดูกาล"].dropna().astype(str).unique().tolist())

# ---------------- Load Model ----------------
xgb_model = joblib.load("xgb_regressor.pkl")
expected_features = joblib.load("xgb_features.pkl")

# ---------------- Helpers ----------------
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

def add_months(year: int, month: int, delta: int):
    base = (year * 12) + (month - 1) + delta
    return base // 12, (base % 12) + 1

# ---------------- Routes ----------------
@app.route("/")
def welcome():
    return render_template("welcome.html")

# ---------- Input Prediction ----------
@app.route("/predict_input", methods=["GET"])
def predict_input_page():
    default_temp = round(float(df["temp_15d_avg"].mean()), 1) if "temp_15d_avg" in df else 0
    default_rain = round(float(df["rain_15d_avg"].mean()), 1) if "rain_15d_avg" in df else 0
    return render_template(
        "predict_input.html",
        features=expected_features,
        tambon_list=tambon_list,
        ampur_list=ampur_list,
        occ_list=occ_list,
        season_list=season_list,
        available_years=available_years,
        default_temp=default_temp,
        default_rain=default_rain
    )

@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        temp = float(request.form.get("temp_15d_avg", 0))
        rain = float(request.form.get("rain_15d_avg", 0))
        occ = request.form.get("อาชีพ", "ไม่ระบุ")
        tambon = request.form.get("ตำบล", "ไม่ระบุ")
        ampur = request.form.get("อำเภอ", "ไม่ระบุ")
        season = request.form.get("ฤดูกาล", "ไม่ระบุ")
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))

        # lag features
        lags = {}
        for k in [1, 2, 3]:
            yk, mk = add_months(year, month, -k)
            lag_sub = df[(df["year"] == yk) & (df["month"] == mk)]
            lags[f"cases_lag{k}"] = float(lag_sub["cases"].sum()) if not lag_sub.empty else 0.0

        # DataFrame input
        base = {
            "temp_15d_avg": temp,
            "rain_15d_avg": rain,
            "อาชีพ": occ,
            "ตำบล": tambon,
            "อำเภอ": ampur,
            "ฤดูกาล": season,
            "year": year,
            "month": month,
            "cases_lag1": lags["cases_lag1"],
            "cases_lag2": lags["cases_lag2"],
            "cases_lag3": lags["cases_lag3"],
        }
        X_input = pd.DataFrame([base])
        X_input = pd.get_dummies(X_input)
        X_input = X_input.reindex(columns=expected_features, fill_value=0)

        pred = int(round(float(xgb_model.predict(X_input)[0])))

        # ค่าจริงจาก dataset
        sub = df[(df["year"] == year) & (df["month"] == month)]
        real_val = int(sub["cases"].sum()) if not sub.empty else None

        return render_template("predict_result.html",
                               pred=pred,
                               real=real_val,
                               temp=temp, rain=rain,
                               occ=occ, tambon=tambon, ampur=ampur, season=season,
                               year=year, month=month, lags=lags)

    except Exception as e:
        return render_template("predict_result.html", pred=None, error=str(e))

# ---------- Yearly ----------
@app.route("/yearly")
def yearly():
    yearly_sum = df.groupby("year")["cases"].sum().reset_index().sort_values("year")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly_sum["year"].astype(str), yearly_sum["cases"], color="skyblue")
    ax.set_title("จำนวนผู้ป่วยไข้เลือดออกรายปี")
    ax.set_xlabel("ปี")
    ax.set_ylabel("จำนวนผู้ป่วย")
    yearly_img = plot_to_img(fig)
    return render_template("yearly.html",
        yearly=yearly_sum.to_dict(orient="records"),
        yearly_img=yearly_img)
# ---------- Phayao Map ----------
@app.route("/phayao")
def phayao():
    year = request.args.get("year", type=int)
    tambon_cases = df.copy()
    if year:
        tambon_cases = tambon_cases[tambon_cases["year"] == year]

    tambon_cases = (
        tambon_cases.groupby("ตำบล", as_index=False)["cases"]
        .sum()
        .rename(columns={"cases": "count"})
    )

    coords = pd.read_csv("phayao_tambon_coordinates.csv")
    coords.columns = coords.columns.str.strip()

    merged = coords.merge(tambon_cases, on="ตำบล", how="left")
    merged["count"] = merged["count"].fillna(0)
    merged = merged.dropna(subset=["lat", "lon"])

    max_count = float(merged["count"].max()) if len(merged) else 1.0
    center_lat = merged["lat"].mean() if len(merged) else 19.25
    center_lon = merged["lon"].mean() if len(merged) else 99.9

    colormap = LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=merged["count"].min(),
        vmax=merged["count"].max()
    )
    colormap.caption = f"จำนวนผู้ป่วย{' ปี ' + str(year) if year else ' (รวมทุกปี)'}"

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="cartodbpositron")

    # ✅ loop ต้องอยู่ในฟังก์ชัน ไม่ใช่นอกฟังก์ชัน
    for _, r in merged.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + (r["count"] / max_count) * 14,
            color=colormap(r["count"]),
            fill=True,
            fill_color=colormap(r["count"]),
            fill_opacity=0.85,
            weight=1,
            popup=f"<b>{r['ตำบล']}</b><br>ผู้ป่วย: {int(r['count'])} คน",
            tooltip=f"{r['ตำบล']} - {int(r['count'])} คน"
        ).add_to(m)

    colormap.add_to(m)

    return render_template(
        "phayao.html",
        map_html=m._repr_html_(),
        year=year,
        years=available_years
    )

if __name__ == "__main__":
    app.run(debug=True)
