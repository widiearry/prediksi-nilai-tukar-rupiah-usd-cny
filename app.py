import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import os


st.set_page_config(
    page_title="Prediksi Nilai Tukar Rupiah",
    page_icon="💱",
    layout="wide"
)


# =========================
# SESSION STATE
# =========================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction_result" not in st.session_state:
    st.session_state.last_prediction_result = None


# =========================
# THEME CONFIG
# =========================
THEME_CONFIG = {
    "USD/IDR": {
        "title": "Prediksi Nilai Tukar Rupiah terhadap Dolar Amerika Serikat",
        "accent_primary": "#1f4d3d",
        "accent_secondary": "#2f6b56",
        "accent_soft": "#edf7ef",
        "accent_soft_2": "#f3f8f5",
        "accent_border": "#d8e5dc",
        "watermark": "$",
        "currency_label": "USD/IDR",
        "currency_name": "Dolar Amerika Serikat"
    },
    "CNY/IDR": {
        "title": "Prediksi Nilai Tukar Rupiah terhadap Yuan Tiongkok",
        "accent_primary": "#8b2e2e",
        "accent_secondary": "#b45c3b",
        "accent_soft": "#fbefec",
        "accent_soft_2": "#fff6f1",
        "accent_border": "#efd8cf",
        "watermark": "¥",
        "currency_label": "CNY/IDR",
        "currency_name": "Yuan Tiongkok"
    }
}


# =========================
# KALENDER OPERASIONAL BANK INDONESIA
# =========================
# Daftar tanggal berikut digunakan agar prediksi tidak diarahkan ke
# akhir pekan, hari libur nasional, atau cuti bersama Bank Indonesia.
# Apabila aplikasi digunakan untuk tahun berikutnya, tambahkan daftar
# libur BI pada dictionary ini.
BI_HOLIDAYS = {
    2026: {
        "2026-01-01",  # Tahun Baru Masehi
        "2026-01-16",  # Isra Mikraj Nabi Muhammad SAW
        "2026-02-17",  # Tahun Baru Imlek
        "2026-03-18",  # Cuti bersama
        "2026-03-19",  # Hari Suci Nyepi
        "2026-03-20",  # Cuti bersama Idulfitri
        "2026-03-21",  # Idulfitri
        "2026-03-22",  # Idulfitri
        "2026-03-23",  # Cuti bersama Idulfitri
        "2026-03-24",  # Cuti bersama Idulfitri
        "2026-04-03",  # Wafat Yesus Kristus
        "2026-04-05",  # Paskah
        "2026-05-01",  # Hari Buruh Internasional
        "2026-05-14",  # Kenaikan Yesus Kristus
        "2026-05-15",  # Cuti bersama Kenaikan Yesus Kristus
        "2026-05-27",  # Iduladha
        "2026-05-28",  # Cuti bersama Iduladha
        "2026-05-31",  # Hari Raya Waisak
        "2026-06-01",  # Hari Lahir Pancasila
        "2026-06-16",  # Tahun Baru Islam
        "2026-08-17",  # Hari Kemerdekaan RI
        "2026-08-25",  # Maulid Nabi Muhammad SAW
        "2026-12-24",  # Cuti bersama Natal
        "2026-12-25",  # Hari Raya Natal
    }
}


# =========================
# HELPER FUNCTION
# =========================
def get_today_date():
    return pd.Timestamp.today().normalize()


def is_weekend(date_value: pd.Timestamp):
    date_value = pd.to_datetime(date_value).normalize()
    return date_value.weekday() >= 5


def is_bi_holiday(date_value: pd.Timestamp):
    date_value = pd.to_datetime(date_value).normalize()
    holiday_set = BI_HOLIDAYS.get(date_value.year, set())
    return date_value.strftime("%Y-%m-%d") in holiday_set


def is_business_day(date_value: pd.Timestamp):
    date_value = pd.to_datetime(date_value).normalize()
    return not is_weekend(date_value) and not is_bi_holiday(date_value)


def get_next_business_day(date_value: pd.Timestamp):
    next_day = pd.to_datetime(date_value).normalize() + pd.Timedelta(days=1)
    while not is_business_day(next_day):
        next_day += pd.Timedelta(days=1)
    return next_day


def get_theme(currency: str):
    return THEME_CONFIG.get(currency, THEME_CONFIG["USD/IDR"])


def format_date_indo(date_value: pd.Timestamp):
    hari = {
        0: "Senin",
        1: "Selasa",
        2: "Rabu",
        3: "Kamis",
        4: "Jumat",
        5: "Sabtu",
        6: "Minggu"
    }

    bulan = {
        1: "Januari",
        2: "Februari",
        3: "Maret",
        4: "April",
        5: "Mei",
        6: "Juni",
        7: "Juli",
        8: "Agustus",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Desember"
    }

    dt = pd.to_datetime(date_value)
    return f"{hari[dt.weekday()]}, {dt.day} {bulan[dt.month]} {dt.year}"


def render_dynamic_css(theme):
    css = f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, #f8fcf9 0%, #f2f8f4 100%);
        }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1350px;
        }}

        .main-title {{
            font-size: 40px;
            font-weight: 800;
            color: {theme["accent_primary"]};
            margin-bottom: 6px;
            letter-spacing: 0.2px;
            line-height: 1.2;
        }}

        .subtitle-text {{
            font-size: 16px;
            color: #536c62;
            margin-bottom: 22px;
            line-height: 1.8;
            text-align: center;
            max-width: 1050px;
            margin-left: auto;
            margin-right: auto;
        }}

        .hero-box {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #ffffff 0%, {theme["accent_soft_2"]} 100%);
            border: 1px solid {theme["accent_border"]};
            border-radius: 24px;
            padding: 28px 32px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.04);
            margin-top: 0.8rem;
            margin-bottom: 20px;
            text-align: center;
        }}

        .hero-watermark {{
            position: absolute;
            right: 26px;
            bottom: -2px;
            font-size: 95px;
            font-weight: 800;
            color: rgba(0, 0, 0, 0.04);
            line-height: 1;
            pointer-events: none;
            user-select: none;
        }}

        .hero-title {{
            font-size: 30px;
            font-weight: 900;
            color: {theme["accent_primary"]};
            margin-bottom: 10px;
            line-height: 1.3;
            text-align: center;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .hero-text {{
            font-size: 15px;
            color: #4f675d;
            line-height: 1.9;
            max-width: 1000px;
            margin: 0 auto;
            text-align: center;
        }}

        .info-box {{
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, #ffffff 0%, {theme["accent_soft_2"]} 100%);
            border: 1px solid {theme["accent_border"]};
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
            margin-bottom: 18px;
        }}

        .info-box::after {{
            content: "{theme["watermark"]}";
            position: absolute;
            right: 18px;
            bottom: -8px;
            font-size: 92px;
            font-weight: 800;
            color: rgba(0, 0, 0, 0.045);
            pointer-events: none;
        }}

        .sidebar-system-title {{
            font-size: 24px;
            font-weight: 800;
            color: {theme["accent_primary"]};
            margin-bottom: 10px;
            line-height: 1.3;
        }}

        .sidebar-system-text {{
            font-size: 15px;
            color: #48645a;
            line-height: 1.8;
            margin-bottom: 18px;
        }}

        .sidebar-divider {{
            border: none;
            border-top: 1px solid {theme["accent_border"]};
            margin-top: 14px;
            margin-bottom: 18px;
        }}

        .info-title {{
            font-size: 22px;
            font-weight: 800;
            color: {theme["accent_primary"]};
            margin-bottom: 14px;
        }}

        .info-text {{
            font-size: 15px;
            line-height: 1.85;
            color: #3f5d53;
        }}

        .section-title {{
            font-size: 28px;
            font-weight: 800;
            color: {theme["accent_primary"]};
            margin-bottom: 6px;
            text-align: center;
            width: 100%;
        }}

        .section-desc {{
            font-size: 15px;
            color: #5f786d;
            margin-bottom: 10px;
            text-align: center;
            width: 100%;
            line-height: 1.7;
        }}

        .small-note {{
            font-size: 14px;
            color: #546d63;
            background: linear-gradient(90deg, #ffffff 0%, {theme["accent_soft_2"]} 100%);
            border-left: 4px solid {theme["accent_secondary"]};
            padding: 12px 14px;
            border-radius: 10px;
            margin-top: 8px;
            margin-bottom: 14px;
            line-height: 1.7;
        }}

        .badge-box {{
            text-align: center;
            margin-top: 2px;
            margin-bottom: 14px;
        }}

        .currency-badge {{
            display: inline-block;
            background: linear-gradient(90deg, {theme["accent_soft"]} 0%, {theme["accent_soft_2"]} 100%);
            color: {theme["accent_primary"]};
            padding: 10px 18px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 700;
            border: 1px solid {theme["accent_border"]};
            letter-spacing: 0.2px;
        }}

        .result-hero {{
            background: linear-gradient(135deg, #ffffff 0%, {theme["accent_soft_2"]} 100%);
            border: 1px solid {theme["accent_border"]};
            border-radius: 22px;
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.04);
        }}

        .result-hero-title {{
            font-size: 16px;
            color: #5d756b;
            text-align: center;
            margin-bottom: 6px;
            font-weight: 600;
        }}

        .result-hero-value {{
            font-size: 38px;
            font-weight: 800;
            text-align: center;
            color: {theme["accent_primary"]};
            line-height: 1.15;
            margin-bottom: 4px;
        }}

        .result-hero-sub {{
            font-size: 15px;
            text-align: center;
            color: #5a7268;
            line-height: 1.7;
        }}

        .status-line {{
            background: linear-gradient(90deg, #ffffff 0%, {theme["accent_soft_2"]} 100%);
            border: 1px solid {theme["accent_border"]};
            border-radius: 16px;
            padding: 14px 16px;
            margin-top: 10px;
            margin-bottom: 14px;
            text-align: center;
            font-size: 15px;
            color: #4d665c;
            line-height: 1.8;
        }}

        .history-title {{
            font-size: 30px;
            font-weight: 800;
            color: {theme["accent_primary"]};
            margin-top: 10px;
            margin-bottom: 10px;
            text-align: center;
            width: 100%;
        }}

        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid {theme["accent_border"]};
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center !important;
        }}

        div[data-testid="stMetric"] label {{
            width: 100% !important;
            text-align: center !important;
            display: flex !important;
            justify-content: center !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            width: 100% !important;
            text-align: center !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {{
            width: 100% !important;
            text-align: center !important;
            margin: 0 auto !important;
            color: #5b7268 !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            width: 100% !important;
            text-align: center !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            font-size: 28px !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            color: {theme["accent_primary"]} !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {{
            width: 100% !important;
            text-align: center !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
            width: 100% !important;
            text-align: center !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-top: 6px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] > div {{
            width: 100% !important;
            text-align: center !important;
            justify-content: center !important;
        }}

        div[data-testid="stTabs"] button {{
            font-size: 17px;
            font-weight: 700;
        }}

        div.stButton > button {{
            width: 100%;
            height: 54px;
            border-radius: 14px;
            border: none;
            font-size: 18px;
            font-weight: 800;
            background: linear-gradient(90deg, {theme["accent_primary"]} 0%, {theme["accent_secondary"]} 70%, #b6913e 100%);
            color: white;
            box-shadow: 0 8px 18px rgba(31, 77, 61, 0.18);
            transition: all 0.2s ease-in-out;
        }}

        div.stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(31, 77, 61, 0.22);
        }}

        div[data-testid="stDataFrame"] table {{
            width: 100% !important;
            margin: 0 auto !important;
        }}

        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td,
        div[data-testid="stDataFrame"] tbody td,
        div[data-testid="stDataFrame"] thead tr th {{
            text-align: center !important;
            vertical-align: middle !important;
        }}

        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3 {{
            text-align: center !important;
            width: 100%;
        }}

        div[data-testid="stExpander"] {{
            border-radius: 14px !important;
            border: 1px solid {theme["accent_border"]} !important;
            background-color: #ffffff !important;
        }}

        .footer-note {{
            text-align: center;
            font-size: 13px;
            color: #6d8279;
            margin-top: 30px;
            padding-top: 10px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# LOAD RESOURCE
# =========================
@st.cache_resource
def load_model_and_scaler(currency: str):
    if currency == "USD/IDR":
        model = tf.keras.models.load_model("models/usd_gru_tuning.keras")
        scaler = joblib.load("scalers/usd_scaler.pkl")
        file_path = "USD.xlsx"
        obs_file_path = "ObsUSD.xlsx"
        window_size = 14
        model_name = "GRU Tuning"
        metrics = {
            "MAE": 40.5664,
            "RMSE": 53.6434,
            "MAPE": 0.2450
        }
    else:
        model = tf.keras.models.load_model("models/cny_gru_tuning.keras")
        scaler = joblib.load("scalers/cny_scaler.pkl")
        file_path = "CNY.xlsx"
        obs_file_path = "ObsCNY.xlsx"
        window_size = 14
        model_name = "GRU Tuning"
        metrics = {
            "MAE": 5.8216,
            "RMSE": 7.6903,
            "MAPE": 0.2506
        }

    return model, scaler, file_path, obs_file_path, window_size, model_name, metrics


def get_file_mtime(file_path: str):
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    return None


def prepare_exchange_rate_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Tanggal" not in df.columns:
        raise ValueError("File data harus memiliki kolom Tanggal.")

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    if "Kurs Tengah" not in df.columns:
        required_columns = {"Kurs Jual", "Kurs Beli"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError("File data harus memiliki kolom Kurs Tengah atau kolom Kurs Jual dan Kurs Beli.")
        df["Kurs Tengah"] = (df["Kurs Jual"] + df["Kurs Beli"]) / 2

    df = df[["Tanggal", "Kurs Tengah"]].copy()
    df["Kurs Tengah"] = pd.to_numeric(df["Kurs Tengah"], errors="coerce")
    df = df.dropna(subset=["Tanggal", "Kurs Tengah"])
    df = df.sort_values("Tanggal").reset_index(drop=True)

    return df


@st.cache_data(ttl=60)
def load_and_prepare_data(file_path: str, obs_file_path: str, currency: str, main_mtime=None, obs_mtime=None):
    df_main = pd.read_excel(file_path)
    df_main = prepare_exchange_rate_dataframe(df_main)

    if os.path.exists(obs_file_path):
        df_obs = pd.read_excel(obs_file_path)
        df_obs.columns = df_obs.columns.str.strip()

        if "Mata Uang" in df_obs.columns:
            df_obs = df_obs[df_obs["Mata Uang"].astype(str).str.strip() == currency]

        df_obs = prepare_exchange_rate_dataframe(df_obs)

        df = pd.concat([df_main, df_obs], ignore_index=True)
        df = df.sort_values("Tanggal").drop_duplicates(subset=["Tanggal"], keep="last")
        df = df.reset_index(drop=True)
    else:
        df = df_main.copy()

    return df


# =========================
# CORE FUNCTION
# =========================
def predict_one_step(model, scaler, df: pd.DataFrame, window_size: int):
    if len(df) < window_size:
        raise ValueError(
            f"Jumlah data tidak mencukupi. Minimal diperlukan {window_size} data historis."
        )

    values = df[["Kurs Tengah"]].values.astype(np.float32)
    values_scaled = scaler.transform(values)

    last_window = values_scaled[-window_size:].reshape(1, window_size, 1)

    y_pred_scaled = model.predict(last_window, verbose=0)[0, 0]
    y_pred = scaler.inverse_transform(
        np.array([[y_pred_scaled]], dtype=np.float32)
    )[0, 0]

    today = get_today_date()
    last_hist_date = pd.to_datetime(df["Tanggal"].iloc[-1]).normalize()
    prediction_date = get_next_business_day(last_hist_date)

    pred_df = pd.DataFrame({
        "Tanggal": [prediction_date],
        "Prediksi Kurs": [float(y_pred)]
    })

    return pred_df, last_hist_date, today, prediction_date


def plot_historical_and_prediction(df: pd.DataFrame, pred_df: pd.DataFrame, history_points: int, theme):
    plot_df = df.tail(history_points).copy()

    hist_x = plot_df["Tanggal"]
    hist_y = plot_df["Kurs Tengah"]
    pred_x = pred_df["Tanggal"]
    pred_y = pred_df["Prediksi Kurs"]

    last_hist_date = hist_x.iloc[-1]
    last_hist_value = hist_y.iloc[-1]
    pred_date = pred_x.iloc[0]
    pred_value = pred_y.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(
        hist_x,
        hist_y,
        label="Data Historis",
        linewidth=2.5
    )

    ax.scatter(
        [last_hist_date],
        [last_hist_value],
        s=90,
        zorder=3,
        label="Nilai Historis Terakhir"
    )

    ax.plot(
        [last_hist_date, pred_date],
        [last_hist_value, pred_value],
        linestyle="--",
        linewidth=1.8,
        alpha=0.9
    )

    ax.scatter(
        pred_x,
        pred_y,
        s=130,
        marker="o",
        zorder=4,
        label="Prediksi Hari Esok"
    )

    ax.annotate(
        f"{pred_value:,.2f}",
        (pred_date, pred_value),
        textcoords="offset points",
        xytext=(0, 12),
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Kurs Tengah")
    ax.set_title(f"Grafik Historis dan Prediksi {theme['currency_label']}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()

    return fig


def get_change_summary(last_value: float, predicted_value: float):
    diff = predicted_value - last_value
    pct = (diff / last_value) * 100

    if diff > 0:
        direction = "Naik"
        interpretation = "Indikasi menguat"
        status_sentence = "Prediksi menunjukkan kecenderungan kenaikan terhadap data historis terakhir."
    elif diff < 0:
        direction = "Turun"
        interpretation = "Indikasi melemah"
        status_sentence = "Prediksi menunjukkan kecenderungan penurunan terhadap data historis terakhir."
    else:
        direction = "Tetap"
        interpretation = "Relatif stabil"
        status_sentence = "Prediksi menunjukkan kondisi yang relatif stabil dibanding data historis terakhir."

    return diff, pct, direction, interpretation, status_sentence


def build_prediction_result(
    df,
    pred_df,
    currency,
    history_points,
    window_size,
    model_name,
    metrics,
    last_hist_date,
    today,
    prediction_date
):
    last_value = float(df["Kurs Tengah"].iloc[-1])

    pred_value = float(pred_df["Prediksi Kurs"].iloc[0])
    pred_date = pred_df["Tanggal"].iloc[0]

    diff, pct, direction, interpretation, status_sentence = get_change_summary(
        last_value,
        pred_value
    )

    history_record = {
        "Waktu Akses": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mata Uang": currency,
        "Tanggal Sistem": today.strftime("%Y-%m-%d"),
        "Data Historis Terakhir": last_hist_date.strftime("%Y-%m-%d"),
        "Tanggal Prediksi": pred_date.strftime("%Y-%m-%d"),
        "Kurs Terakhir": round(last_value, 2),
        "Prediksi Hari Esok": round(pred_value, 2),
        "Selisih": round(diff, 2),
        "Perubahan (%)": round(pct, 4),
        "Arah": direction
    }

    result = {
        "currency": currency,
        "history_points": history_points,
        "window_size": window_size,
        "model_name": model_name,
        "metrics": metrics,
        "df": df.copy(),
        "pred_df": pred_df.copy(),
        "last_value": last_value,
        "pred_value": pred_value,
        "pred_date": pred_date,
        "diff": diff,
        "pct": pct,
        "direction": direction,
        "interpretation": interpretation,
        "status_sentence": status_sentence,
        "last_hist_date": last_hist_date,
        "today": today,
        "prediction_date": prediction_date
    }

    return history_record, result


# =========================
# INITIAL THEME
# =========================
currency_for_theme = st.session_state.get("selected_currency", "USD/IDR")
theme = get_theme(currency_for_theme)
render_dynamic_css(theme)


# =========================
# HEADER
# =========================
st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-watermark">{theme['watermark']}</div>
        <div class="hero-title">APLIKASI PREDIKSI NILAI TUKAR RUPIAH</div>
        <div class="hero-text">
            Aplikasi ini menampilkan prediksi nilai tukar Rupiah terhadap {theme['currency_name']} untuk hari kerja berikutnya
            menggunakan model terbaik hasil penelitian tugas akhir dengan skema one-step ahead forecasting.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">Prediksi disajikan dalam bentuk nilai utama, arah perubahan, ringkasan interpretasi, grafik historis, dan riwayat penggunaan agar mudah dipahami oleh pengguna umum maupun akademik.</div>',
    unsafe_allow_html=True
)


# =========================
# MAIN LAYOUT
# =========================
left_col, right_col = st.columns([1.05, 2.8], gap="large")

with left_col:
    st.markdown(
        f"""
        <div class="info-box">
            <div class="sidebar-system-title">Sistem Prediksi {theme['currency_label']}</div>
            <div class="sidebar-system-text">
                Sistem ini dirancang untuk membantu pengguna melihat estimasi nilai tukar Rupiah terhadap
                {theme['currency_name']} pada hari kerja berikutnya secara lebih ringkas, jelas, dan mudah dibaca.
            </div>
            <hr class="sidebar-divider">
            <div class="info-title">Informasi Aplikasi</div>
            <div class="info-text">
                • Aplikasi menggunakan model final terbaik hasil penelitian.<br><br>
                • Parameter model bersifat tetap dan tidak diubah oleh pengguna.<br><br>
                • Pengguna hanya memilih pasangan mata uang dan jumlah data historis pada grafik.<br><br>
                • Window size 14 hari digunakan sebagai input model terbaik.<br><br>
                • Prediksi ditampilkan untuk hari kerja berikutnya dari tanggal penggunaan aplikasi.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    top_tab_input, top_tab_output = st.tabs(["Input", "Output"])

    with top_tab_input:
        st.markdown('<div class="section-title">Pengaturan Input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">Silakan pilih pasangan mata uang dan jumlah data historis yang ingin ditampilkan pada grafik.</div>',
            unsafe_allow_html=True
        )

        st.divider()

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            currency = st.selectbox(
                "Pilih pasangan mata uang",
                ["USD/IDR", "CNY/IDR"],
                index=0 if currency_for_theme == "USD/IDR" else 1,
                key="selected_currency"
            )

        current_theme = get_theme(currency)
        if currency != currency_for_theme:
            st.rerun()

        with input_col2:
            history_points = st.selectbox(
                "Jumlah data historis pada grafik",
                [30, 60, 90, 180],
                index=1
            )

        theme = current_theme
        model, scaler, file_path, obs_file_path, window_size, model_name, metrics = load_model_and_scaler(currency)
        df = load_and_prepare_data(
            file_path=file_path,
            obs_file_path=obs_file_path,
            currency=currency,
            main_mtime=get_file_mtime(file_path),
            obs_mtime=get_file_mtime(obs_file_path)
        )

        st.divider()

        info_col1, info_col2, info_col3 = st.columns(3, gap="large")
        info_col1.metric("Model Prediksi", model_name)
        info_col2.metric("Panjang Input Model", f"{window_size} hari")
        info_col3.metric("Tanggal Data Historis Terakhir", df["Tanggal"].iloc[-1].strftime("%Y-%m-%d"))

        st.markdown(
            '<div class="small-note"><b>Catatan:</b> Aplikasi ini menggunakan skema one-step ahead forecasting. Model memprediksi satu nilai tukar untuk hari kerja berikutnya berdasarkan 14 data historis terakhir.</div>',
            unsafe_allow_html=True
        )

        with st.expander("Lihat Informasi Model Penelitian"):
            st.markdown("<h3 style='text-align:center;'>Metrik Evaluasi Model</h3>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3 = st.columns(3, gap="large")
            metric_col1.metric("MAE", f"{metrics['MAE']:.4f}")
            metric_col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            metric_col3.metric("MAPE (%)", f"{metrics['MAPE']:.4f}")

        st.write("")

        do_predict = st.button("Tampilkan Hasil Prediksi", use_container_width=True)

        if do_predict:
            pred_df, last_hist_date, today, prediction_date = predict_one_step(
                model=model,
                scaler=scaler,
                df=df,
                window_size=window_size
            )

            history_record, result = build_prediction_result(
                df=df,
                pred_df=pred_df,
                currency=currency,
                history_points=history_points,
                window_size=window_size,
                model_name=model_name,
                metrics=metrics,
                last_hist_date=last_hist_date,
                today=today,
                prediction_date=prediction_date
            )

            st.session_state.prediction_history.append(history_record)
            st.session_state.last_prediction_result = result

            st.success("Prediksi berhasil dibuat. Silakan lihat hasil pada tab Output.")

    with top_tab_output:
        st.markdown('<div class="section-title">Hasil Output Prediksi</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">Bagian ini menampilkan prediksi terbaru untuk hari kerja berikutnya berdasarkan pasangan mata uang yang dipilih.</div>',
            unsafe_allow_html=True
        )

        st.divider()

        if st.session_state.last_prediction_result is None:
            st.info("Belum ada hasil prediksi. Silakan lakukan prediksi terlebih dahulu pada tab Input.")
        else:
            result = st.session_state.last_prediction_result
            result_theme = get_theme(result["currency"])

            st.markdown(
                f"""
                <div class="badge-box">
                    <span class="currency-badge">Pasangan Mata Uang: {result['currency']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-hero">
                    <div class="result-hero-title">Prediksi Nilai Tukar untuk {format_date_indo(result['pred_date'])}</div>
                    <div class="result-hero-value">{result['pred_value']:,.2f}</div>
                    <div class="result-hero-sub">
                        Estimasi nilai tukar Rupiah terhadap {result_theme['currency_name']} pada hari kerja berikutnya.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            result_col1, result_col2, result_col3, result_col4 = st.columns(4, gap="large")
            result_col1.metric(
                "Kurs Terakhir",
                f"{result['last_value']:,.2f}"
            )
            result_col2.metric(
                "Prediksi Hari Esok",
                f"{result['pred_value']:,.2f}",
                f"{result['diff']:,.2f}"
            )
            result_col3.metric(
                "Perubahan (%)",
                f"{result['pct']:.4f}%"
            )
            result_col4.metric(
                "Arah Prediksi",
                result["direction"]
            )

            st.markdown(
                f"""
                <div class="status-line">
                    <b>Ringkasan:</b> {result['status_sentence']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="small-note">
                    <b>Cara membaca hasil:</b><br>
                    • Kurs Terakhir adalah nilai tukar aktual terbaru pada data historis yang tersedia.<br>
                    • Prediksi Hari Esok menunjukkan estimasi nilai tukar untuk hari kerja berikutnya dari tanggal penggunaan aplikasi.<br>
                    • Arah Prediksi menunjukkan kecenderungan perubahan terhadap data historis terakhir, yaitu {result['interpretation'].lower()}.<br>
                    • Prediksi menggunakan skema one-step ahead forecasting sehingga tetap selaras dengan model penelitian.
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style='text-align:center; line-height:2; font-size:16px; color:#4f6f64; margin-top:4px; margin-bottom:10px;'>
                    <b>Tanggal Sistem:</b> {format_date_indo(result['today'])}<br>
                    <b>Data Historis Terakhir:</b> {format_date_indo(result['last_hist_date'])}<br>
                    <b>Tanggal Prediksi:</b> {format_date_indo(result['pred_date'])}<br>
                    <b>Interpretasi Ringkas:</b> {result['interpretation']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            st.markdown("<h3 style='text-align:center;'>Grafik Historis dan Prediksi</h3>", unsafe_allow_html=True)
            fig = plot_historical_and_prediction(
                result["df"],
                result["pred_df"],
                result["history_points"],
                result_theme
            )
            st.pyplot(fig)

            st.markdown("<h3 style='text-align:center;'>Tabel Hasil Prediksi</h3>", unsafe_allow_html=True)
            pred_display = result["pred_df"].copy()
            pred_display["Tanggal"] = pred_display["Tanggal"].apply(format_date_indo)
            pred_display["Prediksi Kurs"] = pred_display["Prediksi Kurs"].round(2)
            st.dataframe(pred_display, use_container_width=True)

            st.markdown("<h3 style='text-align:center;'>Ringkasan Hasil Prediksi</h3>", unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                "Kurs Terakhir": [round(result["last_value"], 2)],
                "Prediksi Hari Esok": [round(result["pred_value"], 2)],
                "Selisih": [round(result["diff"], 2)],
                "Perubahan (%)": [round(result["pct"], 4)],
                "Arah Prediksi": [result["direction"]]
            })
            st.dataframe(summary_df, use_container_width=True)

            with st.expander("Lihat Detail Data Input dan Historis"):
                st.markdown("<h3 style='text-align:center;'>Window Input Terakhir</h3>", unsafe_allow_html=True)
                window_display = result["df"].tail(result["window_size"]).copy()
                window_display["Tanggal"] = window_display["Tanggal"].apply(format_date_indo)
                window_display["Kurs Tengah"] = window_display["Kurs Tengah"].round(2)
                st.dataframe(window_display, use_container_width=True)

                st.markdown("<h3 style='text-align:center;'>Data Historis Terakhir</h3>", unsafe_allow_html=True)
                history_display = result["df"].tail(10).copy()
                history_display["Tanggal"] = history_display["Tanggal"].apply(format_date_indo)
                history_display["Kurs Tengah"] = history_display["Kurs Tengah"].round(2)
                st.dataframe(history_display, use_container_width=True)


# =========================
# HISTORY SECTION
# =========================
active_theme = get_theme(st.session_state.get("selected_currency", "USD/IDR"))

st.markdown('<div class="history-title">Riwayat Prediksi</div>', unsafe_allow_html=True)

history_action_col1, history_action_col2 = st.columns([1, 5])

with history_action_col1:
    if st.button("Hapus Riwayat", use_container_width=True):
        st.session_state.prediction_history = []
        st.session_state.last_prediction_result = None
        st.rerun()

if len(st.session_state.prediction_history) > 0:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df = history_df.fillna("")
    history_df = history_df.iloc[::-1].reset_index(drop=True)

    display_columns = [
        "Waktu Akses",
        "Mata Uang",
        "Tanggal Prediksi",
        "Kurs Terakhir",
        "Prediksi Hari Esok",
        "Selisih",
        "Perubahan (%)",
        "Arah"
    ]
    history_df = history_df[display_columns].head(5)

    st.dataframe(history_df, use_container_width=True)
else:
    st.info("Belum ada riwayat prediksi.")

st.markdown(
    '<div class="footer-note">Sistem prediksi nilai tukar Rupiah berbasis model terbaik hasil penelitian tugas akhir.</div>',
    unsafe_allow_html=True
)