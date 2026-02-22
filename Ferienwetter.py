import datetime as dt
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


LOCATIONS = {
    "Fislisbach":  (47.438, 8.292),
    "Hamburg":     (53.5511, 9.9937),
    "Bodø":        (67.28, 14.405),
    "Sortland":    (68.694, 15.416),
    "Harstad":     (68.798, 16.541),
    "Tromsø":      (69.6492, 18.9553),
    "Alta":        (69.9689, 23.2716),
    "Trondheim":   (63.4305, 10.3951),
    "Måløy":       (61.936, 5.113),
}

BASE_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "windspeed_10m_max",
    "precipitation_sum",
]
HOURLY_VARS = [
    "dewpoint_2m",
    "rain",
    "snowfall",
]


def build_request_params(locations, forecast_days):
    latitudes = ",".join(str(loc[0]) for loc in locations.values())
    longitudes = ",".join(str(loc[1]) for loc in locations.values())

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "timezone": "Europe/Berlin",
        "forecast_days": forecast_days,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
    }
    return params


def fetch_open_meteo(locations, forecast_days):
    params = build_request_params(locations, forecast_days)
    resp = requests.get(BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    else:
        return [data]


def summarize_location_for_index(loc_name, api_result, day_index):
    daily = api_result.get("daily", {})
    hourly = api_result.get("hourly", {})

    tmin = daily.get("temperature_2m_min", [None])[day_index]
    tmax = daily.get("temperature_2m_max", [None])[day_index]
    wind_max = daily.get("windspeed_10m_max", [None])[day_index]
    precip_sum = daily.get("precipitation_sum", [None])[day_index]

    times = hourly.get("time", [])
    dew = hourly.get("dewpoint_2m", [])
    rain = hourly.get("rain", [])
    snow = hourly.get("snowfall", [])

    dew_at_noon = None
    max_rain_24h = 0.0
    max_snow_24h = 0.0

    day_date = dt.date.fromisoformat(daily.get("time", [])[day_index])

    for t_str, d, r, s in zip(times, dew, rain, snow):
        t = dt.datetime.fromisoformat(t_str)
        if t.date() != day_date:
            continue
        if t.hour == 12:
            dew_at_noon = d
        if r is not None and r > max_rain_24h:
            max_rain_24h = r
        if s is not None and s > max_snow_24h:
            max_snow_24h = s

    return {
        "Ort": loc_name,
        "Tmin [°C]": tmin,
        "Tmax [°C]": tmax,
        "Taupunkt 12:00 [°C]": dew_at_noon,
        "max Wind [km/h]": wind_max,
        "Niederschlag Tag [mm]": precip_sum,
        "max Regen 24h [mm/h]": max_rain_24h,
        "max Schnee 24h [mm/h]": max_snow_24h,
    }


def get_forecast_table_for_date(date_obj):
    today = dt.date.today()
    target_date = date_obj
    delta_days = (target_date - today).days

    if delta_days < 0:
        raise ValueError("Datum liegt in der Vergangenheit – Historical API nötig.")
    if delta_days > 13:
        raise ValueError("Open-Meteo Forecast geht typischerweise nur ~14 Tage voraus.")

    forecast_days = max(delta_days + 1, 3)
    results = fetch_open_meteo(LOCATIONS, forecast_days)

    rows = []
    for (name, _), api_res in zip(LOCATIONS.items(), results):
        rows.append(summarize_location_for_index(name, api_res, delta_days))

    df = pd.DataFrame(rows).set_index("Ort")
    return df


def diagonal_table(df):
    n_rows, n_cols = df.shape
    vals = df.to_numpy(dtype=float)
    diag = np.full_like(vals, np.nan)

    for i in range(n_rows):
        for j in range(n_cols):
            target_col = (j + i) % n_cols
            diag[i, target_col] = vals[i, j]

    return pd.DataFrame(diag, index=df.index, columns=df.columns)


def plot_diagonal_table(diag_df, title):
    x = np.arange(len(diag_df.columns))
    fig, ax = plt.subplots(figsize=(10, 6))

    for ort, row in diag_df.iterrows():
        y = row.values.astype(float)
        ax.plot(x, y, marker="o", label=ort)

    ax.set_xticks(x)
    ax.set_xticklabels(diag_df.columns, rotation=30, ha="right")
    ax.set_ylabel("Wert")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------- Streamlit UI ----------------

def main():
    st.title("Wetter-Forecast Nordland / Norwegen")

    today = dt.date.today()
    default_date = today
    selected_date = st.date_input(
        "Datum wählen (Forecast, max. ~14 Tage)",
        value=default_date,
        min_value=today,
        max_value=today + dt.timedelta(days=13),
    )

    if st.button("Forecast holen"):
        try:
            with st.spinner("Hole Daten von Open-Meteo..."):
                df = get_forecast_table_for_date(selected_date)
                diag_df = diagonal_table(df)

            st.subheader(f"Forecast-Tabelle für {selected_date.isoformat()}")
            st.dataframe(df.style.format("{:.1f}"), use_container_width=True)



        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
