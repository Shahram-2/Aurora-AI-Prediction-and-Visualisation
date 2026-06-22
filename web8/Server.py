"""
Earth Live — Algorithm Server
==============================
Run with:  python server.py
Serves on: http://localhost:5000

Endpoints:
  POST /run/v13        — Solar wind aurora model (params: speed, bz)
  POST /run/v2         — Live NOAA space weather fetch
  POST /run/v3         — Time & location aurora forecast (params: location)
  POST /run/v31        — ML aurora prediction (params: lat, lon)
  GET  /cnn/summary    — CNN+LSTM run summary (text + model metadata)
  GET  /cnn/image/<f>  — Serve a plot PNG from aurora_output/plots/
  GET  /lr/summary     — Linear Regression summary (graphs + data file listing)
  GET  /lr/image/<f>   — Serve a graph PNG from Linear Regression/graphs/
  GET  /lr/data        — List CSV/PKL files + read CSV previews
  GET  /xgb/summary    — XGBoost & RF summary for both runs (report + file listing)
  GET  /xgb/image/<run>/<f> — Serve a plot PNG from run 1 or 2
  GET  /xgb/data/<run> — CSV previews + PKL listing for a run
  GET  /lstm/summary          — LSTM summary for all 3 variants (10, 30, kp)
  GET  /lstm/image/<v>/<f>    — Serve a plot PNG from LSTM/<variant>/graphs/
  GET  /lstm/data/<v>         — CSV previews + model file listing for a variant
  GET  /opt/summary/<run>     — Optimisation summary for a run (9.1–9.4, curve)
  GET  /opt/image/<run>/<f>   — Serve a plot PNG from optimal/<run>/
  GET  /health         — Server status check
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
import os
import pathlib
import csv
import io

# CNN data-outputs root (downloaded from Drive) 
CNN_DATA_ROOT  = pathlib.Path(r"C:\Users\shahr\Downloads\data-outputs")
CNN_OUTPUT_DIR = CNN_DATA_ROOT / "aurora_output"
CNN_PLOTS_DIR  = CNN_OUTPUT_DIR / "plots"
CNN_MODELS_DIR = CNN_DATA_ROOT / "aurora_models"

# Linear Regression data root
LR_ROOT       = pathlib.Path(r"C:\Users\shahr\Downloads\data\output-data\Linear Regression")
LR_GRAPHS_DIR = LR_ROOT / "graphs"
LR_DATA_DIR   = LR_ROOT / "data"

# XGBoost & Random Forest data root
XGB_ROOT = pathlib.Path(r"C:\Users\shahr\Downloads\data\output-data\XGBoost & Random Forest")
XGB_RUNS = ["1", "2"]          

# LSTM data root 
LSTM_ROOT     = pathlib.Path(r"C:\Users\shahr\Downloads\data\output-data\LSTM")
LSTM_VARIANTS = ["10", "30", "kp"]   

# Optimisation data root 
OPT_ROOT     = pathlib.Path(r"C:\Users\shahr\Downloads\optimal")
OPT_RUNS     = ["9.2", "9.3", "9.4", "9.5"]   
OPT_CURVE    = "curve"                          

# sklearn imports for v3.1
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)   # allow the HTML page to call from any origin



#  NOAA API URLs
KP_URL     = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
MAG_URL    = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
ALERTS_URL = "https://services.swpc.noaa.gov/products/alerts.json"
OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"


#  V1.3 — AURORA MATH MODEL

def aurora_probability(kp):
    if kp < 2:   return 0.05
    elif kp < 4: return 0.20
    elif kp < 5: return 0.40
    elif kp < 6: return 0.60
    elif kp < 7: return 0.80
    else:        return 0.95


def aurora_latitude_boundary(kp):
    return 66.5 - (kp * 3)


def solar_wind_boost(speed, bz):
    """
    speed : km/s   (typical 300–800)
    bz    : nT     (negative = southward = aurora-favourable)
    """
    if   speed < 400: speed_factor = 0.9
    elif speed < 600: speed_factor = 1.0
    else:             speed_factor = 1.2

    if   bz >  0: bz_factor = 0.7
    elif bz > -5: bz_factor = 1.0
    else:         bz_factor = 1.4

    return speed_factor * bz_factor


@app.route("/run/v13", methods=["POST"])
def run_v13():
    """
    Body JSON: { "speed": 650, "bz": -8 }
    Returns computed probability + latitude tables and parameter summary.
    """
    data  = request.get_json(force=True) or {}
    speed = float(data.get("speed", 650))
    bz    = float(data.get("bz", -8))

    boost = solar_wind_boost(speed, bz)

    kp_values   = list(range(10))
    base_probs  = []
    final_probs = []
    latitudes   = []

    for kp in kp_values:
        base  = aurora_probability(kp)
        final = min(base * boost, 1.0)
        base_probs.append(round(base * 100, 1))
        final_probs.append(round(final * 100, 1))
        latitudes.append(round(aurora_latitude_boundary(kp), 1))

    if   speed < 400: speed_label = f"slow  × 0.9"
    elif speed < 600: speed_label = f"moderate  × 1.0"
    else:             speed_label = f"fast  × 1.2"

    if   bz >  0: bz_label = "northward  × 0.7  (unfavourable)"
    elif bz > -5: bz_label = "mildly southward  × 1.0"
    else:         bz_label = "southward  × 1.4  (aurora-favourable ✓)"

    return jsonify({
        "version":      "v1.3",
        "ran_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "params": {
            "speed":       speed,
            "bz":          bz,
            "speed_label": speed_label,
            "bz_label":    bz_label,
            "boost":       round(boost, 3),
        },
        "kp_values":    kp_values,
        "base_probs":   base_probs,
        "final_probs":  final_probs,
        "latitudes":    latitudes,
    })


#  V2.0 — LIVE NOAA DATA FETCH

def fetch_kp():
    data = requests.get(KP_URL, timeout=10).json()
    latest = data[-1]
    history = [{"time": row[0][11:16], "value": float(row[1])}
               for row in data[-24:] if row[1] is not None]
    return float(latest[1]), latest[0], history


def fetch_solar_wind():
    data   = requests.get(PLASMA_URL, timeout=10).json()
    latest = data[-1]
    history = [
        {"time": row[0][11:16], "value": float(row[2])}
        for row in data[-24:]
        if row[2] is not None and row[2] != ""
    ]
    return {
        "speed":   float(latest[2]),
        "density": float(latest[1]),
        "time":    latest[0],
        "history": history,
    }


def fetch_imf():
    data   = requests.get(MAG_URL, timeout=10).json()
    latest = data[-1]
    history = [
        {"time": row[0][11:16], "value": float(row[3])}
        for row in data[-48:]
        if row[3] is not None and row[3] != ""
    ]
    return {
        "bz":      float(latest[3]),
        "bt":      float(latest[1]),
        "time":    latest[0],
        "history": history,
    }


def fetch_alerts():
    data = requests.get(ALERTS_URL, timeout=10).json()
    return [
        {
            "message": (alert.get("message") or "").split("\n")[0][:120],
            "issue_time": alert.get("issue_time", ""),
        }
        for alert in data[:5]
    ]


@app.route("/run/v2", methods=["POST"])
def run_v2():
    errors = []

    try:
        kp, kp_time, kp_history = fetch_kp()
    except Exception as e:
        errors.append(f"Kp fetch failed: {e}")
        kp, kp_time, kp_history = None, None, []

    try:
        solar = fetch_solar_wind()
    except Exception as e:
        errors.append(f"Solar wind fetch failed: {e}")
        solar = {"speed": None, "density": None, "time": None, "history": []}

    try:
        imf = fetch_imf()
    except Exception as e:
        errors.append(f"IMF fetch failed: {e}")
        imf = {"bz": None, "bt": None, "time": None, "history": []}

    try:
        alerts = fetch_alerts()
    except Exception as e:
        errors.append(f"Alerts fetch failed: {e}")
        alerts = []

    kp_label = "UNKNOWN"
    if kp is not None:
        if   kp >= 7: kp_label = "SEVERE STORM"
        elif kp >= 5: kp_label = "STORM"
        elif kp >= 4: kp_label = "ACTIVE"
        elif kp >= 3: kp_label = "UNSETTLED"
        else:         kp_label = "QUIET"

    bz_label = ""
    if imf["bz"] is not None:
        bz = imf["bz"]
        if   bz < -10: bz_label = "strong southward — excellent aurora conditions"
        elif bz <  -5: bz_label = "southward — aurora activity likely"
        elif bz <   0: bz_label = "mildly southward — monitor"
        else:          bz_label = "northward — aurora unlikely"

    return jsonify({
        "version":   "v2.0",
        "ran_at":    datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "errors":    errors,
        "kp": {
            "value":   kp,
            "time":    kp_time,
            "label":   kp_label,
            "history": kp_history,
        },
        "solar_wind": solar,
        "imf": {
            **imf,
            "bz_label": bz_label,
        },
        "alerts": alerts,
    })


#  V3.0 — TIME & LOCATION AURORA FORECAST

def geocode_location(place_name):
    """Use Nominatim (OpenStreetMap) to resolve a place name to lat/lon."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "EarthLive/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=10).json()
    if not resp:
        raise ValueError(f"Location not found: {place_name}")
    return float(resp[0]["lat"]), float(resp[0]["lon"]), resp[0].get("display_name", place_name)


def auroral_oval_latitude_v3(kp):
    return 67 - kp * 2


def night_factor(local_hour):
    if 22 <= local_hour or local_hour <= 4:
        return 1.0
    if (20 <= local_hour <= 22) or (4 <= local_hour <= 6):
        return 0.6
    return 0.2


def aurora_prob_v3(lat, kp, speed, bz, hour):
    oval_lat = auroral_oval_latitude_v3(kp)
    lat_distance = abs(lat - oval_lat)

    if lat_distance > 15:
        base = 0.05
    else:
        base = max(0, 1 - lat_distance / 15)

    if speed > 600:  base += 0.15
    if bz < -5:      base += 0.20
    if bz < -10:     base += 0.20

    base *= night_factor(hour)
    return round(min(base, 1.0) * 100, 1)


@app.route("/run/v3", methods=["POST"])
def run_v3():
    """
    Body JSON: { "location": "London" }
    Geocodes the location, fetches live Kp + solar wind, returns 8-hour forecast.
    """
    data     = request.get_json(force=True) or {}
    location = data.get("location", "").strip()
    if not location:
        return jsonify({"error": "location parameter is required"}), 400

    errors = []

    # Geocode
    try:
        lat, lon, display_name = geocode_location(location)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Live space weather
    try:
        kp_data = requests.get(KP_URL, timeout=10).json()
        kp = float(kp_data[-1][1])
    except Exception as e:
        errors.append(f"Kp fetch failed: {e}")
        kp = 3.0

    try:
        plasma = requests.get(PLASMA_URL, timeout=10).json()
        speed  = float(plasma[-1][2])
        density = float(plasma[-1][1])
    except Exception as e:
        errors.append(f"Plasma fetch failed: {e}")
        speed, density = 450.0, 5.0

    try:
        mag = requests.get(MAG_URL, timeout=10).json()
        bz  = float(mag[-1][6])
        bt  = float(mag[-1][1])
    except Exception as e:
        errors.append(f"IMF fetch failed: {e}")
        bz, bt = -2.0, 5.0

    # 8-hour forecast every 2 hours
    now = datetime.now(timezone.utc)
    forecast = []
    for h in range(0, 8, 2):
        t    = now + timedelta(hours=h)
        prob = aurora_prob_v3(lat, kp, speed, bz, t.hour)
        forecast.append({
            "time_utc": t.strftime("%H:%M"),
            "hours_ahead": h,
            "probability": prob,
            "night_factor": night_factor(t.hour),
        })

    # Conditions summary
    if   bz < -10: bz_label = "strong southward — excellent"
    elif bz <  -5: bz_label = "southward — favourable"
    elif bz <   0: bz_label = "mildly southward"
    else:          bz_label = "northward — unfavourable"

    oval_lat = auroral_oval_latitude_v3(kp)
    lat_dist = abs(lat - oval_lat)
    if lat_dist <= 3:   proximity = "Inside auroral oval"
    elif lat_dist <= 8: proximity = "Near auroral oval"
    elif lat_dist <= 15: proximity = "Within range"
    else:               proximity = "Outside typical range"

    return jsonify({
        "version":      "v3.0",
        "ran_at":       datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "errors":       errors,
        "location": {
            "query":        location,
            "display_name": display_name,
            "lat":          round(lat, 3),
            "lon":          round(lon, 3),
        },
        "space_weather": {
            "kp":      kp,
            "speed":   round(speed, 1),
            "density": round(density, 1),
            "bz":      round(bz, 1),
            "bt":      round(bt, 1),
            "bz_label": bz_label,
        },
        "auroral_oval_lat": round(oval_lat, 1),
        "lat_distance":     round(lat_dist, 1),
        "proximity":        proximity,
        "forecast":         forecast,
    })


#  V3.1 — ML AURORA PREDICTION (Random Forest)

def kp_to_label(kp):
    if kp <= 2:  return "weak"
    elif kp <= 5: return "moderate"
    else:         return "strong"


def build_and_train_model():
    """Build + train a Random Forest on 500 synthetic samples."""
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(500):
        kp = rng.integers(0, 9)
        samples.append({
            "speed":   float(rng.normal(450 + kp * 30, 50)),
            "density": float(rng.normal(5 + kp * 0.5, 1)),
            "bz":      float(rng.normal(-kp, 2)),
            "bt":      float(rng.normal(5 + kp, 1)),
            "label":   kp_to_label(kp),
        })

    X = np.array([[s["speed"], s["density"], s["bz"], s["bt"]] for s in samples])
    y = [s["label"] for s in samples]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler


@app.route("/run/v31", methods=["POST"])
def run_v31():
    """
    Body JSON: { "lat": 55.9, "lon": -3.2 }
    Trains RF model, fetches live NOAA data, returns aurora class probs,
    formation %, visibility %, colour composition, latitude reach.
    """
    data = request.get_json(force=True) or {}
    lat  = float(data.get("lat",  55.9))
    lon  = float(data.get("lon", -3.2))

    errors = []

    # Live data
    try:
        kp_data = requests.get(KP_URL, timeout=10).json()
        kp_now = float(kp_data[-1][1])
    except Exception as e:
        errors.append(f"Kp fetch failed: {e}")
        kp_now = 3.0

    try:
        plasma  = requests.get(PLASMA_URL, timeout=10).json()
        mag     = requests.get(MAG_URL,    timeout=10).json()
        speed   = float(plasma[-1][2])
        density = float(plasma[-1][1])
        bz      = float(mag[-1][6])
        bt      = float(mag[-1][1])
    except Exception as e:
        errors.append(f"Solar wind fetch failed: {e}")
        speed, density, bz, bt = 450.0, 5.0, -2.0, 5.0

    # Train model + predict
    clf, scaler = build_and_train_model()
    X_live = scaler.transform([[speed, density, bz, bt]])
    probs  = clf.predict_proba(X_live)[0]
    classes = list(clf.classes_)
    class_probs = dict(zip(classes, [round(float(p) * 100, 1) for p in probs]))

    # Derived physics
    cp = {c: float(p) for c, p in zip(classes, probs)}
    formation_pct = (
        cp.get("weak",     0) * 0.30 +
        cp.get("moderate", 0) * 0.70 +
        cp.get("strong",   0) * 0.95
    ) * 100

    vis = 0.4
    if speed > 500:  vis += 0.2
    if speed > 650:  vis += 0.2
    if bz < -5:      vis += 0.2
    if bz < -10:     vis += 0.3
    visibility_pct = round(min(vis, 1.0) * 100, 1)

    green  = min(0.8, 0.4 + kp_now * 0.05)
    red    = min(0.7, max(0.0, (kp_now - 4) * 0.08))
    purple = min(0.6, max(0.0, (-bz - 5) * 0.07))
    total  = green + red + purple or 1
    colour_probs = {
        "green":  round(green  / total * 100, 1),
        "red":    round(red    / total * 100, 1),
        "purple": round(purple / total * 100, 1),
    }

    latitude_reach = round(max(45.0, 67.0 - kp_now * 2), 1)

    # Visibility from observer's latitude
    lat_reach_diff = abs(lat) - latitude_reach
    if lat_reach_diff <= 0:
        lat_visibility = "Aurora may reach your latitude"
    elif lat_reach_diff <= 5:
        lat_visibility = "Near the edge — monitor closely"
    else:
        lat_visibility = f"Aurora unlikely to reach {abs(lat):.1f}° (reach: {latitude_reach}°)"

    # Conditions label
    if   bz < -10: bz_label = "strong southward — excellent"
    elif bz <  -5: bz_label = "southward — favourable"
    elif bz <   0: bz_label = "mildly southward"
    else:          bz_label = "northward — unfavourable"

    return jsonify({
        "version":        "v3.1",
        "ran_at":         datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "errors":         errors,
        "location":       {"lat": lat, "lon": lon},
        "space_weather": {
            "kp":      kp_now,
            "speed":   round(speed, 1),
            "density": round(density, 1),
            "bz":      round(bz, 1),
            "bt":      round(bt, 1),
            "bz_label": bz_label,
        },
        "class_probs":    class_probs,
        "formation_pct":  round(formation_pct, 1),
        "visibility_pct": visibility_pct,
        "colour_probs":   colour_probs,
        "latitude_reach": latitude_reach,
        "lat_visibility": lat_visibility,
    })


#  CNN+LSTM — DATA OUTPUTS 

@app.route("/cnn/summary", methods=["GET"])
def cnn_summary():
    """
    Read run_summary.txt and list available .keras model files.
    Returns structured JSON the frontend can display directly.
    """
    summary_text = None
    summary_path = CNN_OUTPUT_DIR / "run_summary.txt"
    if summary_path.exists():
        summary_text = summary_path.read_text(encoding="utf-8")

    # Parse key-value lines from the summary into a dict for easy rendering
    parsed = {}
    if summary_text:
        for line in summary_text.splitlines():
            line = line.strip()
            if ":" in line and not line.startswith("=") and not line.startswith("*"):
                k, _, v = line.partition(":")
                k = k.strip().lstrip("★* ")
                v = v.strip()
                if k and v:
                    parsed[k] = v

    # List available .keras model files
    models = []
    if CNN_MODELS_DIR.exists():
        for f in sorted(CNN_MODELS_DIR.iterdir()):
            if f.suffix in (".keras", ".tflite"):
                size_kb = round(f.stat().st_size / 1024, 1)
                models.append({"name": f.name, "size_kb": size_kb})

    # List available plots
    plots = []
    if CNN_PLOTS_DIR.exists():
        for f in sorted(CNN_PLOTS_DIR.glob("*.png")):
            plots.append(f.name)

    return jsonify({
        "summary_raw":  summary_text,
        "summary_parsed": parsed,
        "models":       models,
        "plots":        plots,
        "plots_dir":    str(CNN_PLOTS_DIR),
        "data_root":    str(CNN_DATA_ROOT),
        "data_available": CNN_DATA_ROOT.exists(),
    })


@app.route("/cnn/image/<filename>", methods=["GET"])
def cnn_image(filename):
    """Serve a plot PNG from aurora_output/plots/."""
    # Sanitise — only allow simple filenames, no path traversal
    filename = pathlib.Path(filename).name
    if not filename.endswith(".png"):
        return jsonify({"error": "Only .png files are served here"}), 400
    path = CNN_PLOTS_DIR / filename
    if not path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    return send_file(str(path), mimetype="image/png")



#  LINEAR REGRESSION — DATA OUTPUTS 

@app.route("/lr/summary", methods=["GET"])
def lr_summary():
    """
    Returns list of graph PNGs, data files, and folder availability.
    """
    graphs, data_files = [], []

    if LR_GRAPHS_DIR.exists():
        graphs = sorted(f.name for f in LR_GRAPHS_DIR.iterdir() if f.suffix.lower() == ".png")

    if LR_DATA_DIR.exists():
        for f in sorted(LR_DATA_DIR.iterdir()):
            if f.suffix.lower() in (".csv", ".pkl", ".pickle"):
                size_kb = round(f.stat().st_size / 1024, 1)
                data_files.append({"name": f.name, "ext": f.suffix.lower().lstrip("."), "size_kb": size_kb})

    return jsonify({
        "data_available":  LR_ROOT.exists(),
        "graphs_available": LR_GRAPHS_DIR.exists(),
        "data_available_dir": LR_DATA_DIR.exists(),
        "data_root":       str(LR_ROOT),
        "graphs":          graphs,
        "data_files":      data_files,
    })


@app.route("/lr/image/<filename>", methods=["GET"])
def lr_image(filename):
    """Serve a PNG from Linear Regression/graphs/."""
    filename = pathlib.Path(filename).name
    if not filename.lower().endswith(".png"):
        return jsonify({"error": "Only .png files served here"}), 400
    path = LR_GRAPHS_DIR / filename
    if not path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/lr/data", methods=["GET"])
def lr_data():
    """
    Returns metadata for all data files plus a preview of any CSV files
    (first 8 rows, all columns).
    """
    files = []
    if not LR_DATA_DIR.exists():
        return jsonify({"error": "data directory not found", "path": str(LR_DATA_DIR)}), 404

    for f in sorted(LR_DATA_DIR.iterdir()):
        ext = f.suffix.lower()
        if ext not in (".csv", ".pkl", ".pickle"):
            continue
        entry = {
            "name":    f.name,
            "ext":     ext.lstrip("."),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "preview": None,
            "columns": None,
            "rows":    None,
        }
        if ext == ".csv":
            try:
                with open(f, newline="", encoding="utf-8-sig") as fh:
                    reader = csv.DictReader(fh)
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= 8:
                            break
                        rows.append(dict(row))
                    entry["columns"] = reader.fieldnames or []
                    entry["preview"] = rows
                # Count total rows quickly
                with open(f, encoding="utf-8-sig") as fh:
                    entry["rows"] = sum(1 for _ in fh) - 1  # minus header
            except Exception as e:
                entry["preview"] = [{"error": str(e)}]
        elif ext in (".pkl", ".pickle"):
            # Don't load pkl (may require model classes) — just report size
            entry["preview"] = None
        files.append(entry)

    return jsonify({"files": files, "data_dir": str(LR_DATA_DIR)})


#  XGBOOST & RANDOM FOREST — DATA OUTPUTS

def _xgb_run_path(run: str) -> pathlib.Path:
    """Resolve and validate a run sub-folder (1 or 2 only)."""
    if run not in XGB_RUNS:
        return None
    return XGB_ROOT / run


@app.route("/xgb/summary", methods=["GET"])
def xgb_summary():
    """
    For each run (1, 2) returns:
      - report text (.txt in reports/)
      - list of plot PNGs (plots/)
      - list of data files (data/ CSVs + models/ PKLs)
    """
    runs_out = {}
    for run in XGB_RUNS:
        rp = _xgb_run_path(run)
        entry = {
            "available":  rp is not None and rp.exists(),
            "run":        run,
            "report":     None,
            "plots":      [],
            "data_files": [],
            "model_files":[],
        }
        if entry["available"]:
            # Report
            reports_dir = rp / "reports"
            if reports_dir.exists():
                for txt in sorted(reports_dir.glob("*.txt")):
                    entry["report"] = txt.read_text(encoding="utf-8", errors="replace")
                    break  # take the first/only txt

            # Plots
            plots_dir = rp / "plots"
            if plots_dir.exists():
                entry["plots"] = sorted(f.name for f in plots_dir.iterdir()
                                        if f.suffix.lower() == ".png")

            # Data CSVs
            data_dir = rp / "data"
            if data_dir.exists():
                for f in sorted(data_dir.iterdir()):
                    if f.suffix.lower() == ".csv":
                        entry["data_files"].append({
                            "name":    f.name,
                            "size_kb": round(f.stat().st_size / 1024, 1),
                        })

            # Model PKLs
            models_dir = rp / "models"
            if models_dir.exists():
                for f in sorted(models_dir.iterdir()):
                    if f.suffix.lower() in (".pkl", ".pickle"):
                        entry["model_files"].append({
                            "name":    f.name,
                            "size_kb": round(f.stat().st_size / 1024, 1),
                        })

        runs_out[run] = entry

    return jsonify({
        "data_root":    str(XGB_ROOT),
        "data_available": XGB_ROOT.exists(),
        "runs":         runs_out,
    })


@app.route("/xgb/image/<run>/<filename>", methods=["GET"])
def xgb_image(run, filename):
    """Serve a PNG from XGBoost & RF / <run> / plots/."""
    rp = _xgb_run_path(run)
    if rp is None:
        return jsonify({"error": "Invalid run identifier"}), 400
    filename = pathlib.Path(filename).name
    if not filename.lower().endswith(".png"):
        return jsonify({"error": "Only .png files served here"}), 400
    path = rp / "plots" / filename
    if not path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/xgb/data/<run>", methods=["GET"])
def xgb_data(run):
    """Return CSV previews (8 rows) + PKL listing for a given run."""
    rp = _xgb_run_path(run)
    if rp is None:
        return jsonify({"error": "Invalid run identifier"}), 400
    if not rp.exists():
        return jsonify({"error": f"Run directory not found: {rp}"}), 404

    files = []
    data_dir = rp / "data"
    if data_dir.exists():
        for f in sorted(data_dir.iterdir()):
            ext = f.suffix.lower()
            if ext not in (".csv", ".pkl", ".pickle"):
                continue
            entry = {
                "name":    f.name,
                "ext":     ext.lstrip("."),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "preview": None,
                "columns": None,
                "rows":    None,
            }
            if ext == ".csv":
                try:
                    with open(f, newline="", encoding="utf-8-sig") as fh:
                        reader = csv.DictReader(fh)
                        rows = []
                        for i, row in enumerate(reader):
                            if i >= 8:
                                break
                            rows.append(dict(row))
                        entry["columns"] = reader.fieldnames or []
                        entry["preview"] = rows
                    with open(f, encoding="utf-8-sig") as fh:
                        entry["rows"] = sum(1 for _ in fh) - 1
                except Exception as e:
                    entry["preview"] = [{"error": str(e)}]
            files.append(entry)

    return jsonify({"run": run, "files": files, "data_dir": str(data_dir)})



#  LSTM — DATA OUTPUTS

LSTM_VARIANT_LABELS = {
    "10":  "10-step ahead forecast",
    "30":  "30-step ahead forecast",
    "kp":  "Kp-index direct prediction",
}

def _lstm_variant_path(variant: str) -> pathlib.Path | None:
    """Resolve and validate an LSTM variant sub-folder (10 / 30 / kp only)."""
    if variant not in LSTM_VARIANTS:
        return None
    return LSTM_ROOT / variant


@app.route("/lstm/summary", methods=["GET"])
def lstm_summary():
    """
    Returns per-variant availability, graph list, and data-file listing
    for all three LSTM variants (10, 30, kp).
    """
    variants_out = {}
    for variant in LSTM_VARIANTS:
        vp = _lstm_variant_path(variant)
        graphs_dir = vp / "graphs"
        data_dir   = vp / "data"

        graphs, data_files, model_files = [], [], []

        if graphs_dir.exists():
            graphs = sorted(
                f.name for f in graphs_dir.iterdir()
                if f.suffix.lower() == ".png"
            )

        if data_dir.exists():
            for f in sorted(data_dir.iterdir()):
                ext = f.suffix.lower()
                if ext == ".csv":
                    data_files.append({
                        "name":    f.name,
                        "ext":     "csv",
                        "size_kb": round(f.stat().st_size / 1024, 1),
                    })
                elif ext in (".keras",):
                    model_files.append({
                        "name":    f.name,
                        "ext":     "keras",
                        "size_kb": round(f.stat().st_size / 1024, 1),
                    })
                elif ext in (".pkl", ".pickle"):
                    model_files.append({
                        "name":    f.name,
                        "ext":     "pkl",
                        "size_kb": round(f.stat().st_size / 1024, 1),
                    })

        variants_out[variant] = {
            "available":        vp is not None and vp.exists(),
            "variant":          variant,
            "label":            LSTM_VARIANT_LABELS.get(variant, variant),
            "graphs_available": graphs_dir.exists(),
            "data_available":   data_dir.exists(),
            "graphs":           graphs,
            "data_files":       data_files,
            "model_files":      model_files,
        }

    return jsonify({
        "data_root":      str(LSTM_ROOT),
        "data_available": LSTM_ROOT.exists(),
        "variants":       variants_out,
    })


@app.route("/lstm/image/<variant>/<filename>", methods=["GET"])
def lstm_image(variant, filename):
    """Serve a PNG from LSTM/<variant>/graphs/."""
    vp = _lstm_variant_path(variant)
    if vp is None:
        return jsonify({"error": f"Unknown variant '{variant}'. Use: 10, 30, kp"}), 400
    filename = pathlib.Path(filename).name          # strip any path traversal
    if not filename.lower().endswith(".png"):
        return jsonify({"error": "Only .png files served here"}), 400
    path = vp / "graphs" / filename
    if not path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/lstm/data/<variant>", methods=["GET"])
def lstm_data(variant):
    """
    Return CSV previews (first 8 rows) + PKL / Keras file listing
    for LSTM/<variant>/data/.
    """
    vp = _lstm_variant_path(variant)
    if vp is None:
        return jsonify({"error": f"Unknown variant '{variant}'. Use: 10, 30, kp"}), 400
    data_dir = vp / "data"
    if not data_dir.exists():
        return jsonify({"error": f"Data directory not found: {data_dir}"}), 404

    files = []
    for f in sorted(data_dir.iterdir()):
        ext = f.suffix.lower()
        if ext not in (".csv", ".pkl", ".pickle", ".keras"):
            continue
        entry = {
            "name":    f.name,
            "ext":     ext.lstrip("."),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "preview": None,
            "columns": None,
            "rows":    None,
        }
        if ext == ".csv":
            try:
                with open(f, newline="", encoding="utf-8-sig") as fh:
                    reader = csv.DictReader(fh)
                    rows = []
                    for i, row in enumerate(reader):
                        if i >= 8:
                            break
                        rows.append(dict(row))
                    entry["columns"] = reader.fieldnames or []
                    entry["preview"] = rows
                with open(f, encoding="utf-8-sig") as fh:
                    entry["rows"] = sum(1 for _ in fh) - 1
            except Exception as e:
                entry["preview"] = [{"error": str(e)}]
        # .pkl / .keras — report size only, don't deserialise
        files.append(entry)

    return jsonify({
        "variant":  variant,
        "label":    LSTM_VARIANT_LABELS.get(variant, variant),
        "files":    files,
        "data_dir": str(data_dir),
    })




#  OPTIMISATION — DATA OUTPUTS


def _opt_run_path(run: str):
    """Resolve and validate an optimisation run sub-folder."""
    allowed = OPT_RUNS + [OPT_CURVE]
    if run not in allowed:
        return None
    return OPT_ROOT / run


@app.route("/opt/summary/<run>", methods=["GET"])
def opt_summary(run):
    """
    Returns graph PNG list and (for 'curve') CSV file listing + previews.
    Valid run values: 9.1, 9.2, 9.3, 9.4, curve
    """
    rp = _opt_run_path(run)
    if rp is None:
        return jsonify({"error": f"Unknown run '{run}'. Use: 9.2, 9.3, 9.4, 9.5, curve"}), 400

    graphs, csv_files = [], []

    if rp.exists():
        graphs = sorted(f.name for f in rp.iterdir() if f.suffix.lower() == ".png")

        if run == OPT_CURVE:
            for f in sorted(rp.iterdir()):
                if f.suffix.lower() != ".csv":
                    continue
                entry = {
                    "name":    f.name,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "preview": None,
                    "columns": None,
                    "rows":    None,
                }
                try:
                    with open(f, newline="", encoding="utf-8-sig") as fh:
                        reader = csv.DictReader(fh)
                        rows = [dict(r) for i, r in enumerate(reader) if i < 8]
                        entry["columns"] = reader.fieldnames or []
                        entry["preview"] = rows
                    with open(f, encoding="utf-8-sig") as fh:
                        entry["rows"] = sum(1 for _ in fh) - 1
                except Exception as e:
                    entry["preview"] = [{"error": str(e)}]
                csv_files.append(entry)

    return jsonify({
        "run":       run,
        "available": rp.exists(),
        "path":      str(rp),
        "graphs":    graphs,
        "csv_files": csv_files,
    })


@app.route("/opt/image/<run>/<filename>", methods=["GET"])
def opt_image(run, filename):
    """Serve a PNG from optimal/<run>/."""
    rp = _opt_run_path(run)
    if rp is None:
        return jsonify({"error": f"Unknown run '{run}'"}), 400
    filename = pathlib.Path(filename).name       # strip path traversal
    if not filename.lower().endswith(".png"):
        return jsonify({"error": "Only .png files served here"}), 400
    path = rp / filename
    if not path.exists():
        return jsonify({"error": f"File not found: {filename}"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "server": "Earth Live Algorithm Server", "time": datetime.utcnow().isoformat()})




if __name__ == "__main__":
    print("\n  Earth Live — Algorithm Server")
    print("  ─────────────────────────────")
    print("  Running on http://localhost:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=True)