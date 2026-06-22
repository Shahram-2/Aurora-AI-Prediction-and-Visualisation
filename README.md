# AuroraML: Machine Learning Aurora Forecasting Platform

## Overview

AuroraML is a full-stack artificial intelligence and space weather forecasting platform designed to predict geomagnetic activity and aurora visibility using the planetary Kp index.

The project combines machine learning, scientific data processing, live API integrations, and an interactive WebGL-based visualisation environment. Historical space weather datasets are processed through a hybrid AI pipeline consisting of XGBoost, Long Short-Term Memory (LSTM) networks, Genetic Algorithms (GA), and Particle Swarm Optimisation (PSO) to generate short and medium-range Kp forecasts.

The resulting predictions are presented through an immersive browser-based 3D Earth platform where real-world scientific datasets are projected directly onto a virtual globe, allowing users to explore aurora activity, solar conditions, atmospheric data, and space weather events in an intuitive way.

---

# Key Features

## Machine Learning Forecasting System

- Hybrid AI forecasting architecture:
  - XGBoost models for short-term Kp prediction (1–3 days).
  - Multiple LSTM architectures for medium-range forecasting.
  - Ensemble blending of model outputs.

- Advanced optimisation techniques:
  - Genetic Algorithm (GA) feature selection.
  - Particle Swarm Optimisation (PSO) hyperparameter tuning.
  - Reduction of 60+ engineered space weather features into the most significant predictors.

- Feature inputs include:
  - Historical Kp geomagnetic indices.
  - Solar wind velocity and plasma measurements.
  - Interplanetary magnetic field (IMF Bz).
  - Coronal Mass Ejection (CME) event information.
  - Temporal lag features and rolling statistical windows.
  - Solar cycle and seasonal information.

---

## Full Stack Web Visualisation Platform

AuroraML includes a browser-based scientific visualisation platform developed using modern web technologies. The frontend is centred around a 3D Earth model built with **Three.js** and **WebGL**, where live and forecasted datasets are mapped directly onto the globe as dynamic visual layers.

The platform transforms complex scientific measurements into an interactive experience, allowing users to view auroral activity, magnetic field behaviour, solar radiation, and environmental conditions from a global perspective.

### 3D Earth Rendering System

The Earth visualisation uses a multi-layer spherical rendering approach consisting of:

- A realistic Earth surface texture.
- Transparent atmospheric layers.
- Dynamic heatmap textures generated from scientific datasets.
- Aurora probability overlays.
- Magnetic and environmental data projections.

The layered approach allows a browser application to approximate three-dimensional atmospheric phenomena while maintaining high performance through GPU acceleration.

### Visualisation Modes

The platform supports multiple live data modes:

- **Aurora Mode**  
  Displays NOAA OVATION aurora probability overlays showing the location and intensity of auroral activity.

- **Sunlight & UV Mode**  
  Displays global sunlight exposure and UV radiation intensity.

- **Magnetic Field Mode**  
  Visualises geomagnetic activity and Earth's magnetic environment.

- **Atmospheric Mode**  
  Displays environmental datasets such as NO₂ concentration.

- **Live Flight Tracking**  
  A dedicated geospatial interface showing global aircraft paths, proximity to aurora regions, and potential space weather exposure levels.

---

# System Demonstration

## 3D Earth — Live Aurora Activity

The live aurora visualisation projects real-time aurora probability data directly onto the Earth model.

![Live Aurora 3D Earth](https://github.com/user-attachments/assets/bbdda212-f627-4863-b6f5-32926739d916)

---

## 3D Earth — Sunlight and UV Index

The Earth model can display live solar radiation and UV exposure using colour-mapped global overlays.

![Sunlight and UV Index](https://github.com/user-attachments/assets/3d19f63f-1f64-472e-8596-85aba9de9414)

---

## 3D Earth — Magnetic Field Visualisation

Magnetic field and space weather information are rendered as dynamic visual layers around the globe.

![Magnetic Field Visualisation](https://github.com/user-attachments/assets/440b2fac-d400-46fb-ba60-c611136ad4b7)

---

## Aviation and Aurora Exposure Monitoring

The platform includes a 2D global flight tracking system displaying real-time aircraft paths, their proximity to active auroral regions, and potential geomagnetic risk levels.

![Flight Paths and Aurora Risk](https://github.com/user-attachments/assets/ac3f2c3d-fbb8-4fb7-b3d1-22ccbbb4f879)

---

## Aurora Forecast Dashboard

The forecasting dashboard combines live space weather information with machine learning predictions.

Features include:

- Live Kp index gauge.
- 27-day forecast calendar.
- Aurora visibility calculations.
- Northern and Southern hemisphere OVATION visualisations.
- Space weather status indicators.

![Aurora Forecast Dashboard](https://github.com/user-attachments/assets/80e07e16-24b3-40ba-8325-71a32b0f5bbf)

---

# Artificial Intelligence Model Development

The machine learning pipeline includes multiple neural network architectures and optimisation algorithms. Model behaviour, training convergence, and performance are analysed using detailed visual comparisons.

## AI Training Model Comparison

Comparison of different machine learning architectures and their training behaviour.

![AI Model Comparison](https://github.com/user-attachments/assets/b0765917-d3e3-470b-9c6b-893b9f0580cb)

---

## Neural Network Training Performance

Training curves and evaluation metrics for the neural network forecasting models.

![Neural Network Performance](https://github.com/user-attachments/assets/699e7450-e530-4e00-9fba-baa20892126b)

---

# Machine Learning Forecast Output

The final AI forecasting model produces Kp index predictions which are transformed into practical aurora forecasting outputs.

These outputs include:

- Geomagnetic storm prediction.
- Aurora probability estimates.
- Future visibility conditions.
- Forecast confidence analysis.

![AI Forecast Output](https://github.com/user-attachments/assets/ee6ce57b-bb55-4935-b591-a3f8f7fae095)

---

# System Architecture

```
                     Scientific Data Sources
                               |
      ------------------------------------------------
      |                     |                        |
   GFZ Kp              NASA OMNI                NOAA DONKI
      |                     |                        |
      ------------------------------------------------
                               |
                    Data Processing Pipeline
                               |
                    Feature Engineering (60+)
                               |
                    Genetic Algorithm Selection
                               |
                  ------------------------------
                  |                            |
              XGBoost                     LSTM Models
             (Short Range)               (Medium Range)
                  |                            |
                  ------------------------------
                               |
                       Ensemble Forecast
                               |
                  Forecast Files & Live Validation
                               |
                         FastAPI Backend
                               |
                 --------------------------------
                 |                              |
          Three.js 3D Globe              Dashboard UI
                 |                              |
                 --------------------------------
                               |
                    Interactive Web Application
```

---

# Technology Stack

## Machine Learning

- Python 3.10+
- TensorFlow / Keras
- XGBoost
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

## Optimisation

- Genetic Algorithms (GA)
- Particle Swarm Optimisation (PSO)

## Frontend

- HTML5
- CSS3
- JavaScript ES6+
- Three.js
- WebGL
- Chart.js
- Leaflet.js

## Backend

- FastAPI
- REST API architecture
- Automated data retrieval systems
- Model output management

---

# Data Sources

The platform integrates multiple public scientific datasets.

| Source | Purpose |
|--------|---------|
| GFZ Potsdam | Historical Kp geomagnetic records |
| NASA OMNI | Solar wind and magnetic field data |
| NOAA DONKI | Coronal Mass Ejection (CME) events |
| NOAA SWPC | Live space weather observations |
| NOAA OVATION | Aurora probability maps |

---

# Project Structure

```
AuroraML/
│
├── ML_Pipeline/
│   ├── Data_Collection/
│   ├── Feature_Engineering/
│   ├── GA_Feature_Selection/
│   ├── XGBoost_Model/
│   ├── LSTM_Models/
│   └── Ensemble/
│
├── Web_Platform/
│   ├── ThreeJS_Globe/
│   ├── Dashboard/
│   ├── Flight_Tracker/
│   └── API_Integration/
│
├── CNN_Aurora_Classifier/
│
├── Data/
├── Models/
├── Outputs/
│   ├── Forecast_Calendar/
│   ├── Aurora_Maps/
│   └── Visibility_Tables/
│
├── server.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AuroraML.git
cd AuroraML
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Machine Learning Pipeline

Open the training notebook:

```bash
jupyter notebook AuroraML_Pipeline.ipynb
```

Pipeline workflow:

1. Download and preprocess scientific datasets.
2. Perform feature engineering.
3. Apply GA feature selection.
4. Optimise model parameters using PSO.
5. Train XGBoost and LSTM networks.
6. Generate ensemble predictions.
7. Validate against live NOAA observations.
8. Export forecast outputs.

---

# Running the Web Platform

Launch a local server:

```bash
python -m http.server 8000
```

Open:

```
http://localhost:8000
```

---

# Current Limitations

- The current deployment uses pre-generated machine learning outputs combined with live API data.
- A complete real-time FastAPI prediction service is planned for future implementation.
- Long-range aurora prediction remains constrained by the chaotic nature of solar activity and available space weather measurements.




