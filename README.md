# 🌡 Temperature Agent — Real-Time Urban Data Platform

A Flask-based agentic AI web application that answers natural language weather queries using **DeepSeek-V3.1:671b-cloud** via **Ollama**, backed by real-time data from the **Open-Meteo API**.

---

## Features

- **Natural language queries** — ask anything: *"What is the temperature in Detroit in Fahrenheit?"*
- **Smart intent detection** — if you ask for only Fahrenheit, only Celsius, wind speed, feels-like, or conditions, the UI shows *just that*, not the full report
- **Real-time weather data** — fetched live from Open-Meteo (no API key needed for weather)
- **Agentic architecture** — DeepSeek reasons, decides which tool to call, executes it, and synthesizes the final response
- **Clean UI per query** — previous results are cleared automatically on each new search

---

## Project Structure

```
.
├── app.py                  # Flask app + temperature agent logic
├── templates/
│   └── index.html          # Frontend UI
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API key)
└── README.md               # This file
```

---

## Setup

### 1. Clone / open the project

```bash
cd Final-Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Sign in to Ollama & pull the cloud model

```bash
ollama signin
ollama pull deepseek-v3.1:671b-cloud
```

### 5. Configure your API key

Copy `.env` and fill in your key (get it from [ollama.com/settings/keys](https://ollama.com/settings/keys)):

```bash
cp .env .env.local   # optional — .env is already read by default
```

Edit `.env`:

```
OLLAMA_API_KEY=your_ollama_api_key_here
```

### 6. Start Ollama server

```bash
ollama serve &
```

### 7. Run the Flask app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Example Queries

| Query | What you'll see |
|---|---|
| *"What is the temperature in Detroit in Fahrenheit?"* | Only °F value, large and prominent |
| *"How cold is London in Celsius?"* | Only °C value |
| *"Wind speed in Tokyo"* | Only wind speed in km/h |
| *"What does it feel like in Dubai?"* | Only feels-like temperature |
| *"Weather conditions in Sydney"* | Only the sky condition |
| *"Current weather in Paris"* | Full report (temp, feels-like, wind, conditions) |

---

## Architecture

```
User Query
    │
    ▼
Flask /query endpoint
    │
    ▼
detect_intent()  ──── determines: fahrenheit / celsius / wind / feels_like / condition / full
    │
    ▼
DeepSeek-V3.1:671b-cloud (via Ollama Cloud API)
    │  reasons about the query, extracts city name
    ▼
get_temperature(city)  ──── Open-Meteo Geocoding + Weather API
    │
    ▼
DeepSeek generates final natural-language summary
    │
    ▼
Flask returns { status, intent, weather_data, message }
    │
    ▼
Frontend renders only the requested data (intent-aware)
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OLLAMA_API_KEY` | Your Ollama cloud API key | Yes |
| `FLASK_HOST` | Host to bind Flask (default: `0.0.0.0`) | No |
| `FLASK_PORT` | Port for Flask (default: `5000`) | No |
| `FLASK_DEBUG` | Enable debug mode (default: `false`) | No |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Main web UI |
| `POST` | `/query` | Submit a weather query (JSON: `{"query": "..."}`) |
| `GET` | `/health` | Health check — returns model and host info |

---

## Model

**deepseek-v3.1:671b-cloud** — Run via Ollama's cloud service. No local GPU required. The model is automatically offloaded to Ollama's infrastructure.

---

## Data Source

Weather data powered by [Open-Meteo](https://open-meteo.com/) — free, no API key required for weather lookups.

---

## Ros Group — AI Systems Project
