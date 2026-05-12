import os
import re
import json
import requests
import concurrent.futures
from ollama import Client
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL_NAME = "deepseek-v3.1:671b-cloud"

_api_key = os.environ.get("OLLAMA_API_KEY", "")
ollama_client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {_api_key}"}
)

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")
OPENAQ_HEADERS = {
    "X-API-Key": OPENAQ_API_KEY,
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Weather Descriptions
# ---------------------------------------------------------------------------

WEATHER_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

NOMINATIM_HEADERS = {"User-Agent": "RosGroupWeatherApp/1.0 (contact@rosgroup.com)"}

# AQI breakpoints for PM2.5 (µg/m³) — US EPA standard
AQI_BREAKPOINTS = [
    (0, 12.0,    0,  50,  "Good",            "#22c55e"),
    (12.1, 35.4, 51, 100, "Moderate",        "#eab308"),
    (35.5, 55.4, 101,150, "Unhealthy for Sensitive Groups", "#f97316"),
    (55.5, 150.4,151,200, "Unhealthy",       "#ef4444"),
    (150.5,250.4,201,300, "Very Unhealthy",  "#a855f7"),
    (250.5,500.4,301,500, "Hazardous",       "#7f1d1d"),
]

POLLUTANT_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "o3":   "Ozone (O₃)",
    "no2":  "NO₂",
    "so2":  "SO₂",
    "co":   "CO",
    "bc":   "Black Carbon",
}

POLLUTANT_UNITS = {
    "pm25": "µg/m³",
    "pm10": "µg/m³",
    "o3":   "µg/m³",
    "no2":  "µg/m³",
    "so2":  "µg/m³",
    "co":   "µg/m³",
    "bc":   "µg/m³",
}

# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def _geocode_nominatim(city: str, state: str = "", country: str = "") -> list:
    query_parts = [p for p in [city, state, country] if p]
    query = ", ".join(query_parts)
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": 10,
        "featuretype": "city",
    }
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=params,
        headers=NOMINATIM_HEADERS,
        timeout=10,
    )
    return resp.json()


def _build_location_label(result: dict) -> str:
    addr = result.get("address", {})
    parts = []
    for key in ("city", "town", "village", "hamlet", "county", "state", "country"):
        val = addr.get(key)
        if val and val not in parts:
            parts.append(val)
    return ", ".join(parts) if parts else result.get("display_name", "Unknown")

# ---------------------------------------------------------------------------
# Weather Fetch (Open-Meteo)
# ---------------------------------------------------------------------------

def _fetch_weather(lat: float, lon: float) -> dict:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "apparent_temperature,relativehumidity_2m,uv_index,visibility",
            "daily": "sunrise,sunset",
            "timezone": "auto",
            "forecast_days": 1,
        },
        timeout=10,
    )
    return resp.json()

# ---------------------------------------------------------------------------
# Air Quality Fetch (OpenAQ v3)
# ---------------------------------------------------------------------------

def _calculate_aqi_pm25(pm25: float) -> tuple[int, str, str]:
    """Return (aqi_value, category_label, hex_color) for a PM2.5 reading."""
    for c_low, c_high, i_low, i_high, label, color in AQI_BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            aqi = round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
            return aqi, label, color
    if pm25 > 500.4:
        return 500, "Hazardous", "#7f1d1d"
    return 0, "Good", "#22c55e"


def _get_sensor_latest_value(args: tuple) -> tuple | None:
    """
    Fetch the single most-recent measurement for one sensor.
    Used as the fallback strategy per OpenAQ docs:
    GET /v3/sensors/{id}/measurements?limit=1
    """
    sensor_id, info = args
    try:
        resp = requests.get(
            f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements",
            params={"limit": 1},
            headers=OPENAQ_HEADERS,
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        results = data.get("results") or []
        if not results or not isinstance(results[0], dict):
            return None
        value = results[0].get("value")
        if value is None:
            return None
        return (info["param_name"], {
            "value": round(float(value), 2),
            "unit":  info["unit"],
            "label": info["label"],
        })
    except Exception:
        return None


def _fetch_air_quality(lat: float, lon: float) -> dict:
    """
    Fetch air quality data from OpenAQ v3 API.

    Two-strategy approach (based on official docs):
    ┌──────────────────────────────────────────────────────────────┐
    │ Step 1 — GET /v3/locations?coordinates={lat},{lon}&radius=N  │
    │  → results[].sensors[].{id, parameter:{name,units}}         │
    │  → build sensor_id ──► {param_name, unit, label}            │
    ├──────────────────────────────────────────────────────────────┤
    │ Strategy A (fast)  — GET /v3/locations/{id}/latest           │
    │  → results[].{sensors_id, value}  cross-ref sensor_map      │
    ├──────────────────────────────────────────────────────────────┤
    │ Strategy B (fallback, parallel)                              │
    │  — GET /v3/sensors/{id}/measurements?limit=1  (per sensor)  │
    │  (documented example: "Fetch original measurements")        │
    └──────────────────────────────────────────────────────────────┘
    """
    KEY_PARAMS = {"pm25", "pm10", "o3", "no2", "so2", "co", "bc"}

    def _http_get(url, params=None):
        resp = requests.get(url, params=params, headers=OPENAQ_HEADERS, timeout=12)
        return resp

    try:
        # ── Step 1: Find nearby stations (try 25 km, expand to 50 km if empty) ──
        locations = []
        for search_radius in (25000, 50000):
            resp = _http_get(
                "https://api.openaq.org/v3/locations",
                params={"coordinates": f"{lat},{lon}", "radius": search_radius, "limit": 10},
            )
            if resp.status_code == 401:
                return {"error": "OpenAQ: Unauthorized — check API key."}
            if resp.status_code == 403:
                return {"error": "OpenAQ: Forbidden — API key may be invalid."}
            if resp.status_code == 429:
                return {"error": "OpenAQ: Rate limit exceeded, try again later."}
            if resp.status_code != 200:
                return {"error": f"OpenAQ returned HTTP {resp.status_code}: {resp.text[:120]}"}

            rjson = resp.json()
            if not isinstance(rjson, dict):
                return {"error": f"OpenAQ: unexpected response type ({type(rjson).__name__})."}

            locations = [loc for loc in (rjson.get("results") or []) if isinstance(loc, dict)]
            if locations:
                break   # found stations, stop expanding

        if not locations:
            return {"error": "No air quality stations found within 50 km of this location."}

        # ── Step 2: Build global sensor_id → param info map ──────────────────
        # Documented sensor structure:
        #   {"id": 23534, "name": "pm25 µg/m³",
        #    "parameter": {"id":2,"name":"pm25","units":"µg/m³","displayName":"PM2.5"}}
        sensor_map: dict[int, dict] = {}
        station_names: list[str] = []
        loc_ids: list[int] = []

        for loc in locations:
            loc_id   = loc.get("id")
            loc_name = str(loc.get("name") or "Unknown Station")
            if loc_name not in station_names:
                station_names.append(loc_name)
            if loc_id is not None:
                loc_ids.append(int(loc_id))

            for sensor in (loc.get("sensors") or []):
                if not isinstance(sensor, dict):
                    continue
                sid = sensor.get("id")
                if sid is None:
                    continue

                param = sensor.get("parameter") or {}
                if isinstance(param, dict):
                    pname = str(param.get("name") or "").lower().strip()
                    unit  = str(param.get("units") or "µg/m³").strip()
                elif isinstance(param, str):
                    pname = param.lower().strip()
                    unit  = "µg/m³"
                else:
                    pname = unit = ""

                # Last-resort: parse "pm25 µg/m³" from sensor.name
                if not pname:
                    parts = str(sensor.get("name") or "").split()
                    pname = parts[0].lower() if parts else ""
                    unit  = parts[1] if len(parts) > 1 else "µg/m³"

                if not pname:
                    continue

                sensor_map[int(sid)] = {
                    "param_name": pname,
                    "unit":       unit,
                    "label":      POLLUTANT_LABELS.get(pname, pname.upper()),
                }

        if not sensor_map:
            return {"error": "No sensors found at nearby stations."}

        pollutants: dict[str, dict] = {}

        # ── Strategy A: /locations/{id}/latest  (fast, cross-ref sensor_map) ─
        for loc_id in loc_ids[:3]:
            try:
                r = _http_get(f"https://api.openaq.org/v3/locations/{loc_id}/latest")
                if r.status_code != 200:
                    continue
                rdata = r.json()
                if not isinstance(rdata, dict):
                    continue
                for reading in (rdata.get("results") or []):
                    if not isinstance(reading, dict):
                        continue
                    sensors_id = reading.get("sensors_id")
                    value      = reading.get("value")
                    if sensors_id is None or value is None:
                        continue
                    info = sensor_map.get(int(sensors_id))
                    if not info or info["param_name"] in pollutants:
                        continue
                    try:
                        pollutants[info["param_name"]] = {
                            "value": round(float(value), 2),
                            "unit":  info["unit"],
                            "label": info["label"],
                        }
                    except (TypeError, ValueError):
                        continue
            except Exception:
                continue

        # ── Strategy B: per-sensor measurements (parallel, documented fallback) ─
        # Use when Strategy A returned nothing, or for key params still missing
        missing_key = {
            sid: info for sid, info in sensor_map.items()
            if info["param_name"] in KEY_PARAMS and info["param_name"] not in pollutants
        }
        if not pollutants or missing_key:
            target = missing_key if pollutants else {
                sid: info for sid, info in sensor_map.items()
                if info["param_name"] in KEY_PARAMS
            }
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                for result in pool.map(_get_sensor_latest_value, target.items()):
                    if result:
                        pname, pdata = result
                        if pname not in pollutants:
                            pollutants[pname] = pdata

        if not pollutants:
            return {"error": "Air quality measurements not available for this location."}

        # ── AQI from PM2.5 ────────────────────────────────────────────────────
        aqi_info = None
        if "pm25" in pollutants:
            aqi_val, aqi_cat, aqi_color = _calculate_aqi_pm25(pollutants["pm25"]["value"])
            aqi_info = {"value": aqi_val, "category": aqi_cat, "color": aqi_color}

        return {
            "pollutants":  pollutants,
            "aqi":         aqi_info,
            "station":     station_names[0] if station_names else "Nearby Station",
            "data_source": "OpenAQ API v3",
        }

    except requests.exceptions.Timeout:
        return {"error": "Air quality request timed out."}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to OpenAQ API."}
    except Exception as e:
        return {"error": f"Air quality error: {str(e)}"}



# ---------------------------------------------------------------------------
# Main Temperature + AQ tool
# ---------------------------------------------------------------------------

def get_temperature(city: str, state: str = "", country: str = "") -> dict:
    """
    Fetches real-time weather (Open-Meteo) + air quality (OpenAQ) in parallel.
    """
    try:
        results = _geocode_nominatim(city, state, country)

        if not results:
            hint = f" in {state}" if state else ""
            hint += f", {country}" if country else ""
            return {"error": f"Location '{city}{hint}' not found."}

        place_types = {"city", "town", "village", "administrative", "hamlet", "municipality"}
        filtered = [
            r for r in results
            if r.get("type", "") in place_types or r.get("class", "") == "place"
        ]
        if not filtered:
            filtered = results

        seen_labels: dict[str, dict] = {}
        for r in filtered:
            label = _build_location_label(r)
            if label not in seen_labels:
                seen_labels[label] = r

        unique_places = list(seen_labels.items())

        if len(unique_places) > 1:
            options = [
                {
                    "label": label,
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "display_name": r.get("display_name", label),
                }
                for label, r in unique_places[:8]
            ]
            return {
                "disambiguation": True,
                "query": city,
                "message": f"Multiple places named '{city}' were found. Please select the exact location:",
                "options": options,
            }

        best_label, best_result = unique_places[0]
        lat = float(best_result["lat"])
        lon = float(best_result["lon"])
        addr = best_result.get("address", {})
        resolved_country = addr.get("country", country or "")

        # Fetch weather + air quality in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_weather = executor.submit(_fetch_weather, lat, lon)
            fut_aq      = executor.submit(_fetch_air_quality, lat, lon)
            weather_raw = fut_weather.result()
            aq_data     = fut_aq.result()

        if "current_weather" not in weather_raw:
            return {"error": "Weather data unavailable for this location."}

        current    = weather_raw["current_weather"]
        temp_c     = current["temperature"]
        windspeed  = current["windspeed"]
        weathercode = current["weathercode"]
        description = WEATHER_DESCRIPTIONS.get(weathercode, "Unknown conditions")

        hourly = weather_raw.get("hourly", {})
        apparent_temps = hourly.get("apparent_temperature", [])
        feels_like = apparent_temps[0] if apparent_temps else None
        humidity   = (hourly.get("relativehumidity_2m") or [None])[0]
        uv_index   = (hourly.get("uv_index") or [None])[0]
        visibility = (hourly.get("visibility") or [None])[0]

        daily = weather_raw.get("daily", {})
        sunrise = (daily.get("sunrise") or [""])[0]
        sunset  = (daily.get("sunset") or [""])[0]

        return {
            "city": best_label,
            "country": resolved_country,
            "latitude": lat,
            "longitude": lon,
            "temperature_celsius": temp_c,
            "feels_like_celsius": feels_like,
            "wind_speed_kmh": windspeed,
            "weather_condition": description,
            "humidity_pct": humidity,
            "uv_index": uv_index,
            "visibility_m": visibility,
            "sunrise": sunrise,
            "sunset": sunset,
            "air_quality": aq_data,
            "data_source": "Open-Meteo · OpenAQ v3 · OpenStreetMap/Nominatim",
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Network error. Check internet connection."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def get_temperature_by_coords(lat: float, lon: float, label: str) -> dict:
    """Fetch weather+AQ for an already-resolved lat/lon (post-disambiguation)."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_weather = executor.submit(_fetch_weather, lat, lon)
            fut_aq      = executor.submit(_fetch_air_quality, lat, lon)
            weather_raw = fut_weather.result()
            aq_data     = fut_aq.result()

        if "current_weather" not in weather_raw:
            return {"error": "Weather data unavailable for this location."}

        current     = weather_raw["current_weather"]
        temp_c      = current["temperature"]
        windspeed   = current["windspeed"]
        weathercode = current["weathercode"]
        description = WEATHER_DESCRIPTIONS.get(weathercode, "Unknown conditions")

        hourly = weather_raw.get("hourly", {})
        feels_like = (hourly.get("apparent_temperature") or [None])[0]
        humidity   = (hourly.get("relativehumidity_2m") or [None])[0]
        uv_index   = (hourly.get("uv_index") or [None])[0]
        visibility = (hourly.get("visibility") or [None])[0]

        daily = weather_raw.get("daily", {})
        sunrise = (daily.get("sunrise") or [""])[0]
        sunset  = (daily.get("sunset") or [""])[0]

        parts = [p.strip() for p in label.split(",")]
        resolved_country = parts[-1] if len(parts) > 1 else ""

        return {
            "city": label,
            "country": resolved_country,
            "latitude": lat,
            "longitude": lon,
            "temperature_celsius": temp_c,
            "feels_like_celsius": feels_like,
            "wind_speed_kmh": windspeed,
            "weather_condition": description,
            "humidity_pct": humidity,
            "uv_index": uv_index,
            "visibility_m": visibility,
            "sunrise": sunrise,
            "sunset": sunset,
            "air_quality": aq_data,
            "data_source": "Open-Meteo · OpenAQ v3 · OpenStreetMap/Nominatim",
        }
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# ---------------------------------------------------------------------------
# Tool Registry & Schema
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {"get_temperature": get_temperature}

TEMPERATURE_TOOL_SCHEMA = {
    "name": "get_temperature",
    "description": (
        "Retrieves real-time temperature, weather conditions, and air quality data for a city. "
        "Use whenever the user asks about temperature, weather, or air quality. "
        "Pass state/country if specified to avoid ambiguity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city":    {"type": "string", "description": "City name"},
            "state":   {"type": "string", "description": "Optional state/province"},
            "country": {"type": "string", "description": "Optional country"},
        },
        "required": ["city"],
    },
}

SYSTEM_PROMPT = """You are a Weather & Air Quality Agent — part of a Real-Time Urban Environmental Monitoring System.

Your job: help users get real-time temperature, weather conditions, and air quality data for any city.

Tool available:
- get_temperature(city, state="", country=""): fetches weather + air quality.
  - Pass state/country if the user specifies them to avoid ambiguity.

INSTRUCTIONS:
1. Extract the city (and optional state/country) from the user's query.
2. Respond with ONLY a JSON tool call — nothing else:
   {"tool": "get_temperature", "parameters": {"city": "<city>", "state": "<state_if_given>", "country": "<country_if_given>"}}
   Omit state/country keys if not specified.
3. If no city is mentioned, ask the user to specify one.
"""

# ---------------------------------------------------------------------------
# Intent Detection
# ---------------------------------------------------------------------------

def detect_intent(query: str) -> str:
    q = query.lower()
    wants_f     = any(kw in q for kw in ["fahrenheit", "in °f", " °f", "in f ", "°f"])
    wants_c     = any(kw in q for kw in ["celsius", "in °c", " °c", "in c ", "°c"])
    wants_wind  = "wind" in q and not any(kw in q for kw in ["weather", "condition", "temperature", "temp"])
    wants_feels = any(kw in q for kw in ["feels like", "feel like", "apparent", "feels_like"])
    wants_cond  = any(kw in q for kw in ["condition", "sky", "cloudy", "sunny", "raining", "snowing"]) \
                  and not any(kw in q for kw in ["temperature", "temp", "degree"])
    wants_aq    = any(kw in q for kw in ["air quality", "aqi", "pollution", "pm2.5", "pm10", "pollutant"])

    if wants_aq:        return "air_quality"
    if wants_f and not wants_c: return "fahrenheit"
    if wants_c and not wants_f: return "celsius"
    if wants_wind:      return "wind"
    if wants_feels:     return "feels_like"
    if wants_cond:      return "condition"
    return "full"

# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------

def extract_tool_call(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "tool" in parsed and "parameters" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    try:
        parsed = json.loads(text)
        if "tool" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _build_weather_data(tool_result: dict, temp_c: float, temp_f: float,
                         feels_c, feels_f) -> dict:
    return {
        "city":                  tool_result["city"],
        "country":               tool_result["country"],
        "temperature_celsius":   temp_c,
        "temperature_fahrenheit": temp_f,
        "feels_like_celsius":    feels_c,
        "feels_like_fahrenheit": feels_f,
        "wind_speed_kmh":        tool_result["wind_speed_kmh"],
        "weather_condition":     tool_result["weather_condition"],
        "humidity_pct":          tool_result.get("humidity_pct"),
        "uv_index":              tool_result.get("uv_index"),
        "visibility_m":          tool_result.get("visibility_m"),
        "sunrise":               tool_result.get("sunrise", ""),
        "sunset":                tool_result.get("sunset", ""),
        "latitude":              tool_result["latitude"],
        "longitude":             tool_result["longitude"],
        "data_source":           tool_result["data_source"],
        "air_quality":           tool_result.get("air_quality"),
    }


def run_temperature_agent(user_query: str) -> dict:
    intent = detect_intent(user_query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]
    response = ollama_client.chat(model=MODEL_NAME, messages=messages)
    agent_response = response["message"]["content"]
    tool_call = extract_tool_call(agent_response)

    if not tool_call:
        clean = re.sub(r"<think>.*?</think>", "", agent_response, flags=re.DOTALL).strip()
        return {"status": "clarification", "message": clean, "weather_data": None, "intent": intent}

    tool_name   = tool_call.get("tool")
    tool_params = tool_call.get("parameters", {})

    if tool_name not in AVAILABLE_TOOLS:
        return {"status": "error", "message": f"Tool '{tool_name}' not available.", "weather_data": None, "intent": intent}

    tool_result = AVAILABLE_TOOLS[tool_name](**tool_params)

    if tool_result.get("disambiguation"):
        return {
            "status": "disambiguation",
            "message": tool_result["message"],
            "options": tool_result["options"],
            "weather_data": None,
            "intent": intent,
        }

    if "error" in tool_result:
        return {"status": "error", "message": tool_result["error"], "weather_data": None, "intent": intent}

    temp_c  = tool_result["temperature_celsius"]
    temp_f  = round(temp_c * 9 / 5 + 32, 1)
    feels_c = tool_result.get("feels_like_celsius")
    feels_f = round(feels_c * 9 / 5 + 32, 1) if feels_c is not None else None

    weather_data = _build_weather_data(tool_result, temp_c, temp_f, feels_c, feels_f)

    # Generate conversational summary for all intents
    aq = tool_result.get("air_quality", {})
    aq_snippet = ""
    if aq and not aq.get("error"):
        aqi = aq.get("aqi")
        if aqi:
            aq_snippet = f" Air quality is {aqi['category']} (AQI {aqi['value']})."

    final_prompt = (
        f'The user asked: "{user_query}"\n\n'
        f"City: {tool_result['city']}, {tool_result['country']}\n"
        f"Temperature: {temp_c}°C / {temp_f}°F\n"
        f"Feels like: {feels_c}°C / {feels_f}°F\n"
        f"Wind: {tool_result['wind_speed_kmh']} km/h\n"
        f"Conditions: {tool_result['weather_condition']}\n"
        f"{aq_snippet}\n\n"
        f"Write 1-2 natural, friendly sentences answering the user's question. "
        f"No headers, no bullet points, no JSON."
    )
    final_messages = [
        {"role": "system", "content": "You are a friendly weather assistant. 1-2 sentences only. Natural language."},
        {"role": "user",   "content": final_prompt},
    ]
    final_response = ollama_client.chat(model=MODEL_NAME, messages=final_messages)
    final_text = final_response["message"]["content"]
    final_text = re.sub(r"<think>.*?</think>", "", final_text, flags=re.DOTALL).strip()

    return {
        "status":       "success",
        "message":      final_text,
        "intent":       intent,
        "weather_data": weather_data,
    }

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    user_query = (data.get("query") or "").strip()
    if not user_query:
        return jsonify({"status": "error", "message": "Please enter a query."}), 400
    try:
        result = run_temperature_agent(user_query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Agent error: {str(e)}"}), 500


@app.route("/weather_by_coords", methods=["POST"])
def weather_by_coords():
    data = request.get_json(force=True)
    try:
        lat   = float(data["lat"])
        lon   = float(data["lon"])
        label = str(data.get("label", ""))
        original_query = str(data.get("original_query", label))
    except (KeyError, ValueError) as e:
        return jsonify({"status": "error", "message": f"Invalid request: {e}"}), 400

    try:
        tool_result = get_temperature_by_coords(lat, lon, label)
        if "error" in tool_result:
            return jsonify({"status": "error", "message": tool_result["error"], "weather_data": None})

        intent  = detect_intent(original_query)
        temp_c  = tool_result["temperature_celsius"]
        temp_f  = round(temp_c * 9 / 5 + 32, 1)
        feels_c = tool_result.get("feels_like_celsius")
        feels_f = round(feels_c * 9 / 5 + 32, 1) if feels_c is not None else None

        weather_data = _build_weather_data(tool_result, temp_c, temp_f, feels_c, feels_f)

        aq = tool_result.get("air_quality", {})
        aq_snippet = ""
        if aq and not aq.get("error"):
            aqi = aq.get("aqi")
            if aqi:
                aq_snippet = f" Air quality is {aqi['category']} (AQI {aqi['value']})."

        final_prompt = (
            f'User asked: "{original_query}" (resolved: {label})\n'
            f"Temp: {temp_c}°C / {temp_f}°F, Feels like: {feels_c}°C, "
            f"Wind: {tool_result['wind_speed_kmh']} km/h, "
            f"Conditions: {tool_result['weather_condition']}.{aq_snippet}\n"
            f"Write 1-2 friendly sentences. No headers."
        )
        final_messages = [
            {"role": "system", "content": "Friendly weather assistant. 1-2 sentences only."},
            {"role": "user",   "content": final_prompt},
        ]
        final_response = ollama_client.chat(model=MODEL_NAME, messages=final_messages)
        final_text = re.sub(r"<think>.*?</think>", "",
                            final_response["message"]["content"], flags=re.DOTALL).strip()

        return jsonify({
            "status":       "success",
            "message":      final_text,
            "intent":       intent,
            "weather_data": weather_data,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Agent error: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/debug/aq")
def debug_aq():
    """
    Debug endpoint: shows raw OpenAQ API response.
    Usage: /debug/aq?lat=51.507&lon=-0.127
    """
    try:
        lat = float(request.args.get("lat", 51.507))
        lon = float(request.args.get("lon", -0.127))
        resp = requests.get(
            "https://api.openaq.org/v3/locations",
            params={"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 3},
            headers=OPENAQ_HEADERS,
            timeout=12,
        )
        return jsonify({
            "http_status": resp.status_code,
            "response_type": type(resp.json()).__name__,
            "raw": resp.json(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "raw_text": resp.text[:500] if 'resp' in locals() else "no response"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
