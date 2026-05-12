import os
import requests
import json
import re
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

# ---------------------------------------------------------------------------
# Temperature Tool
# ---------------------------------------------------------------------------

def get_temperature(city: str) -> dict:
    """Fetches real-time temperature for a given city using the Open-Meteo API."""
    try:
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
        geo_response = requests.get(geocode_url, params=geo_params, timeout=10)
        geo_data = geo_response.json()

        if "results" not in geo_data or len(geo_data["results"]) == 0:
            return {"error": f"City '{city}' not found. Please check the city name."}

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        country = location.get("country", "")
        resolved_name = location.get("name", city)

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "hourly": "temperature_2m,relativehumidity_2m,apparent_temperature",
            "forecast_days": 1,
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_data = weather_response.json()

        if "current_weather" not in weather_data:
            return {"error": "Weather data unavailable for this location."}

        current = weather_data["current_weather"]
        temp_c = current["temperature"]
        windspeed = current["windspeed"]
        weathercode = current["weathercode"]

        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
        }
        description = weather_descriptions.get(weathercode, "Unknown conditions")

        hourly = weather_data.get("hourly", {})
        apparent_temps = hourly.get("apparent_temperature", [])
        feels_like = apparent_temps[0] if apparent_temps else None

        return {
            "city": resolved_name,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "temperature_celsius": temp_c,
            "feels_like_celsius": feels_like,
            "wind_speed_kmh": windspeed,
            "weather_condition": description,
            "data_source": "Open-Meteo API (Real-Time)",
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Network error. Check internet connection."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ---------------------------------------------------------------------------
# Tool Registry & Schema
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {"get_temperature": get_temperature}

TEMPERATURE_TOOL_SCHEMA = {
    "name": "get_temperature",
    "description": (
        "Retrieves real-time temperature and current weather conditions for a specified city. "
        "Use this tool whenever the user asks about temperature, current weather, "
        "how hot or cold it is, or weather conditions in any city or location."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get temperature for (e.g., 'New York', 'London', 'Tokyo')",
            }
        },
        "required": ["city"],
    },
}

SYSTEM_PROMPT = """You are a Temperature Agent — part of a Real-Time Urban Environmental Monitoring System.

Your ONLY job is to help users get real-time temperature and weather information for any city.

You have access to this tool:
- get_temperature(city: str): Fetches live temperature, feels-like temperature, wind speed, and weather conditions for a city.

INSTRUCTIONS:
1. When the user asks about temperature or weather in a city, extract the city name.
2. Respond with ONLY a JSON tool call in this exact format (nothing else):
   {"tool": "get_temperature", "parameters": {"city": "<city_name>"}}
3. Do NOT add any explanation before or after the JSON when making a tool call.
4. If the user's query is unclear or no city is mentioned, ask them to specify a city.
"""


# ---------------------------------------------------------------------------
# Intent Detection
# ---------------------------------------------------------------------------

def detect_intent(query: str) -> str:
    """
    Detect what the user specifically wants from the query.
    Returns one of: 'fahrenheit', 'celsius', 'wind', 'feels_like', 'condition', 'full'
    """
    q = query.lower()
    wants_f = any(kw in q for kw in ["fahrenheit", "in °f", " °f", "in f ", "°f"])
    wants_c = any(kw in q for kw in ["celsius", "in °c", " °c", "in c ", "°c"])
    wants_wind = "wind" in q and not any(kw in q for kw in ["weather", "condition", "temperature", "temp"])
    wants_feels = any(kw in q for kw in ["feels like", "feel like", "apparent", "feels_like"])
    wants_condition = any(kw in q for kw in ["condition", "sky", "cloudy", "sunny", "raining", "snowing"]) \
                      and not any(kw in q for kw in ["temperature", "temp", "degree"])

    if wants_f and not wants_c:
        return "fahrenheit"
    if wants_c and not wants_f:
        return "celsius"
    if wants_wind:
        return "wind"
    if wants_feels:
        return "feels_like"
    if wants_condition:
        return "condition"
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


def run_temperature_agent(user_query: str) -> dict:
    """
    Run the temperature agent and return a structured result dict.
    Returns keys: status, message, weather_data (optional), intent
    """
    intent = detect_intent(user_query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    response = ollama_client.chat(model=MODEL_NAME, messages=messages)
    agent_response = response["message"]["content"]

    tool_call = extract_tool_call(agent_response)

    if not tool_call:
        clean = re.sub(r"<think>.*?</think>", "", agent_response, flags=re.DOTALL).strip()
        return {"status": "clarification", "message": clean, "weather_data": None, "intent": intent}

    tool_name = tool_call.get("tool")
    tool_params = tool_call.get("parameters", {})

    if tool_name not in AVAILABLE_TOOLS:
        return {"status": "error", "message": f"Tool '{tool_name}' not available.", "weather_data": None, "intent": intent}

    tool_result = AVAILABLE_TOOLS[tool_name](**tool_params)

    if "error" in tool_result:
        return {"status": "error", "message": tool_result["error"], "weather_data": None, "intent": intent}

    # Build Celsius → Fahrenheit conversions
    temp_c = tool_result["temperature_celsius"]
    temp_f = round(temp_c * 9 / 5 + 32, 1)
    feels_c = tool_result.get("feels_like_celsius")
    feels_f = round(feels_c * 9 / 5 + 32, 1) if feels_c is not None else None

    final_prompt = f"""The user asked: "{user_query}"

Here is the real-time temperature data retrieved from the API:
{json.dumps(tool_result, indent=2)}

Respond using EXACTLY this format, with no extra text before or after:

Weather Report
--------------
City           : {tool_result['city']}, {tool_result['country']}
Temperature    : {temp_c}°C / {temp_f}°F
Feels Like     : {f"{feels_c}°C / {feels_f}°F" if feels_c is not None else "N/A"}
Wind Speed     : {tool_result['wind_speed_kmh']} km/h
Conditions     : {tool_result['weather_condition']}
Data Source    : {tool_result['data_source']}

Final Response:
<A 2-3 sentence conversational summary mentioning the temperature in both Celsius and Fahrenheit, the feels-like temperature, current conditions, and a practical tip for the user.>

Do not include any JSON, markdown, bullet points, or additional sections beyond what is shown above."""

    final_messages = [
        {
            "role": "system",
            "content": "You are a weather reporting assistant. Output only the structured weather report followed by the Final Response section as specified. No extra text.",
        },.3
        
        {"role": "user", "content": final_prompt},
    ]

    final_response = ollama_client.chat(model=MODEL_NAME, messages=final_messages)
    final_text = final_response["message"]["content"]
    final_text = re.sub(r"<think>.*?</think>", "", final_text, flags=re.DOTALL).strip()

    return {
        "status": "success",
        "message": final_text,
        "intent": intent,
        "weather_data": {
            "city": tool_result["city"],
            "country": tool_result["country"],
            "temperature_celsius": temp_c,
            "temperature_fahrenheit": temp_f,
            "feels_like_celsius": feels_c,
            "feels_like_fahrenheit": feels_f,
            "wind_speed_kmh": tool_result["wind_speed_kmh"],
            "weather_condition": tool_result["weather_condition"],
            "latitude": tool_result["latitude"],
            "longitude": tool_result["longitude"],
            "data_source": tool_result["data_source"],
        },
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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "host": "https://ollama.com"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
