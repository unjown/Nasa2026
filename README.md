# NASA Space Apps — Environmental Data Explorer

Explore NASA environmental observations across selected Philippine regions with trend analysis, seasonal patterns, anomalies, variable correlation, AI explanations, and experimental forecasts.

## Run locally

1. Install Python dependencies:
   ```cmd
   python -m pip install -r requirements.txt
   ```
2. Install [Ollama](https://ollama.com/) and pull the interpretation model:
   ```cmd
   ollama pull qwen3:1.7b
   ```
3. Start the dashboard from the project root:
   ```cmd
   python -m uvicorn main:app --reload --port 8000
   ```
4. Open **http://127.0.0.1:8000/dashboard/**.

NASA CSV files must be in `data/raw/`. The first forecast may download TimesFM weights and take longer to run. An internet connection is needed for initial model downloads and dashboard CDN scripts.

**Note:** Forecasts and AI explanations are experimental; they are not disaster warnings or confirmed causal findings.
