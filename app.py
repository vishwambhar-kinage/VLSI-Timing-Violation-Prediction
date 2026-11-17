from flask import Flask, render_template, request, jsonify
from ml.pipeline import get_dashboard_data, predict_path, get_analysis_data, get_model_comparison

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/dashboard")
def dashboard():
    return jsonify(get_dashboard_data())

@app.get("/api/analysis")
def analysis():
    return jsonify(get_analysis_data())

@app.get("/api/models")
def models():
    return jsonify(get_model_comparison())

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_path(payload)
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the submitted values and try again."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port= int(__import__("os").environ.get("PORT", 5000)), debug=False)
