import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide a 'text' field"}), 400

    text = data["text"].lower().strip()
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    return jsonify({
        "prediction": "True News" if prediction == 1 else "Fake News",
        "confidence": round(float(max(probability)) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)