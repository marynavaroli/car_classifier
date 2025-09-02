from flask import Flask, request, jsonify
from PIL import Image
from BaggingRegressor import get_prediction
from CNN import get_cnn_prediction
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    try:
        if request.method == "OPTIONS":
            # Preflight CORS request
            return jsonify({"status": "ok"}), 200

        data = request.get_json(silent=True)  # Parse JSON body safely

        if not data:
            return jsonify({"error": "Empty or invalid JSON body"}), 400

        speed = data.get("speed")
        torque = data.get("torque")
        charging = data.get("charging")
        height = data.get("height")
        body = data.get("body")

        # Validate required fields
        if None in [speed, torque, charging, height, body]:
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Convert to numeric if needed
        try:
            speed = float(speed)
            torque = float(torque)
            height = float(height)
        except ValueError:
            return jsonify({"error": "Numeric fields must be valid numbers"}), 400

        prediction = float(get_prediction(speed, torque, charging, height, body)[0])

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/CNN", methods=["POST", "OPTIONS"])
def cnn():
    try:
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' in form-data"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Load image
        img = Image.open(file.stream).convert("RGB")

        # Get prediction
        class_str = get_cnn_prediction(img)

        return jsonify({
            "filename": file.filename,
            "class": class_str
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)