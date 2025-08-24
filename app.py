from flask import Flask, request, jsonify
from PIL import Image
from KNN import classify_image
from BaggingRegressor import get_prediction
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    try:
        if request.method == "OPTIONS":
            # Preflight request
            return jsonify({"status": "ok"}), 200

        print("FILES:", request.files)
        print("FORM:", request.form)
        if "image" not in request.files:
            return jsonify({"error": "Missing 'image' in form-data"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Load image with Pillow
        img = Image.open(file.stream)

        # Dummy classification logic (replace with ML model)
        class_str = classify_image(img)

        return jsonify({
            "filename": file.filename,
            "class": class_str
        })

    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({"error": str(e)}), 500

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


if __name__ == "__main__":
    app.run(debug=True)