from flask import Flask, request, jsonify
from PIL import Image
import io
from KNN import classify_image
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


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)