from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        text = data["text"]

        # Dummy classifier logic (replace with your model)
        if "cat" in text.lower():
            label = "animal"
        elif "car" in text.lower():
            label = "vehicle"
        else:
            label = "unknown"

        return jsonify({
            "input": text,
            "label": label
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)