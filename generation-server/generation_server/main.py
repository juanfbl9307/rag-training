from flask import Flask, request, jsonify
from service import generate_response
app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    response = generate_response(data["message"])

    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)