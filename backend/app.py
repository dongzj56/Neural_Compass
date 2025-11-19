from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_llm_response

app = Flask(__name__)
CORS(app) 

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    image_base64 = request.json.get('image_base64') 

    if not user_message and not image_base64:
        return jsonify({'error': 'No message or image provided'}), 400
    
    response_text = get_llm_response(user_message, image_base64)

    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(debug=True, port=5000)