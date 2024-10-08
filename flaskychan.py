from generatyresponsychan import generate_response
from flask import Flask, jsonify, make_response, request
import pandas as pd
from flask_cors import CORS
import json
import threading

app = Flask(__name__)


@app.route("/")
def hello():
    return "I am alive!"

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    texto = request.form.get('texto')
    if not request.form.get('texto') or not texto:
        texto = "HOLIS"
    respuestas = generate_response(texto)
    return jsonify(respuestas)

threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':3000}).start()