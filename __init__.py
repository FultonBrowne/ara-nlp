from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import jsonformat
import tf_results
import os
import json
from flask import Flask, request

tf_get = tf_results.__init__()
app = Flask(__name__)
@app.route("/v0/intent")
def intent():
    data = request.args.get("input")
    print(data)
    jsondata = jsonformat.getData(tf_get.getIntent(data))
    return json.dumps(jsondata)
@app.route("/v0/pos")
def pos():
    data = request.args.get("input")
    return tf_get.getChuncks(data, "en")

if __name__ == "__main__":
    app.run()
