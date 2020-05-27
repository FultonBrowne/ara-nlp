from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import tf_results
import os
import json
from flask import Flask, request

hostName = "localhost"
serverPort = 5555
tf_get = tf_results.__init__()
tf_get.getAll("text fulton")
app = Flask(__name__)     
@app.route("/v0/intent")
def hello():
    data = request.args.get("input")
    print(data)
    return tf_get.getAll(data)

if __name__ == "__main__":
    app.run()
