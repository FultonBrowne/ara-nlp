from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import jsonformat
import tf_results
import os
import json
from flask import Flask, request, Response

tf_get = tf_results.__init__()
app = Flask(__name__)
@app.route("/v0/intent")
#def intent():
#    data = request.args.get("input")
#   print(data)
#   jsondata = jsonformat.getData(tf_get.getIntent(data))
#   return json.dumps(jsondata)
@app.route("/v0/pos")
def pos():
    data = request.args.get("input")
    lang = request.args.get("lang")
    return Response(json.dumps(jsonformat.getData(tf_get.getPos(data, lang))), mimetype="application/json")
@app.route("/v0/dpos")
def dpos():
    data = request.args.get("input")
    lang = request.args.get("lang")
    return Response(json.dumps(jsonformat.getData(tf_get.getDpos(data, lang))), mimetype="application/json")

if __name__ == "__main__":
    app.run()
