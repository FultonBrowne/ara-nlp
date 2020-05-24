from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import tf_results
import os


hostName = "localhost"
serverPort = 5555
tf_get = tf_results.__init__()
tf_get.getAll("text bob")
def route(path):
        if path.startswith("/v0/"):
            if(path.startswith("/v0/pos")):
                print("pos")
class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        print(self.path)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        route(self.path)
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

if os.environ.get('train') == "pls train":
    import train
    train.main()
elif __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
