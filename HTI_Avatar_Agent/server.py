import http.server
import socketserver
import json
import urllib.request
import os
from pathlib import Path

PORT = 5173
# Automatically serve the folder containing this server.py file
DIRECTORY = Path(__file__).resolve().parent

class AvatarServerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def do_POST(self):
        if self.path == '/api/save-avatar':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            url = post_data.get('url')
            if url:
                print(f"[Server] Downloading production avatar from: {url}")
                target = DIRECTORY / "final" / "meera.glb"
                target.parent.mkdir(parents=True, exist_ok=True)
                
                if "morphTargets=" not in url:
                    url += "?morphTargets=ARKit,Oculus+Visemes"
                
                urllib.request.urlretrieve(url, str(target))
                print(f"[Server] Successfully saved avatar to {target}")

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "success", "saved": true}')
                return

        elif self.path == '/api/upload-avatar':
            content_length = int(self.headers.get('Content-Length', 0))
            data = self.rfile.read(content_length)
            target = DIRECTORY / "final" / "meera.glb"
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'wb') as f:
                f.write(data)
            print(f"[Server] Uploaded and saved new avatar to {target} ({len(data)} bytes)")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "success", "saved": true}')
            return
        
        self.send_response(404)
        self.end_headers()

if __name__ == '__main__':
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), AvatarServerHandler) as httpd:
        print("=" * 65)
        print("  Meera — HTI Lab 3D Conversational Avatar Server")
        print(f"  URL: http://localhost:{PORT}/web/index.html")
        print("=" * 65)
        print("(Press Ctrl+C to stop the server)")
        httpd.serve_forever()
