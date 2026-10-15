"""
=============================================================================
NextWord AI - Local Web Server Launcher
=============================================================================
Runs a lightweight local HTTP server and automatically opens NextWord AI
in your default web browser.

Usage:
  python server.py
=============================================================================
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")

def main():
    os.chdir(DIRECTORY)
    print("=" * 65)
    print("  🚀 NEXTWORD AI - LOCAL WEB SERVER")
    print("=" * 65)
    print(f"  Serving files from: {DIRECTORY}")
    print(f"  URL: http://localhost:{PORT}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 65)

    # Automatically launch default browser
    url = f"http://localhost:{PORT}"
    webbrowser.open(url)

    # Allow immediate address reuse
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped. Goodbye!")
    except OSError as e:
        if e.errno == 98 or e.winerror == 10048:
            print(f"\nPort {PORT} is already in use. Opening browser at {url} directly...")
            webbrowser.open(url)
        else:
            raise

if __name__ == "__main__":
    main()
