import os
import sys

# Add project root directory to sys.path so backend imports work seamlessly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app import app

# Vercel Serverless WSGI entry point
# '@vercel/python' detects 'app' and delegates all incoming requests to Flask
if __name__ == "__main__":
    app.run(debug=True)

