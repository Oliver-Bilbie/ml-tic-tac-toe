#!/usr/bin/python
from waitress import serve
from src import app

serve(app, host='0.0.0.0', port=8080, threads=1)
