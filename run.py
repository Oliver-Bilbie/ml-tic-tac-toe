from distutils.log import debug
from tabnanny import verbose
from waitress import serve
from src.service.api import app

serve(app, host='0.0.0.0', port=8080, threads=1)
