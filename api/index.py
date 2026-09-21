import sys
import os

# Pastikan root direktori masuk ke sys.path agar modul app dan nlp_processor dapat diimpor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app


class VercelPathFixer:
    """
    WSGI Middleware untuk menormalkan PATH_INFO jika Vercel
    meneruskan prefix /api/index atau /api/index.py dari rewrite rules.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        for prefix in ['/api/index.py', '/api/index']:
            if path == prefix:
                environ['PATH_INFO'] = '/'
                break
            elif path.startswith(prefix + '/'):
                environ['PATH_INFO'] = path[len(prefix):]
                break
        return self.wsgi_app(environ, start_response)


# Ekspor app sebagai WSGI entrypoint Vercel
app = VercelPathFixer(flask_app)
