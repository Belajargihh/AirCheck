import sys
import os
from urllib.parse import parse_qs

# Pastikan root direktori masuk ke sys.path agar modul app dan nlp_processor dapat diimpor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app


class VercelPathFixer:
    """
    WSGI Middleware untuk merekonstruksi PATH_INFO yang tepat
    dari parameter rewrite Vercel (__path=$1).
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query = environ.get('QUERY_STRING', '')
        if '__path=' in query:
            params = parse_qs(query)
            path_val = params.get('__path', [''])[0]
            if path_val:
                environ['PATH_INFO'] = '/' + path_val.lstrip('/')
            else:
                environ['PATH_INFO'] = '/'
            
            # Bersihkan __path dari QUERY_STRING agar tidak mengganggu request.args pengguna
            query_parts = [p for p in query.split('&') if not p.startswith('__path=')]
            environ['QUERY_STRING'] = '&'.join(query_parts)
        elif environ.get('PATH_INFO') in ('/api/index.py', '/api/index'):
            environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)


# Ekspor app sebagai WSGI entrypoint Vercel
app = VercelPathFixer(flask_app)
