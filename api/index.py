import sys
import os
from urllib.parse import parse_qs

# Pastikan root direktori masuk ke sys.path agar modul app dan nlp_processor dapat diimpor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app


class VercelRouteMiddleware:
    """
    WSGI Middleware untuk memetakan rute dari query parameter 'route'
    yang diteruskan oleh vercel.json.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        query = environ.get('QUERY_STRING', '')
        if query:
            params = parse_qs(query)
            route = params.get('route', [''])[0]
            if route == 'analisis':
                environ['PATH_INFO'] = '/analisis'
            elif route == 'predict':
                environ['PATH_INFO'] = '/predict'
            elif route == 'tentang':
                environ['PATH_INFO'] = '/tentang'
            elif route == 'index':
                environ['PATH_INFO'] = '/'

            # Bersihkan parameter 'route' agar tidak mengganggu request Flask
            clean_query = '&'.join(p for p in query.split('&') if not p.startswith('route='))
            environ['QUERY_STRING'] = clean_query
        elif environ.get('PATH_INFO') in ('/api/index.py', '/api/index'):
            environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)


app = VercelRouteMiddleware(flask_app)
