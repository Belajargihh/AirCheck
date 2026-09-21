import sys
import os
import json

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
        # Endpoint debug cepat untuk melihat environment WSGI di Vercel
        path = environ.get('PATH_INFO', '')
        if '/_env_debug' in path:
            start_response('200 OK', [('Content-Type', 'application/json')])
            data = {k: str(v) for k, v in environ.items() if isinstance(v, (str, int, float, bool))}
            return [json.dumps(data, indent=2).encode('utf-8')]

        # Cek jika Vercel menyertakan path asli pada HTTP header
        matched_path = environ.get('HTTP_X_MATCHED_PATH') or environ.get('HTTP_X_NOW_ROUTE_MATCHES')
        if matched_path and not matched_path.startswith('/api/'):
            environ['PATH_INFO'] = matched_path
            return self.wsgi_app(environ, start_response)

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
