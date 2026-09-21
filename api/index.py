import sys
import os

# Pastikan root direktori masuk ke sys.path agar modul app dan nlp_processor dapat diimpor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
