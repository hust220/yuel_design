import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
import time
import os
from pathlib import Path

def load_db_config():
    """Load database configuration from config file"""
    config_path = Path(__file__).parent / 'db_config'
    if not config_path.exists():
        raise FileNotFoundError(f"Database configuration file not found at {config_path}")
    
    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    
    return {
        'dbname': config['DB_NAME'],
        'user': config['DB_USER'],
        'host': config['DB_HOST'],
        'port': config['DB_PORT']
    }

DB_PARAMS = load_db_config()

@contextmanager
def db_connection():
    """Database connection with retry mechanism"""
    attempts = 0
    conn = None
    try:
        while True:
            try:
                conn = psycopg2.connect(**DB_PARAMS)
                conn.autocommit = False
                break
            except psycopg2.OperationalError as e:
                if attempts < 10:
                    time.sleep(2 ** attempts)
                    attempts += 1
                    continue
                else:
                    raise
        yield conn
    finally:
        if conn:
            conn.close()
