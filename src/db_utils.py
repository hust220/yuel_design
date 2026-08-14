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

def create_connection():
    db_params = load_db_config()
    attempts = 0
    while True:
        try:
            conn = psycopg2.connect(**db_params)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            if attempts < 3:
                time.sleep(2 ** attempts)
                attempts += 1
                continue
            else:
                raise

@contextmanager
def db_connection():
    """Database connection with retry mechanism"""
    db_params = load_db_config()
    attempts = 0
    conn = None
    try:
        while True:
            try:
                conn = psycopg2.connect(**db_params)
                conn.autocommit = False
                break
            except psycopg2.OperationalError as e:
                if attempts < 3:
                    time.sleep(2 ** attempts)
                    attempts += 1
                    continue
                else:
                    raise
        yield conn
    finally:
        if conn:
            conn.close()


def ensure_column_exists(table_name: str, column_name: str, column_definition: str):
    """
    Ensure a column exists on a table, adding it if necessary.

    Args:
        table_name: Name of the target table.
        column_name: Column to check/add.
        column_definition: SQL snippet defining the column type and constraints
                           (e.g., "TEXT", "BOOLEAN DEFAULT FALSE").
    """
    with db_connection() as conn:
        with conn.cursor() as c:
            c.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
                """,
                (table_name, column_name),
            )
            exists = c.fetchone() is not None
            if not exists:
                c.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                        sql.Identifier(table_name),
                        sql.Identifier(column_name),
                        sql.SQL(column_definition),
                    )
                )
                conn.commit()
            else:
                conn.rollback()
