#!/usr/bin/env python3
"""Clear all vectors from Supabase hermes_vectors table."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Change to project root
os.chdir(Path(__file__).parent.parent)
load_dotenv()

db_url = os.environ.get('SUPABASE_DB_URL')
if not db_url:
    print("ERROR: SUPABASE_DB_URL not found")
    sys.exit(1)

conn = psycopg2.connect(db_url)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()

print("Clearing existing embeddings from hermes_vectors...")
cursor.execute("DELETE FROM hermes_vectors")
cursor.execute("SELECT COUNT(*) FROM hermes_vectors")
count = cursor.fetchone()[0]
print(f"✓ Table cleared. Current count: {count}")

cursor.close()
conn.close()
print("\nReady to regenerate embeddings with gemini-embedding-001!")

