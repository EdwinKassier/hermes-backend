#!/usr/bin/env python3
"""Verify Supabase schema setup."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

# Change to script directory first
os.chdir(Path(__file__).parent.parent)
load_dotenv()

db_url = os.environ.get('SUPABASE_DB_URL')
if not db_url:
    print("ERROR: SUPABASE_DB_URL not found")
    sys.exit(1)

conn = psycopg2.connect(db_url)
cursor = conn.cursor()

print("Checking Supabase schema setup...")
print("=" * 70)

# Check if pgvector extension is enabled
cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
vector_enabled = cursor.fetchone()[0]
print(f"pgvector extension: {'✓ ENABLED' if vector_enabled else '❌ NOT ENABLED'}")

if not vector_enabled:
    print("\nPlease enable pgvector in Supabase:")
    print("1. Go to Database → Extensions")
    print("2. Search for 'vector' and enable it")
    cursor.close()
    conn.close()
    sys.exit(1)

# Check if table exists
cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = 'hermes_vectors'
    )
""")
table_exists = cursor.fetchone()[0]
print(f"hermes_vectors table: {'✓ EXISTS' if table_exists else '❌ DOES NOT EXIST'}")

if table_exists:
    # Check columns
    cursor.execute("""
        SELECT column_name, udt_name
        FROM information_schema.columns 
        WHERE table_name = 'hermes_vectors'
        ORDER BY ordinal_position
    """)
    columns = cursor.fetchall()
    print(f"\nColumns ({len(columns)}):")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}")
    
    # Check for records
    cursor.execute("SELECT COUNT(*) FROM hermes_vectors")
    count = cursor.fetchone()[0]
    print(f"\nCurrent record count: {count}")
    
    # Check indexes
    cursor.execute("""
        SELECT indexname 
        FROM pg_indexes 
        WHERE tablename = 'hermes_vectors'
    """)
    indexes = cursor.fetchall()
    print(f"\nIndexes ({len(indexes)}):")
    for idx in indexes:
        print(f"  - {idx[0]}")
    
    # Check function
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM pg_proc WHERE proname = 'match_documents'
        )
    """)
    func_exists = cursor.fetchone()[0]
    print(f"\nmatch_documents function: {'✓ EXISTS' if func_exists else '❌ DOES NOT EXIST'}")
    
    if not func_exists:
        print("\n⚠️  WARNING: match_documents function missing!")
        print("You'll need to create it manually in Supabase SQL Editor.")

cursor.close()
conn.close()

print("\n" + "=" * 70)
if table_exists and vector_enabled:
    print("✅ Schema verification complete - ready to generate embeddings!")
else:
    print("❌ Schema setup incomplete - please follow instructions above")

