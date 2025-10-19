#!/usr/bin/env python3
"""
Execute Supabase schema setup SQL using direct PostgreSQL connection.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment
load_dotenv()

db_url = os.environ.get('SUPABASE_DB_URL')
if not db_url:
    print("ERROR: SUPABASE_DB_URL not found in .env file")
    sys.exit(1)

# Read SQL file
sql_file = Path(__file__).parent / 'supabase_schema.sql'
with open(sql_file, 'r') as f:
    sql_content = f.read()

print("Connecting to Supabase PostgreSQL database...")
print("=" * 70)

try:
    # Connect to the database
    conn = psycopg2.connect(db_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    print("✓ Connected successfully!")
    print("\nExecuting schema setup SQL...")
    print("=" * 70)
    
    # Execute the entire SQL as one block to preserve function definitions
    try:
        cursor.execute(sql_content)
        print("✓ SQL executed successfully!")
    except psycopg2.Error as e:
        error_msg = str(e)
        if 'type "vector" does not exist' in error_msg:
            print("\n" + "=" * 70)
            print("❌ ERROR: pgvector extension is not enabled.")
            print("\nPlease enable it in Supabase:")
            print("1. Go to https://supabase.com/dashboard")
            print("2. Select your project")
            print("3. Go to 'Database' → 'Extensions'")
            print("4. Search for 'vector' and enable it")
            print("5. Re-run this script")
            print("=" * 70)
        else:
            print(f"❌ SQL execution failed: {error_msg}")
        raise
    
    print("\n✅ Schema setup completed successfully!")
    print("=" * 70)
    
    # Verify the table was created
    cursor.execute("""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name = 'hermes_vectors'
    """)
    result = cursor.fetchone()
    
    if result[0] > 0:
        print("\n✓ hermes_vectors table created successfully")
        
        # Check if pgvector extension is enabled
        cursor.execute("""
            SELECT COUNT(*) 
            FROM pg_extension 
            WHERE extname = 'vector'
        """)
        result = cursor.fetchone()
        if result[0] > 0:
            print("✓ pgvector extension enabled")
        
        # Check indexes
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename = 'hermes_vectors'
        """)
        indexes = cursor.fetchall()
        print(f"✓ {len(indexes)} index(es) created")
        for idx in indexes:
            print(f"  - {idx[0]}")
        
        # Check function
        cursor.execute("""
            SELECT COUNT(*) 
            FROM pg_proc 
            WHERE proname = 'match_documents'
        """)
        result = cursor.fetchone()
        if result[0] > 0:
            print("✓ match_documents function created")
    
    cursor.close()
    conn.close()
    
    print("\n🎉 Supabase vector store is ready!")
    print("\nYou can now run: python3 scripts/generate_embeddings_supabase.py")
    
except psycopg2.Error as e:
    print(f"\n❌ Database error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)

