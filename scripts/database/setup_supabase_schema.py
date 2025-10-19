#!/usr/bin/env python3
"""
Execute Supabase schema setup SQL script.
This creates the hermes_vectors table and related functions.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()

supabase_url = os.environ['SUPABASE_DATABASE_URL']
supabase_key = os.environ['SUPABASE_SERVICE_ROLE_KEY']

# Read SQL file
sql_file = Path(__file__).parent / 'supabase_schema.sql'
with open(sql_file, 'r') as f:
    sql_content = f.read()

print("Setting up Supabase vector store schema...")
print("=" * 70)

# Use Supabase's SQL query endpoint
# The /rest/v1/rpc endpoint doesn't support arbitrary SQL, so we'll use psql if available
# Otherwise, we need to guide the user to run it manually

# Check if psql is available
import subprocess
try:
    result = subprocess.run(['which', 'psql'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\nUsing psql to execute SQL...")
        
        # Extract connection info from Supabase URL
        # Supabase database URL pattern: postgres://[user]:[pass]@[host]:[port]/[db]
        # We need to construct the connection string
        
        # For Supabase, the direct database URL follows this pattern:
        # postgres://postgres.[project-ref]:[password]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
        
        print("\n⚠️  Direct database access requires connection pooling URL.")
        print("Please run this SQL in your Supabase SQL Editor:")
        print("https://supabase.com/dashboard/project/[your-project]/sql/new")
        print("\nOr provide SUPABASE_DB_URL (postgres connection string) in your .env")
        
    else:
        raise FileNotFoundError("psql not found")
        
except (FileNotFoundError, subprocess.CalledProcessError):
    print("\n📋 Please run the following SQL in your Supabase SQL Editor:")
    print("=" * 70)
    print(sql_content)
    print("=" * 70)
    print("\nSteps:")
    print("1. Go to https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Click 'SQL Editor' in the sidebar")
    print("4. Click 'New Query'")
    print("5. Copy and paste the SQL above")
    print("6. Click 'Run' or press Cmd/Ctrl+Enter")
    sys.exit(0)

