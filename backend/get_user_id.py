"""
Quick helper to find the user_id from Supabase profiles table.
"""
import sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.supabase_client import supabase

resp = supabase.table("profiles").select("id, full_name, email").execute()
if resp.data:
    print("Users in profiles table:")
    for row in resp.data:
        print(f"  ID: {row['id']}  Name: {row.get('full_name', '?')}  Email: {row.get('email', '?')}")
else:
    print("No users found in profiles table.")
