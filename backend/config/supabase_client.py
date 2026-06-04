from supabase import create_client, Client

SUPABASE_URL = "https://hpangwejkwlgftzimcra.supabase.co"

SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhwYW5nd2Vqa3dsZ2Z0emltY3JhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5ODQ2MDUsImV4cCI6MjA5NDU2MDYwNX0.24RYub3SxpbQmTehVyfYLQ6Yn-i5ddwJ0lf9AkscdMs"

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)