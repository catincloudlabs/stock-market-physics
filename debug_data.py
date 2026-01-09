import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

print("🕵️ DIAGNOSTIC PROBE")
print("-------------------")

# 1. Check News Table
print("\n1. Checking 'news_vectors' table...")
try:
    news = supabase.table("news_vectors").select("*").limit(1).execute()
    if news.data:
        row = news.data[0]
        print(f"   ✅ Found Row!")
        print(f"   - ID Type: {type(row.get('id'))} (Value: {row.get('id')})")
        print(f"   - Published At: {row.get('published_at')}")
    else:
        print("   ❌ Table appears empty.")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Check Graph Table
print("\n2. Checking 'knowledge_graph' table...")
try:
    edge = supabase.table("knowledge_graph").select("*").limit(1).execute()
    if edge.data:
        row = edge.data[0]
        print(f"   ✅ Found Edge!")
        print(f"   - Source Node Type: {type(row.get('source_node'))} (Value: {row.get('source_node')})")
        print(f"   - Edge Type: {row.get('edge_type')}")
    else:
        print("   ❌ Graph appears empty.")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Test Join Logic
if news.data and edge.data:
    news_id = news.data[0]['id']
    print(f"\n3. Testing Query Match...")
    print(f"   Searching graph for source_node = {news_id} (Type: {type(news_id)})")
    
    # Test strict match
    q1 = supabase.table("knowledge_graph").select("*").eq("source_node", news_id).execute()
    print(f"   - Result using raw ID: {len(q1.data)} matches")
    
    # Test string match
    q2 = supabase.table("knowledge_graph").select("*").eq("source_node", str(news_id)).execute()
    print(f"   - Result using string ID: {len(q2.data)} matches")
