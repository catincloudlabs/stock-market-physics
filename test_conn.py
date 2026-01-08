import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Load secrets from the .env file
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")

if not url or not key:
    print("❌ Error: Missing API Keys. Check your .env file.")
    exit()

# 2. Initialize the Client
supabase: Client = create_client(url, key)

print(f"🔌 Connecting to: {url}...")

try:
    # 3. Write Test: Insert a dummy particle
    test_data = {
        "ticker": "TEST_PHYSICS", 
        "date": "2024-01-01", 
        "close": 420.69,
        "volume": 1000
    }
    
    # Execute the insert
    response = supabase.table("stocks_ohlc").insert(test_data).execute()
    
    # Check if data came back
    if response.data:
        print("✅ Write Successful! Particle inserted.")
        print(f"   Data: {response.data}")
    else:
        print("⚠️ Write returned no data (Check RLS policies if using anon key).")

    # 4. Read Test: Fetch it back
    read_response = supabase.table("stocks_ohlc").select("*").eq("ticker", "TEST_PHYSICS").execute()
    if len(read_response.data) > 0:
        print("✅ Read Successful! Particle detected.")

    # 5. Cleanup: Delete the dummy particle
    supabase.table("stocks_ohlc").delete().eq("ticker", "TEST_PHYSICS").execute()
    print("✅ Cleanup Successful! Particle removed.")
    
    print("\n🚀 SYSTEM READY. Your local machine is connected to the cloud DB.")

except Exception as e:
    print(f"\n❌ Connection Failed: {e}")
    print("Tip: Check if your IP is allowed or if the table 'stocks_ohlc' exists.")
