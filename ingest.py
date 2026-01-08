import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# 1. SETUP: Load Secrets & Initialize Engines
load_dotenv()

# The Source (Massive.com / Polygon)
MASSIVE_KEY = os.getenv("MASSIVE_API_KEY")
MASSIVE_BASE_URL = "https://api.polygon.io" # Confirmed URL based on endpoint structure

# The Brain (OpenAI for Embeddings)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The Storage (Supabase)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# --- EXPANDED UNIVERSE: S&P 100 + Key ETFs ---
# We use a set to avoid duplicates.
TICKER_UNIVERSE = [
    "AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "BRK.B", "LLY", "AVGO",
    "JPM", "XOM", "UNH", "V", "PG", "COST", "JNJ", "HD", "MA", "ABBV",
    "CVX", "MRK", "KO", "PEP", "ADBE", "WMT", "BAC", "ACN", "MCD", "LIN",
    "TMO", "ABT", "AMD", "NFLX", "DHR", "DIS", "INTC", "WFC", "CSCO", "INTU",
    "QCOM", "TXN", "PM", "CAT", "IBM", "AMGN", "GE", "UNP", "NOW", "SPGI",
    "SPY", "QQQ", "IWM" # Added major ETFs for market context
]

def get_embedding(text):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

# --- PHASE 1: STOCK DATA (THE PARTICLES) ---

def fetch_stock_history(ticker):
    """
    Fetches yesterday's OHLC data.
    """
    # Calculate 'Yesterday' (or last Friday if today is Monday)
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Endpoint: Daily Open/Close
    url = f"{MASSIVE_BASE_URL}/v1/open-close/{ticker}/{yesterday}?adjusted=true&apiKey={MASSIVE_KEY}"
    
    try:
        resp = requests.get(url)
        if resp.status_code == 429:
            print(f" ⏳ Rate limit hit. Sleeping for 60s...")
            time.sleep(60)
            return fetch_stock_history(ticker) # Retry
            
        if resp.status_code != 200:
            return None
            
        data = resp.json()
        return {
            "ticker": data.get("symbol"),
            "date": data.get("from"),
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "close": data.get("close"),
            "volume": data.get("volume")
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def upload_stock_batch(records):
    """
    Batch uploads to reduce Supabase API calls.
    """
    if not records: return
    try:
        # Supabase can handle bulk inserts
        data = supabase.table("stocks_ohlc").upsert(records).execute()
        print(f"   ✅ Batch saved {len(records)} tickers.")
    except Exception as e:
        print(f"   ❌ Error saving batch: {e}")

# --- PHASE 2: NEWS DATA (THE VECTORS) ---

def fetch_market_news(limit=20): # Increased limit to match larger universe
    print(f"📰 Fetching Top {limit} News Articles...")
    url = f"{MASSIVE_BASE_URL}/v2/reference/news?limit={limit}&apiKey={MASSIVE_KEY}"
    
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Error fetching news: {resp.status_code}")
        return []

    data = resp.json()
    results = data.get("results", [])
    processed_news = []
    
    for article in results:
        headline = article.get("title", "")
        description = article.get("description", "") or ""
        text_content = f"{headline}: {description}"
        
        # Only process if we have actual content
        if len(text_content) < 10: continue

        vector = get_embedding(text_content)
        
        processed_news.append({
            "ticker": "MARKET",
            "headline": headline,
            "published_at": article.get("published_utc"),
            "url": article.get("article_url"),
            "embedding": vector,
            "related_tickers": article.get("tickers", [])
        })
        
    return processed_news

def upload_news_and_build_graph(news_list):
    for item in news_list:
        payload = {k: v for k, v in item.items() if k != "related_tickers"}
        
        try:
            response = supabase.table("news_vectors").upsert(
                payload, on_conflict="url"
            ).execute()
            
            if not response.data: continue
            
            news_id = response.data[0]['id']
            related_tickers = item['related_tickers']
            
            edges = []
            for ticker in related_tickers:
                # We normalize the ticker to ensure graph consistency
                edges.append({
                    "source_node": str(news_id),
                    "target_node": ticker,
                    "edge_type": "MENTIONS",
                    "weight": 1.0
                })
            
            if edges:
                supabase.table("knowledge_graph").insert(edges).execute()
                print(f"   🔗 Connected News to {related_tickers}")
                
        except Exception as e:
            # Duplicate key errors are common if re-running, just skip
            if "duplicate key" not in str(e):
                print(f"   ❌ Error: {e}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"🚀 Starting Ingestion for {len(TICKER_UNIVERSE)} Tickers...")
    
    # 1. Update Particles (Stocks) - Batched
    stock_buffer = []
    BATCH_SIZE = 10
    
    for i, t in enumerate(TICKER_UNIVERSE):
        print(f"[{i+1}/{len(TICKER_UNIVERSE)}] Fetching {t}...")
        record = fetch_stock_history(t)
        
        if record:
            stock_buffer.append(record)
        
        # Sleep slightly to be kind to the API (5 calls per second max typically for free tiers)
        time.sleep(0.2) 
        
        # Upload when buffer is full
        if len(stock_buffer) >= BATCH_SIZE:
            upload_stock_batch(stock_buffer)
            stock_buffer = []

    # Upload remaining
    if stock_buffer:
        upload_stock_batch(stock_buffer)
        
    # 2. Update Fields & Bonds (News + Graph)
    latest_news = fetch_market_news(limit=20)
    upload_news_and_build_graph(latest_news)
    
    print("\n✨ UNIVERSE EXPANSION COMPLETE.")
