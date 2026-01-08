import os
import requests
import pandas as pd
import networkx as nx
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# 1. SETUP: Load Secrets & Initialize Engines
load_dotenv()

# The Source (Massive.com)
MASSIVE_KEY = os.getenv("MASSIVE_API_KEY")
MASSIVE_BASE_URL = "https://api.massive.com"  # Note: Check if your endpoint is api.polygon.io or massive.com

# The Brain (OpenAI for Embeddings)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The Storage (Supabase)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def get_embedding(text):
    """
    Turns text into a 1536-dimensional physics vector.
    """
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

# --- PHASE 1: STOCK DATA (THE PARTICLES) ---

def fetch_stock_history(ticker):
    """
    Fetches yesterday's OHLC data (The state of the particle).
    """
    print(f"📉 Fetching Stocks for {ticker}...")
    
    # Calculate 'Yesterday'
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Endpoint: Daily Open/Close
    # Note: Massive/Polygon uses /v1/open-close/{ticker}/{date}
    url = f"{MASSIVE_BASE_URL}/v1/open-close/{ticker}/{yesterday}?adjusted=true&apiKey={MASSIVE_KEY}"
    
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"   ⚠️ No data for {ticker} (might be weekend/holiday)")
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

def upload_stock_data(record):
    if not record: return
    try:
        supabase.table("stocks_ohlc").upsert(record).execute()
        print(f"   ✅ Saved {record['ticker']} price.")
    except Exception as e:
        print(f"   ❌ Error saving stock: {e}")

# --- PHASE 2: NEWS DATA (THE VECTORS) ---

def fetch_market_news(limit=10):
    """
    Fetches the latest market news.
    """
    print(f"📰 Fetching Top {limit} News Articles...")
    
    # Endpoint: Reference News
    url = f"{MASSIVE_BASE_URL}/v2/reference/news?limit={limit}&apiKey={MASSIVE_KEY}"
    
    resp = requests.get(url)
    data = resp.json()
    
    results = data.get("results", [])
    processed_news = []
    
    for article in results:
        # 1. Clean the data
        headline = article.get("title", "")
        description = article.get("description", "") or ""
        text_content = f"{headline}: {description}"
        
        # 2. Vectorize (The Physics)
        # We embed the combined title + description for better semantic search
        vector = get_embedding(text_content)
        
        processed_news.append({
            "ticker": "MARKET", # Generic tag, or use primary ticker if available
            "headline": headline,
            "published_at": article.get("published_utc"),
            "url": article.get("article_url"),
            "embedding": vector,
            "related_tickers": article.get("tickers", []) # Crucial for Graph!
        })
        
    return processed_news

def upload_news_and_build_graph(news_list):
    """
    Uploads vectors AND builds the connective tissue (Graph) in one pass.
    """
    for item in news_list:
        # A. Upload Vector (The Node)
        payload = {k: v for k, v in item.items() if k != "related_tickers"}
        
        try:
            # We insert and RETURN the new ID so we can attach edges to it
            response = supabase.table("news_vectors").upsert(
                payload, on_conflict="url" # Avoid duplicates
            ).execute()
            
            if not response.data: continue
            
            news_id = response.data[0]['id']
            related_tickers = item['related_tickers']
            
            # B. Build Graph Edges (The Bonds)
            # Logic: If news mentions [AAPL, MSFT], create edges:
            # (News) -> AAPL
            # (News) -> MSFT
            
            edges = []
            for ticker in related_tickers:
                edges.append({
                    "source_node": str(news_id),
                    "target_node": ticker,
                    "edge_type": "MENTIONS",
                    "weight": 1.0
                })
            
            if edges:
                supabase.table("knowledge_graph").insert(edges).execute()
                print(f"   🔗 Connected News '{item['headline'][:20]}...' to {related_tickers}")
                
        except Exception as e:
            print(f"   ❌ Error uploading news: {e}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Update Particles (Stocks)
    my_portfolio = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"]
    for t in my_portfolio:
        stock_record = fetch_stock_history(t)
        upload_stock_data(stock_record)
        
    # 2. Update Fields & Bonds (News + Graph)
    latest_news = fetch_market_news(limit=5)
    upload_news_and_build_graph(latest_news)
    
    print("\n✨ INGESTION COMPLETE. The physics engine is updated.")
