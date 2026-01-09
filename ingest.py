import os
import requests
import time
import concurrent.futures
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# 1. SETUP: Load Secrets & Initialize Engines
load_dotenv()

# The Source (Massive.com / Polygon)
MASSIVE_KEY = os.getenv("MASSIVE_API_KEY")
MASSIVE_BASE_URL = "https://api.polygon.io" 

# The Brain (OpenAI)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The Storage (Supabase)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# --- CONFIGURATION ---
MAX_WORKERS = 10  # Number of simultaneous connections
DB_BATCH_SIZE = 50 # Write to DB in larger chunks

# --- MARKET UNIVERSE (S&P 500 Core + Nasdaq 100 + High Volatility) ---
TICKER_UNIVERSE = [
    # 1. THE MAGNIFICENT 7 & MEGA CAP (The Sun)
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "LLY", "AVGO", "JPM",
    
    # 2. SEMICONDUCTORS & AI (High Beta / Velocity)
    "AMD", "QCOM", "TXN", "INTC", "AMAT", "MU", "LRCX", "ADI", "KLAC", "MRVL", "SNPS", 
    "CDNS", "PANW", "CRWD", "PLTR", "SMCI", "ARM", "TSM", "ASML", "ON", "MCHP", "STM",
    
    # 3. SOFTWARE & CLOUD (Growth Engines)
    "ORCL", "ADBE", "CRM", "INTU", "IBM", "NOW", "UBER", "SAP", "FI", "ADP", "ACN", 
    "CSCO", "SQ", "SHOP", "WDAY", "SNOW", "TEAM", "ADSK", "DDOG", "ZM", "NET", "TTD",
    "MDB", "ZS", "GIB", "FICO", "ANET", "ESTC",
    
    # 4. FINANCE & PAYMENTS (Economic Flow)
    "V", "MA", "BAC", "WFC", "MS", "GS", "C", "BLK", "SPGI", "AXP", "MCO", "PGR", "CB", 
    "MMC", "AON", "USB", "PNC", "TFC", "COF", "DFS", "PYPL", "AFRM", "HOOD", "COIN",
    "KKR", "BX", "APO", "TRV", "ALL", "HIG", "MET",
    
    # 5. HEALTHCARE & BIO (Defensive + Speculative)
    "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN", "ISRG", "ELV", 
    "VRTX", "REGN", "ZTS", "BSX", "BDX", "GILD", "HCA", "MCK", "CI", "HUM", "CVS", 
    "BMY", "SYK", "EW", "MDT", "DXCM", "ILMN", "ALGN", "BIIB", "MRNA", "BNTX",
    
    # 6. CONSUMER & RETAIL (The Economy)
    "WMT", "PG", "COST", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "LOW", "PM", 
    "TGT", "TJX", "EL", "CL", "MO", "LULU", "CMG", "MAR", "BKNG", "ABNB", "HLT", "YUM",
    "DE", "CAT", "HON", "GE", "MMM", "ETN", "ITW", "EMR", "PH", "CMI", "PCAR", "TT",
    
    # 7. ENERGY & INDUSTRIALS (Cyclical Mass)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES", "KMI", "WMB",
    "LMT", "RTX", "BA", "GD", "NOC", "LHX", "TDG", "GE", "WM", "RSG", "UNP", "CSX", "NSC",
    "DAL", "UAL", "AAL", "LUV", "FDX", "UPS",
    
    # 8. MACRO ETFS (The "Fields")
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", # Broad Markets
    "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", # Sectors
    "SMH", "SOXX", "XBI", "KRE", "KBE", "JETS", "ITB", # Sub-sectors
    "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND", # Bonds / Rates
    "GLD", "SLV", "USO", "UNG", "DBC", # Commodities
    
    # 9. VOLATILITY & LEVERAGE (Chaos Metrics)
    "VIXY", "UVXY", "VXX", # Volatility
    "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "LABU", "LABD", "TMF", "TMV"
]

def get_embedding(text):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

# --- PHASE 1: STOCK DATA (PARALLEL FETCH) ---

def fetch_single_stock(ticker):
    """
    Fetches a single stock. Designed to be run by a worker thread.
    Includes auto-retry for 429s.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"{MASSIVE_BASE_URL}/v1/open-close/{ticker}/{yesterday}?adjusted=true&apiKey={MASSIVE_KEY}"
    
    attempts = 0
    max_retries = 3
    
    while attempts < max_retries:
        try:
            resp = requests.get(url, timeout=10)
            
            # Handle Rate Limits (Backoff)
            if resp.status_code == 429:
                wait_time = 1.5 * (2 ** attempts) # Exponential backoff: 1.5s, 3s, 6s...
                print(f" ⏳ 429 Limit on {ticker}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                attempts += 1
                continue
            
            # Handle Missing Data (Weekend/Holiday/Delisted)
            if resp.status_code == 404:
                return None
                
            if resp.status_code != 200:
                print(f" ❌ Error {resp.status_code} for {ticker}")
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
            print(f" ❌ Exception fetching {ticker}: {e}")
            return None
            
    return None

def upload_stock_batch(records):
    if not records: return
    try:
        supabase.table("stocks_ohlc").upsert(records).execute()
        print(f" 💾 DB Commit: Saved {len(records)} tickers.")
    except Exception as e:
        print(f" ❌ DB Error: {e}")

# --- PHASE 2: NEWS DATA ---

def fetch_market_news(limit=50):
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
        
        if len(text_content) < 15: continue # Skip empty/junk news

        # Note: OpenAI calls are fast, but sequential here. 
        # Could also be parallelized if volume is huge.
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
            res = supabase.table("news_vectors").upsert(payload, on_conflict="url").execute()
            if not res.data: continue
            
            news_id = res.data[0]['id']
            edges = [{"source_node": str(news_id), "target_node": t, "edge_type": "MENTIONS", "weight": 1.0} for t in item['related_tickers']]
            
            if edges:
                supabase.table("knowledge_graph").insert(edges).execute()
        except Exception as e:
            if "duplicate key" not in str(e): print(f"Error: {e}")
    print(f" ✅ Processed {len(news_list)} articles.")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    start_time = time.time()
    print(f"🚀 Starting High-Speed Ingestion for {len(TICKER_UNIVERSE)} Tickers...")
    
    valid_records = []
    
    # PARALLEL EXECUTION
    # We use a ThreadPool to fire off requests simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_ticker = {executor.submit(fetch_single_stock, t): t for t in TICKER_UNIVERSE}
        
        # Process as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            if result:
                valid_records.append(result)
            
            completed_count += 1
            if completed_count % 50 == 0:
                print(f" ... fetched {completed_count}/{len(TICKER_UNIVERSE)}")

    # Batch Upload to DB (Minimizing DB Connections)
    # We slice the list into chunks of DB_BATCH_SIZE
    for i in range(0, len(valid_records), DB_BATCH_SIZE):
        batch = valid_records[i:i + DB_BATCH_SIZE]
        upload_stock_batch(batch)

    # 2. Update News
    latest_news = fetch_market_news(limit=50)
    upload_news_and_build_graph(latest_news)
    
    duration = time.time() - start_time
    print(f"\n✨ COMPLETE in {duration:.2f} seconds.")
