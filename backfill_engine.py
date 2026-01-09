import os
import requests
import time
import concurrent.futures
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# 1. SETUP
load_dotenv()
MASSIVE_KEY = os.getenv("MASSIVE_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io" 

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# --- CONFIGURATION ---
BACKFILL_DAYS = 30
MAX_WORKERS = 10 
DB_BATCH_SIZE = 200

# The Curated Universe (242 Tickers)
TICKER_UNIVERSE = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "LLY", "AVGO", "JPM",
    "AMD", "QCOM", "TXN", "INTC", "AMAT", "MU", "LRCX", "ADI", "KLAC", "MRVL", "SNPS", 
    "CDNS", "PANW", "CRWD", "PLTR", "SMCI", "ARM", "TSM", "ASML", "ON", "MCHP", "STM",
    "ORCL", "ADBE", "CRM", "INTU", "IBM", "NOW", "UBER", "SAP", "FI", "ADP", "ACN", 
    "CSCO", "SQ", "SHOP", "WDAY", "SNOW", "TEAM", "ADSK", "DDOG", "ZM", "NET", "TTD",
    "MDB", "ZS", "GIB", "FICO", "ANET", "ESTC",
    "V", "MA", "BAC", "WFC", "MS", "GS", "C", "BLK", "SPGI", "AXP", "MCO", "PGR", "CB", 
    "MMC", "AON", "USB", "PNC", "TFC", "COF", "DFS", "PYPL", "AFRM", "HOOD", "COIN",
    "KKR", "BX", "APO", "TRV", "ALL", "HIG", "MET",
    "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN", "ISRG", "ELV", 
    "VRTX", "REGN", "ZTS", "BSX", "BDX", "GILD", "HCA", "MCK", "CI", "HUM", "CVS", 
    "BMY", "SYK", "EW", "MDT", "DXCM", "ILMN", "ALGN", "BIIB", "MRNA", "BNTX",
    "WMT", "PG", "COST", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "SBUX", "LOW", "PM", 
    "TGT", "TJX", "EL", "CL", "MO", "LULU", "CMG", "MAR", "BKNG", "ABNB", "HLT", "YUM",
    "DE", "CAT", "HON", "GE", "MMM", "ETN", "ITW", "EMR", "PH", "CMI", "PCAR", "TT",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES", "KMI", "WMB",
    "LMT", "RTX", "BA", "GD", "NOC", "LHX", "TDG", "GE", "WM", "RSG", "UNP", "CSX", "NSC",
    "DAL", "UAL", "AAL", "LUV", "FDX", "UPS",
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
    "XLK", "XLV", "XLF", "XLE", "XLC", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    "SMH", "SOXX", "XBI", "KRE", "KBE", "JETS", "ITB",
    "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND",
    "GLD", "SLV", "USO", "UNG", "DBC",
    "VIXY", "UVXY", "VXX", "TQQQ", "SQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "LABU", "LABD", "TMF", "TMV"
]

# --- ROBUST UTILS ---

# FIX 1: Retry Logic for OpenAI Network Errors
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

def upload_stock_batch(records):
    if not records: return
    try:
        # Ignore duplicates for Stocks is safe because we don't need the returned ID
        supabase.table("stocks_ohlc").upsert(
            records, 
            on_conflict="ticker,date", 
            ignore_duplicates=True
        ).execute()
        print(f" 💾 Saved {len(records)} rows to DB.")
    except Exception as e:
        print(f" ❌ DB Error (Batch Skipped): {str(e)[:100]}...")

# --- PART 1: OPTIMIZED STOCK HISTORY (AGGREGATES) ---

# FIX 2: Use Decorator for Retry instead of dangerous recursion
@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10))
def fetch_ticker_history(ticker):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=BACKFILL_DAYS)).strftime('%Y-%m-%d')
    
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={MASSIVE_KEY}"
    
    resp = requests.get(url, timeout=15)
    
    # Raise exception for 429 so @retry catches it
    if resp.status_code == 429:
        raise Exception("Rate Limited")
        
    if resp.status_code != 200:
        print(f" ❌ Error fetching {ticker}: {resp.status_code}")
        return []
        
    data = resp.json()
    if "results" not in data: return []
        
    records = []
    for bar in data["results"]:
        date_str = datetime.fromtimestamp(bar["t"] / 1000).strftime('%Y-%m-%d')
        records.append({
            "ticker": ticker,
            "date": date_str,
            "open": bar.get("o"),
            "high": bar.get("h"),
            "low": bar.get("l"),
            "close": bar.get("c"),
            "volume": bar.get("v")
        })
    return records

# --- PART 2: NEWS ARCHIVE ---

def backfill_news():
    print(f"\n📰 Starting News Backfill (Last {BACKFILL_DAYS} Days)...")
    
    start_date = (datetime.now() - timedelta(days=BACKFILL_DAYS)).strftime('%Y-%m-%d')
    url = f"{POLYGON_BASE_URL}/v2/reference/news?published_utc.gte={start_date}&limit=1000&sort=published_utc&order=desc&apiKey={MASSIVE_KEY}"
    
    total_processed = 0
    next_url = url
    
    while next_url:
        try:
            resp = requests.get(next_url, timeout=20)
            if resp.status_code != 200:
                print(f"❌ News Error {resp.status_code}")
                break
                
            data = resp.json()
            articles = data.get("results", [])
            
            if not articles: break
            
            print(f"  ... fetched page of {len(articles)} articles. Processing...")
            
            sub_batch_size = 10
            for i in range(0, len(articles), sub_batch_size):
                chunk = articles[i:i + sub_batch_size]
                batch_vectors = []
                batch_edges = []
                
                for article in chunk:
                    headline = article.get("title", "")
                    description = article.get("description", "") or ""
                    text_content = f"{headline}: {description}"
                    
                    if len(text_content) < 20: continue
                    
                    try:
                        vector = get_embedding(text_content) 
                        
                        news_item = {
                            "ticker": "MARKET",
                            "headline": headline,
                            "published_at": article.get("published_utc"),
                            "url": article.get("article_url"),
                            "embedding": vector,
                            "related_tickers": article.get("tickers", []) 
                        }
                        
                        # FIX 3: REMOVED ignore_duplicates=True
                        # We must allow the update to happen so Supabase returns the 'id'.
                        # If we ignore duplicates, 'res.data' is empty and the script crashes.
                        res = supabase.table("news_vectors").upsert(
                            {k:v for k,v in news_item.items() if k!="related_tickers"}, 
                            on_conflict="url"
                        ).execute()
                        
                        if res.data:
                            news_id = res.data[0]['id']
                            for t in news_item["related_tickers"]:
                                batch_edges.append({
                                    "source_node": str(news_id),
                                    "target_node": t,
                                    "edge_type": "MENTIONS",
                                    "weight": 1.0
                                })
                                
                    except Exception as e:
                        print(f"Skipping article: {e}")

                if batch_edges:
                    try:
                        # Graph edges CAN ignore duplicates because we don't need a return value
                        supabase.table("knowledge_graph").upsert(
                            batch_edges, on_conflict="source_node,target_node,edge_type", ignore_duplicates=True
                        ).execute()
                    except Exception as e:
                        pass
                
                print(f"    - Saved chunk {i}-{i+len(chunk)}")
                time.sleep(0.1)

            total_processed += len(articles)
            print(f"  ✅ Page complete. Total so far: {total_processed}")
            
            next_url = data.get("next_url")
            if next_url: next_url += f"&apiKey={MASSIVE_KEY}"
            else: break
            
        except Exception as e:
            print(f" ❌ Critical Error in News Loop: {e}")
            break

# --- EXECUTION ---

if __name__ == "__main__":
    start_time = time.time()
    
    # 1. STOCKS
    print(f"🚀 Backfilling STOCKS for {len(TICKER_UNIVERSE)} tickers...")
    all_stock_records = []
    
    # Using 'executor.map' or manual submission is fine. Manual allows us to count progress easier.
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(fetch_ticker_history, t): t for t in TICKER_UNIVERSE}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                data = future.result()
                if data: all_stock_records.extend(data)
            except Exception as e:
                print(f"Worker Error: {e}")
                
            completed += 1
            if completed % 50 == 0: print(f"  ... {completed}/{len(TICKER_UNIVERSE)} tickers fetched.")

    # Upload Stocks in large chunks
    print(f"📦 Uploading {len(all_stock_records)} historical stock rows...")
    for i in range(0, len(all_stock_records), DB_BATCH_SIZE):
        batch = all_stock_records[i:i + DB_BATCH_SIZE]
        upload_stock_batch(batch)
    
    # 2. NEWS
    backfill_news()
    
    print(f"\n✨ BACKFILL COMPLETE in {time.time() - start_time:.2f}s")
