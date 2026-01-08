import os
import json
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx

# 1. SETUP
load_dotenv()

# FIX: Explicitly increase timeouts for the HTTP client
timeout_config = httpx.Timeout(60.0, connect=10.0, read=60.0)
opts = ClientOptions(postgrest_client_timeout=60)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_KEY"),
    options=opts
)

# CONFIG
HISTORY_DAYS = 30
PERPLEXITY = 30  

# --- RETRY LOGIC FOR DB ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_daily_vectors_rpc(target_date):
    """
    Fetches daily averages using Server-Side Paging.
    Page size reduced to 25 to ensure rapid response times.
    """
    all_records = []
    page = 0
    page_size = 25 # Smaller chunks to prevent socket hangs
    
    while True:
        try:
            # Call the Paginated RPC
            resp = supabase.rpc("get_daily_market_vectors", {
                "target_date": target_date,
                "page_size": page_size,
                "page_num": page
            }).execute()
            
            # If no data returned, we are done
            if not resp.data:
                break
                
            # Process this batch
            for item in resp.data:
                vec = item['vector']
                # Handle different return formats (string vs list)
                if isinstance(vec, str):
                    vec = json.loads(vec)
                
                all_records.append({
                    "ticker": item['ticker'],
                    "vector": np.array(vec)
                })
            
            # If we got fewer rows than the page size, it's the last page
            if len(resp.data) < page_size:
                break
                
            page += 1
            
        except Exception as e:
            print(f"    ⚠️ Error on page {page}: {e}")
            raise e
            
    return pd.DataFrame(all_records)

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"🚀 Starting Walk-Forward Generation (Last {HISTORY_DAYS} Days)...")
    
    # 1. Generate Date List (Oldest to Newest)
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(HISTORY_DAYS)]
    dates.reverse() 
    
    full_history = []
    previous_positions = None 
    
    # 2. Iterate Through Time
    for date_str in dates:
        print(f"📅 Processing {date_str}...", end=" ", flush=True)
        
        try:
            df = fetch_daily_vectors_rpc(date_str)
        except Exception as e:
            print(f"\n   ❌ Failed to fetch {date_str}: {e}")
            continue
        
        if df is None or df.empty:
            print("Skipped (No data)")
            continue
        
        print(f"✅ Found {len(df)} tickers.")
            
        # Prepare Matrix
        if len(df) < PERPLEXITY + 1:
            print(f"   ⚠️ Not enough data points ({len(df)}) for t-SNE.")
            continue
            
        matrix = np.stack(df['vector'].values)
        
        # 3. RUN t-SNE (WALK-FORWARD)
        n_samples = matrix.shape[0]
        init_matrix = None
        
        if previous_positions is not None:
            init_build = []
            
            for t in df['ticker']:
                if t in previous_positions:
                    init_build.append(previous_positions[t])
                else:
                    init_build.append(np.random.rand(2)) 
            
            init_matrix = np.array(init_build)
            
        tsne = TSNE(
            n_components=2, 
            perplexity=min(PERPLEXITY, n_samples - 1), 
            n_iter=500, 
            learning_rate=100,
            init=init_matrix if init_matrix is not None else 'pca',
            random_state=42
        )
        
        embeddings = tsne.fit_transform(matrix)
        
        # 4. Save & Update State
        day_map = {}
        for i, row in df.iterrows():
            ticker = row['ticker']
            x, y = embeddings[i]
            
            # Store for next iteration
            day_map[ticker] = [x, y]
            
            # Add to export list
            full_history.append({
                "date": date_str,
                "ticker": ticker,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "label": "News"
            })
            
        previous_positions = day_map

    # 5. EXPORT
    output_path = "market_physics_history.json"
    with open(output_path, "w") as f:
        json.dump({"data": full_history}, f)
        
    print(f"\n✨ DONE. History saved to {output_path}")
