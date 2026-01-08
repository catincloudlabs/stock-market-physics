import os
import json
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential

# 1. SETUP
load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# CONFIG
HISTORY_DAYS = 30
PERPLEXITY = 30  

# --- RETRY LOGIC FOR DB ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_daily_vectors_rpc(target_date):
    """
    Calls the server-side Postgres function to get daily averages.
    This is FAST because no heavy data is transferred over the network.
    """
    try:
        # Call the RPC function we just created
        resp = supabase.rpc("get_daily_market_vectors", {"target_date": target_date}).execute()
        
        if not resp.data:
            return None
            
        # The data comes back already aggregated: [{'ticker': 'AAPL', 'vector': [...]}, ...]
        # We just need to convert the vector string/list to numpy
        cleaned_data = []
        for item in resp.data:
            # Supabase might return the vector as a string or list depending on the driver version
            vec = item['vector']
            if isinstance(vec, str):
                vec = json.loads(vec)
            
            cleaned_data.append({
                "ticker": item['ticker'],
                "vector": np.array(vec)
            })
            
        return pd.DataFrame(cleaned_data)
        
    except Exception as e:
        print(f"    ⚠️ RPC Error: {e}")
        raise e

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
        print(f"📅 Processing {date_str}...", end=" ")
        
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
