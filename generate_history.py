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
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_daily_vectors_safe(target_date):
    """
    Fetches news vectors for a specific date and averages them by ticker.
    Includes BATCHING to prevent URL overflow errors on large datasets.
    """
    start = f"{target_date} 00:00:00"
    end = f"{target_date} 23:59:59"
    
    # A. Get News Vectors (IDs are Integers)
    # This usually returns fine even with large ranges
    news_resp = supabase.table("news_vectors") \
        .select("id, embedding") \
        .gte("published_at", start) \
        .lte("published_at", end) \
        .execute()
        
    if not news_resp.data:
        return None
        
    # Map: Integer ID -> Vector
    news_map_int = {item['id']: np.array(item['embedding']) for item in news_resp.data}
    
    # Convert Integers to Strings for Graph Query
    news_ids_str = [str(nid) for nid in news_map_int.keys()]
    
    if not news_ids_str:
        return None

    # B. Get Edges (News -> Ticker) - BATCHED
    # We chunk the IDs to avoid "URI Too Long" errors / Socket timeouts
    BATCH_SIZE = 50 
    all_edges = []
    
    # Iterate in chunks
    for i in range(0, len(news_ids_str), BATCH_SIZE):
        batch_ids = news_ids_str[i : i + BATCH_SIZE]
        
        try:
            # Supabase 'in_' filter works best with smaller lists
            edges_resp = supabase.table("knowledge_graph") \
                .select("source_node, target_node") \
                .in_("source_node", batch_ids) \
                .eq("edge_type", "MENTIONS") \
                .execute()
            
            if edges_resp.data:
                all_edges.extend(edges_resp.data)
                
        except Exception as e:
            print(f"    ⚠️ Batch fetch error (skipping chunk): {e}")
            continue

    if not all_edges:
        return None

    # C. Aggregate Vectors by Ticker
    ticker_vectors = {}
    
    for edge in all_edges:
        ticker = edge['target_node']
        source_id_str = edge['source_node'] # This is a string from DB
        
        # Convert string back to int to look up the vector
        try:
            source_id_int = int(source_id_str)
            if source_id_int in news_map_int:
                if ticker not in ticker_vectors:
                    ticker_vectors[ticker] = []
                ticker_vectors[ticker].append(news_map_int[source_id_int])
        except ValueError:
            continue
    
    # D. Average them (Centroid)
    final_data = []
    for ticker, vectors in ticker_vectors.items():
        if len(vectors) > 0:
            centroid = np.mean(vectors, axis=0)
            final_data.append({
                "ticker": ticker,
                "vector": centroid
            })
            
    return pd.DataFrame(final_data)

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
        print(f"📅 Processing {date_str}...")
        
        try:
            df = fetch_daily_vectors_safe(date_str)
        except Exception as e:
            print(f"   ❌ Failed to fetch {date_str}: {e}")
            continue
        
        if df is None or df.empty:
            print("   ⚠️ No data found (weekend/holiday?), skipping.")
            continue
            
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
        print(f"   ✅ Processed {len(df)} tickers.")

    # 5. EXPORT
    output_path = "market_physics_history.json"
    with open(output_path, "w") as f:
        json.dump({"data": full_history}, f)
        
    print(f"\n✨ DONE. History saved to {output_path}")
