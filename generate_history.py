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
    Refactored to fetch IDs first, then fetch data in small chunks to avoid
    timeouts on large payloads.
    """
    start = f"{target_date} 00:00:00"
    end = f"{target_date} 23:59:59"
    
    # 1. Get ONLY IDs for the day (Fast, lightweight)
    # This avoids downloading 50MB+ of vectors in one go.
    ids_resp = supabase.table("news_vectors") \
        .select("id") \
        .gte("published_at", start) \
        .lte("published_at", end) \
        .execute()
        
    if not ids_resp.data:
        return None
        
    all_ids = [item['id'] for item in ids_resp.data]
    if not all_ids:
        return None
    
    # 2. Process in manageable chunks (e.g., 100 articles at a time)
    CHUNK_SIZE = 100
    ticker_vectors = {} # {ticker: [list of vectors]}
    
    # Iterate through the IDs in batches
    for i in range(0, len(all_ids), CHUNK_SIZE):
        chunk_ids = all_ids[i : i + CHUNK_SIZE]
        
        try:
            # A. Fetch Embeddings for this small chunk
            vec_resp = supabase.table("news_vectors") \
                .select("id, embedding") \
                .in_("id", chunk_ids) \
                .execute()
                
            if not vec_resp.data: continue
            
            # Map ID -> Vector
            temp_map = {item['id']: np.array(item['embedding']) for item in vec_resp.data}
            
            # B. Fetch Edges for this small chunk
            # Note: knowledge_graph uses string source_ids
            chunk_ids_str = [str(uid) for uid in chunk_ids]
            
            edges_resp = supabase.table("knowledge_graph") \
                .select("source_node, target_node") \
                .in_("source_node", chunk_ids_str) \
                .eq("edge_type", "MENTIONS") \
                .execute()
            
            # C. Aggregate immediately to save memory
            if edges_resp.data:
                for edge in edges_resp.data:
                    ticker = edge['target_node']
                    # source_node is string, convert to int for lookup
                    try:
                        src_id = int(edge['source_node'])
                        if src_id in temp_map:
                            if ticker not in ticker_vectors:
                                ticker_vectors[ticker] = []
                            ticker_vectors[ticker].append(temp_map[src_id])
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"    ⚠️ Chunk error {i}-{i+CHUNK_SIZE}: {e}")
            continue

    # 3. Final Averaging (Centroid Calculation)
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
