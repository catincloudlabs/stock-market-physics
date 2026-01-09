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
from textblob import TextBlob

# 1. SETUP
load_dotenv()

# Modern timeout configuration for reliability
timeout_config = httpx.Timeout(120.0, connect=10.0, read=120.0)
opts = ClientOptions(postgrest_client_timeout=120)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_KEY"),
    options=opts
)

# CONFIG
HISTORY_DAYS = 30  
PERPLEXITY = 30  

# --- MATH UTILS: Procrustes Alignment ---
# This mathematically "locks" today's chart to yesterday's orientation.
def align_to_reference(source_matrix, target_matrix, source_tickers, target_tickers):
    """
    Rotates, scales, and translates 'target_matrix' (Today) to best fit 'source_matrix' (Yesterday).
    This stabilizes the chart so tickers don't fly around randomly due to t-SNE rotation.
    """
    # 1. Identify Anchors (Tickers present in both days)
    common_tickers = list(set(source_tickers) & set(target_tickers))
    
    if len(common_tickers) < 3:
        # Not enough points to lock orientation, return original
        return target_matrix

    # 2. Extract Anchor Points
    src_indices = [source_tickers.index(t) for t in common_tickers]
    tgt_indices = [target_tickers.index(t) for t in common_tickers]
    
    A = source_matrix[src_indices] # Yesterday (Reference)
    B = target_matrix[tgt_indices] # Today (Target)

    # 3. Center the data
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # 4. SVD for optimal rotation (Kabsch Algorithm)
    # This finds the best rotation matrix R to minimize distance between AA and BB
    H = np.dot(BB.T, AA)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Handle reflection case (ensure we don't accidentally "mirror" the chart)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 5. Apply Transformation to ENTIRE target_matrix (Today)
    # Move Today to origin -> Rotate -> Move to Yesterday's centroid
    return np.dot(target_matrix - centroid_B, R) + centroid_A

# --- DB FETCHING ---
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_daily_vectors_rpc(target_date):
    """
    Fetches daily averages using Server-Side Paging.
    """
    all_records = []
    page = 0
    page_size = 100
    
    while True:
        try:
            resp = supabase.rpc("get_daily_market_vectors", {
                "target_date": target_date,
                "page_size": page_size,
                "page_num": page
            }).execute()
            
            if not resp.data:
                break
                
            for item in resp.data:
                vec = item['vector']
                # Handle PostgREST stringification if present
                if isinstance(vec, str):
                    vec = json.loads(vec)
                
                # Enforce float32 for consistency
                vec_np = np.array(vec, dtype=np.float32)
                
                # Basic sentiment extraction
                headline = item.get('headline', '')
                sentiment = 0
                if headline:
                    try:
                        sentiment = TextBlob(headline).sentiment.polarity
                    except:
                        sentiment = 0
                
                all_records.append({
                    "ticker": item['ticker'],
                    "vector": vec_np,
                    "headline": headline,    
                    "sentiment": sentiment   
                })
            
            if len(resp.data) < page_size:
                break
            page += 1
            
        except Exception as e:
            print(f"    ⚠️ Error on page {page}: {e}")
            raise e
    
    return pd.DataFrame(all_records)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting Stabilized Walk-Forward Generation (Last {HISTORY_DAYS} Days)...")
    
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(HISTORY_DAYS)]
    dates.reverse() # Process oldest to newest
    
    full_history = []
    
    # State tracking for stabilization
    prev_matrix = None
    prev_tickers = None
    
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
            
        # Prepare Matrix
        current_matrix = np.stack(df['vector'].values)
        current_tickers = df['ticker'].tolist()
        
        if len(df) < 5: 
            print("Skipped (Not enough data)")
            continue

        # 1. Initialization Hint (Soft Stabilization)
        # We start the optimization where the points were yesterday.
        n_samples = current_matrix.shape[0]
        init_matrix = None
        
        if prev_matrix is not None:
            prev_map = {t: pos for t, pos in zip(prev_tickers, prev_matrix)}
            init_build = []
            for t in current_tickers:
                if t in prev_map:
                    init_build.append(prev_map[t])
                else:
                    # New ticker? Start it random.
                    init_build.append(np.random.rand(2)) 
            init_matrix = np.array(init_build)
            
        # 2. Run t-SNE (Rigorous Configuration)
        # Using exact parameters ensures reproducibility.
        tsne = TSNE(
            n_components=2, 
            perplexity=min(PERPLEXITY, n_samples - 1), 
            init=init_matrix if init_matrix is not None else 'pca',
            learning_rate='auto',
            n_iter=1000,     # Explicitly set iterations for stability
            random_state=42, # Lock random seed
            method='barnes_hut'
        )
        
        embeddings_raw = tsne.fit_transform(current_matrix)
        
        # 3. Apply Procrustes Alignment (Hard Stabilization)
        # This mathematically rotates today's map to align with yesterday's map.
        if prev_matrix is not None:
            embeddings_stabilized = align_to_reference(
                source_matrix=prev_matrix, 
                target_matrix=embeddings_raw, 
                source_tickers=prev_tickers, 
                target_tickers=current_tickers
            )
        else:
            # First day defines the coordinate system
            embeddings_stabilized = embeddings_raw
            
        # 4. Save & Update State
        for i, row in df.iterrows():
            ticker = row['ticker']
            x, y = embeddings_stabilized[i]
            
            full_history.append({
                "date": date_str,
                "ticker": ticker,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "headline": row['headline'],
                "sentiment": round(row['sentiment'], 2)
            })
            
        # Update references for the next iteration
        prev_matrix = embeddings_stabilized
        prev_tickers = current_tickers
        
        print(f"✅ Aligned & Saved ({len(df)} tickers)")

    # 5. EXPORT
    output_path = "market_physics_history.json"
    with open(output_path, "w") as f:
        json.dump({"data": full_history}, f)
        
    print(f"\n✨ DONE. Stabilized History saved to {output_path}")
