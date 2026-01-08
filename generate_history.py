import os
import json
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. SETUP
load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# CONFIG
HISTORY_DAYS = 30
PERPLEXITY = 30  # Tuning parameter for t-SNE (5-50)

def fetch_daily_vectors(target_date):
    """
    Fetches news vectors for a specific date and averages them by ticker.
    Returns: DataFrame [ticker, vector_array]
    """
    # 1. Get News IDs for the date
    # Note: In a real prod environment, you'd likely have a materialized view for this.
    # For now, we query raw tables.
    
    print(f"  ... fetching news for {target_date}")
    
    # Time range: The whole day
    start = f"{target_date} 00:00:00"
    end = f"{target_date} 23:59:59"
    
    try:
        # A. Get News Vectors
        news_resp = supabase.table("news_vectors") \
            .select("id, embedding") \
            .gte("published_at", start) \
            .lte("published_at", end) \
            .execute()
            
        if not news_resp.data:
            return None
            
        news_map = {item['id']: np.array(item['embedding']) for item in news_resp.data}
        news_ids = list(news_map.keys())
        
        # B. Get Edges (News -> Ticker)
        # We process in chunks if needed, but for daily volume this usually fits
        edges_resp = supabase.table("knowledge_graph") \
            .select("source_node, target_node") \
            .in_("source_node", news_ids) \
            .eq("edge_type", "MENTIONS") \
            .execute()
            
        if not edges_resp.data:
            return None

        # C. Aggregate Vectors by Ticker
        ticker_vectors = {}
        
        for edge in edges_resp.data:
            ticker = edge['target_node']
            news_id = int(edge['source_node']) # Ensure ID type matches
            
            if news_id in news_map:
                if ticker not in ticker_vectors:
                    ticker_vectors[ticker] = []
                ticker_vectors[ticker].append(news_map[news_id])
        
        # D. Average them (Centroid)
        final_data = []
        for ticker, vectors in ticker_vectors.items():
            if len(vectors) > 0:
                # Mean vector of all news mentioning this stock today
                centroid = np.mean(vectors, axis=0)
                final_data.append({
                    "ticker": ticker,
                    "vector": centroid
                })
                
        return pd.DataFrame(final_data)

    except Exception as e:
        print(f"Error fetching day {target_date}: {e}")
        return None

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"🚀 Starting Walk-Forward Generation (Last {HISTORY_DAYS} Days)...")
    
    # 1. Generate Date List
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(HISTORY_DAYS)]
    dates.reverse() # Process oldest to newest!
    
    full_history = []
    previous_positions = None # The anchor for walk-forward
    
    # 2. Iterate Through Time
    for date_str in dates:
        print(f"📅 Processing {date_str}...")
        
        df = fetch_daily_vectors(date_str)
        
        if df is None or df.empty:
            print("   ⚠️ No data found (weekend/holiday?), skipping.")
            continue
            
        # Prepare Matrix
        # Ensure we have enough data for t-SNE (needs > perplexity samples)
        if len(df) < PERPLEXITY + 1:
            print(f"   ⚠️ Not enough data points ({len(df)}) for t-SNE.")
            continue
            
        matrix = np.stack(df['vector'].values)
        
        # 3. RUN t-SNE (WALK-FORWARD)
        # Initialization Logic:
        # If we have previous positions for these specific tickers, use them.
        # Otherwise, random init.
        
        n_samples = matrix.shape[0]
        init_matrix = None
        
        if previous_positions is not None:
            # We need to map yesterday's coordinates to today's tickers.
            # Tickers appear/disappear based on news flow.
            
            init_build = []
            random_count = 0
            
            for t in df['ticker']:
                if t in previous_positions:
                    init_build.append(previous_positions[t])
                else:
                    # New ticker today? Initialize randomly
                    init_build.append(np.random.rand(2)) 
                    random_count += 1
            
            init_matrix = np.array(init_build)
            # Normalize init range to help t-SNE
            # init_matrix = init_matrix * 0.0001 
            
        # Run Algorithm
        # Note: 'init' parameter supports array
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
                # Optional: Add cluster label logic here later
                "label": "News"
            })
            
        previous_positions = day_map
        print(f"   ✅ Processed {len(df)} tickers.")

    # 5. EXPORT
    output_path = "market_physics_history.json"
    with open(output_path, "w") as f:
        json.dump({"data": full_history}, f)
        
    print(f"\n✨ DONE. History saved to {output_path}")
