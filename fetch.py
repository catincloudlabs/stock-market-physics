import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json

# 1. FETCH DATA (Mocking your Supabase fetch for brevity)
# In reality: SELECT * FROM news_vectors WHERE date >= NOW() - 30 days
print("📉 Fetching 30-day vector history...")

# Structure: Date -> Ticker -> Vector (1536 dim)
# We simulate a "drift" here for demonstration
dates = pd.date_range(end=pd.Timestamp.now(), periods=30).strftime('%Y-%m-%d').tolist()
tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "PLTR"] # + 500 more

# Placeholder: Create 'drifting' vectors to simulate market changing
history_data = []
current_vectors = np.random.rand(len(tickers), 50) # Reduced dims for speed

for date in dates:
    # Drift the physics slightly (market evolution)
    noise = np.random.normal(0, 0.05, current_vectors.shape)
    current_vectors += noise 
    
    history_data.append({
        "date": date,
        "vectors": current_vectors.copy(), 
        "tickers": tickers
    })

# 2. WALK-FORWARD t-SNE ENGINE
print("🧠 Calculating stable trajectories...")

stabilized_map = []
previous_embedding = None

for day_idx, day_data in enumerate(history_data):
    print(f"  Processing Day {day_idx+1}/{len(history_data)}: {day_data['date']}")
    
    # A. INITIALIZATION STRATEGY
    # Day 0: Use PCA (deterministic start)
    # Day N: Use Day N-1's coordinates as the starting positions
    if previous_embedding is None:
        init_mode = 'pca'
    else:
        init_mode = previous_embedding # The "Memory" of the system

    # B. RUN t-SNE
    # We lower the learning rate for Day N to prevent explosive jumps
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        init=init_mode, 
        learning_rate=100 if previous_embedding is None else 50,
        n_iter=500,
        random_state=42
    )
    
    embedding = tsne.fit_transform(day_data['vectors'])
    previous_embedding = embedding # Save for tomorrow
    
    # C. FORMAT FOR FRONTEND
    day_snapshot = []
    for i, ticker in enumerate(day_data['tickers']):
        day_snapshot.append({
            "ticker": ticker,
            "x": round(float(embedding[i, 0]), 2),
            "y": round(float(embedding[i, 1]), 2),
            "date": day_data['date'],
            "cluster_id": 0, # You would calculate KMeans here
            "label": "Sector"
        })
    
    stabilized_map.extend(day_snapshot)

# 3. SAVE THE MOVIE
with open("market_physics_history.json", "w") as f:
    json.dump({"data": stabilized_map}, f)

print("✨ Trajectories generated. Ready for React.")
