-- 1. Accelerate Date Filtering
-- Allows the DB to instantly grab only today's news
create index if not exists idx_news_published on news_vectors(published_at);

-- 2. Accelerate Graph Joins
-- Allows the DB to instantly find edges for a specific news ID
create index if not exists idx_kg_source on knowledge_graph(source_node);

-- 3. Accelerate Edge Filtering
-- Allows the DB to ignore non-MENTIONS edges quickly
create index if not exists idx_kg_type on knowledge_graph(edge_type);

-- 4. Update Database Statistics
-- Tells the query planner that these indexes exist
analyze news_vectors;
analyze knowledge_graph;
