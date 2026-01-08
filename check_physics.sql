-- Count the particles
select 
  (select count(*) from stocks_ohlc) as stock_count,
  (select count(*) from news_vectors) as news_count,
  (select count(*) from knowledge_graph) as edge_count;


-- Inspect news physics
select 
  id, 
  headline, 
  substring(cast(embedding as text) from 1 for 50) as vector_preview 
from news_vectors 
order by id desc 
limit 5;

-- Connectivity check
select 
  kg.source_node as news_id,
  nv.headline,
  kg.edge_type,
  kg.target_node as connected_stock
from knowledge_graph kg
join news_vectors nv on cast(nv.id as text) = kg.source_node
limit 10;

-- We need a fake vector for the query (in production, OpenAI creates this)
-- Here we just ask for "Anything" by passing a zero-vector, 
-- or you can rely on keyword matching if we added that. 
-- For now, let's just test the function call structure:

select * from hybrid_graph_search(
  (select embedding from news_vectors limit 1), -- Borrow an existing vector to simulate a query
  0.5, -- Similarity Threshold
  5    -- Limit
);
