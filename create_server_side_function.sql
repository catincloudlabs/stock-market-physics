create or replace function get_daily_market_vectors(target_date date)
returns table (ticker text, vector vector(1536))
language sql
as $$
  select 
    kg.target_node as ticker,
    -- Calculate the average (centroid) vector for the ticker
    avg(nv.embedding) as vector
  from news_vectors nv
  -- Join News with the Knowledge Graph
  join knowledge_graph kg on cast(nv.id as text) = kg.source_node
  where 
    -- Filter by the specific day
    nv.published_at >= target_date::timestamp
    and nv.published_at < (target_date + 1)::timestamp
    and kg.edge_type = 'MENTIONS'
  group by kg.target_node;
$$;
