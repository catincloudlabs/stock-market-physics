drop function if exists get_daily_market_vectors;

create or replace function get_daily_market_vectors(
  target_date date, 
  page_size int default 100, 
  page_num int default 0
)
returns table (ticker text, vector vector(1536))
language sql
as $$
  with daily_news as (
    -- 1. Optimize: Filter news for the specific date FIRST
    select id, embedding
    from news_vectors
    where published_at >= target_date::timestamp
      and published_at < (target_date + 1)::timestamp
  )
  select 
    kg.target_node as ticker,
    avg(dn.embedding) as vector
  from daily_news dn
  -- 2. Join only the small subset of daily news to the graph
  join knowledge_graph kg on kg.source_node = cast(dn.id as text)
  where kg.edge_type = 'MENTIONS'
  group by kg.target_node
  order by kg.target_node
  -- 3. Pagination limits
  limit page_size offset (page_num * page_size);
$$;
