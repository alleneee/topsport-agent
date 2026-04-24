-- sliding_window.lua — atomic multi-dimension rate limiter
-- version: 1  (bump if KEY format changes; Python side must prefix keys with vN)
--
-- Args:
--   KEYS[i]  : ZSET key per rule ("ratelimit:sw:v1:{scope}:{identity}")
--   ARGV[1]  : now_ms
--   ARGV[2]  : unique member suffix (uuid fragment, avoids ZADD dedup on same-ms requests)
--   ARGV[3..] : limit_i, window_ms_i  (alternating pairs, length = 2*#KEYS)
--
-- Returns: {allowed (1/0), denied_idx_1based, count, limit, reset_at_ms}
--   - on deny: allowed=0, denied_idx is the rule that tripped, count is current bucket count, limit is that rule's limit, reset_at_ms = now + window
--   - on allow: allowed=1, other fields 0 (middleware computes headers from known rules)

local now = tonumber(ARGV[1])
local suffix = ARGV[2]
local n = #KEYS

-- Phase 1: check each rule in order; short-circuit on first over-quota
for i = 1, n do
  local limit  = tonumber(ARGV[1 + i*2])
  local window = tonumber(ARGV[2 + i*2])
  redis.call('ZREMRANGEBYSCORE', KEYS[i], 0, now - window)
  local count = redis.call('ZCARD', KEYS[i])
  if count >= limit then
    return {0, i, count, limit, now + window}
  end
end

-- Phase 2: all passed — record the request in every dimension
for i = 1, n do
  local window = tonumber(ARGV[2 + i*2])
  redis.call('ZADD', KEYS[i], now, now .. ':' .. suffix)
  redis.call('PEXPIRE', KEYS[i], window)
end
return {1, 0, 0, 0, 0}
