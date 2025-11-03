import redis

r = redis.Redis(host='localhost', port=6379)

# Option 1: Fast approximate count (Redis >= 4.0)
count = r.dbsize()  # total keys in DB
print(f"Total keys in Redis DB: {count}")

# Option 2: Exact count of your vectors only
vector_keys = r.keys("vector:*")
print(f"Number of stored vectors in Redis: {len(vector_keys)}")
