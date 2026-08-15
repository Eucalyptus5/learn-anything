from src.pool import ConnectionPool

pool = ConnectionPool(size=4)
conn = pool.acquire()
result = conn
