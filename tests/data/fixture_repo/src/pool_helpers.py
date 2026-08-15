from src.pool import ConnectionPool


def with_connection(pool: ConnectionPool, fn):
    conn = pool.acquire()
    try:
        return fn(conn)
    finally:
        pool.release(conn)


def acquire_batch(pool: ConnectionPool, count):
    return [pool.acquire() for _ in range(count)]
