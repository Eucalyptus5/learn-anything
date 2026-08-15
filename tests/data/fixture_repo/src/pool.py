acquire_timeout_s = 5.0
release_timeout_s = 2.0


class ConnectionPool:
    def __init__(self, size):
        self.size = size
        self.available = list(range(size))
        self.in_use = set()

    def acquire(self):
        if not self.available:
            raise RuntimeError("pool exhausted")
        acquired_conn = self.available.pop()
        self.in_use.add(acquired_conn)
        return acquired_conn

    def release(self, conn):
        self.in_use.discard(conn)
        self.available.append(conn)

    def stats(self):
        return {"available": len(self.available), "in_use": len(self.in_use)}


def acquire_or_wait(pool, retries=3):
    for _ in range(retries):
        if pool.available:
            return pool.acquire()
        pool.release(None)
    return pool.acquire()
