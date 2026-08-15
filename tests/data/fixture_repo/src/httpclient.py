from src.pool import ConnectionPool

DEFAULT_TIMEOUT_S = 10.0


class HttpClient:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.request_count = 0

    def get(self, url):
        conn = self.pool.acquire()
        self.request_count += 1
        return send_request(conn, "GET", url)

    def post(self, url, body):
        conn = self.pool.available.pop()
        self.request_count += 1
        return send_request(conn, "POST", url, body=body)


def send_request(conn, method, url, body=None):
    return {"conn": conn, "method": method, "url": url, "body": body}
