# Release Notes

## 0.3.0

- Added a bounded connection pool in src/pool.py so callers stop opening a
  new socket per request.
- HttpClient now calls pool.acquire() before every outbound request and
  returns the connection when the response is read.
- Fixed a leak where a failed request never called release, starving the
  pool under load.
- pool_helpers.with_connection wraps acquire and release in a single
  helper so callers cannot forget the cleanup step.

## 0.2.0

- Initial httpclient module, no pooling yet, one socket per request.
- Basic retry loop added to acquire_or_wait for transient failures.
