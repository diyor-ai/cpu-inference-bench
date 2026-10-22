"""File-like object over HTTP Range requests so pyarrow reads only the bytes it needs."""
import io, time, urllib.request

BASE = "https://hf-mirror.com/datasets/Tsomaros/Imagenet-1k_validation/resolve/main/data/"


def shard_url(i, n=15):
    return f"{BASE}validation-{i:05d}-of-{n:05d}.parquet"


class RangeFile(io.RawIOBase):
    def __init__(self, url, retries=4):
        self.url, self.pos, self.retries, self.bytes_fetched = url, 0, retries, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
