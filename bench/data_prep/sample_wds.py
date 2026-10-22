"""Random-offset sampling of ImageNet-val images from class-sorted WebDataset tar shards.

Reads only ~300 KB windows via HTTP Range (hf-mirror.com). For each random byte offset:
find the next valid ustar header (checksum + magic + name pattern), take the first complete
"<key>.cls" + "<key>.jpg" pair from the same key, skip + count anything that fails.
Offset sampling is mildly size-biased (an image's chance depends on the *preceding* member's size).
"""
import json, random, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "clip-benchmark/wds_imagenet1k"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/test/"
OUT = Path(__file__).resolve().parents[2] / "data"
N_EVAL, N_CALIB, SEED = 1000, 300, 0
WINDOW, MAX_WINDOW, BYTE_CAP = 384 * 1024, 2 * 1024 * 1024, 800 * 1024 * 1024
NAME = re.compile(rb"^[A-Za-z0-9_\-]+\.(cls|jpg)$")

fetched = 0
_lock = threading.Lock()
WORKERS = 6


def get(url, a, b):
    """Bytes [a, b) of url (clamped by server), with retries."""
    global fetched
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            with _lock:
                fetched += len(data)
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def size_of(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[1])


def parse_header(blk):
    """Return (name, size) if blk is a valid ustar header, else None."""
    if len(blk) < 512 or blk[257:262] != b"ustar":
        return None
    try:
        stored = int(blk[148:154].strip(b"\0 ") or b"x", 8)
    except ValueError:
        return None
    if stored != sum(blk[:148]) + 8 * 32 + sum(blk[156:512]):
        return None
    name = blk[:100].split(b"\0")[0]
    if not NAME.match(name):
        return None
    try:
        return name.decode(), int(blk[124:136].strip(b"\0 "), 8)
    except ValueError:
        return None


def members(buf, start):
    """Yield (name, size, data_start) for consecutive valid headers beginning at buf[start]."""
    pos = start
    while pos + 512 <= len(buf):
        h = parse_header(buf[pos:pos + 512])
        if h is None:
            return
        yield h[0], h[1], pos + 512
        pos += 512 + (h[1] + 511) // 512 * 512


def sample_one(url, shard_size, off):
    """Return (key, label, jpg_bytes) or a skip-reason string."""
    a = (off + 511) // 512 * 512
    window = WINDOW
    while True:
        buf = get(url, a, min(a + window, shard_size))
        # first valid header at a 512-aligned position
        first = next((p for p in range(0, len(buf) - 511, 512) if parse_header(buf[p:p + 512])), None)
        if first is None:
            if window >= MAX_WINDOW or a + len(buf) >= shard_size:
                return "no_header"
            window *= 2; continue
        ms = list(members(buf, first))
        # complete pair: <key>.cls immediately followed by <key>.jpg, both fully inside buf
        for i in range(len(ms) - 1):
            (n1, s1, d1), (n2, s2, d2) = ms[i], ms[i + 1]
            k1, e1 = n1.rsplit(".", 1); k2, e2 = n2.rsplit(".", 1)
            if e1 == "cls" and e2 == "jpg" and k1 == k2 and d2 + s2 <= len(buf):
                try:
                    label = int(buf[d1:d1 + s1].decode().strip())
                except ValueError:
                    return "bad_label"
                jpg = buf[d2:d2 + s2]
                if jpg[:2] != b"\xff\xd8":
                    return "bad_jpeg"
                return k1, label, jpg
        if window >= MAX_WINDOW or a + len(buf) >= shard_size:
            return "no_complete_pair"
        window *= 2


def main():
    tree = json.load(urllib.request.urlopen(f"https://hf-mirror.com/api/datasets/{REPO}/tree/main/test", timeout=60))
    shards = sorted((f["path"].split("/")[-1], f["size"]) for f in tree if f["path"].endswith(".tar"))
    shards.sort(key=lambda x: int(x[0].split(".")[0]))
    total = sum(s for _, s in shards)
    print(f"{len(shards)} shards, {total/1e9:.2f} GB", flush=True)
    rng = random.Random(SEED)
    got, seen, skips, t0 = [], set(), {}, time.time()

    def draw():
        g = rng.randrange(total)
        for name, size in shards:
            if g < size: break
            g -= size
        return name, size, g

    pool = ThreadPoolExecutor(WORKERS)
    while len(got) < N_EVAL + N_CALIB:
        if fetched > BYTE_CAP:
            print(f"ABORT: byte cap {BYTE_CAP/1e6:.0f} MB reached with {len(got)} images", flush=True); break
        draws = [draw() for _ in range(WORKERS * 4)]            # offsets are seeded; results consumed in draw order
        for (name, size, g), r in zip(draws, pool.map(lambda d: sample_one(BASE + d[0], d[1], d[2]), draws)):
            if len(got) >= N_EVAL + N_CALIB: break
            if isinstance(r, str):
                skips[r] = skips.get(r, 0) + 1; continue
            key = (name, r[0])
            if key in seen:
                skips["duplicate"] = skips.get("duplicate", 0) + 1; continue
            seen.add(key); got.append((name, r[0], r[1], r[2]))
        print(f"{len(got)} images, {fetched/1e6:.0f} MB, {time.time()-t0:.0f}s, skips={skips}", flush=True)
    manifest = {"repo": REPO, "seed": SEED, "bytes_fetched": fetched, "skips": skips, "eval": [], "calib": []}
    for i, (name, key, label, jpg) in enumerate(got):
        split = "eval" if i < N_EVAL else "calib"
        d = OUT / split; d.mkdir(parents=True, exist_ok=True)
        fn = f"{split}_{i:04d}.jpg"
        (d / fn).write_bytes(jpg)
        manifest[split].append({"file": fn, "shard": name, "key": key, "label": label})
    ids = lambda s: {(m["shard"], m["key"]) for m in manifest[s]}
    assert not (ids("eval") & ids("calib")), "eval/calib overlap"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ev = [m["label"] for m in manifest["eval"]]
    print(f"DONE eval={len(ev)} calib={len(manifest['calib'])} distinct eval classes={len(set(ev))} "
          f"skips={skips} fetched={fetched/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
