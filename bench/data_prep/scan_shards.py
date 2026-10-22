"""Read footers + label columns of every shard (no image bytes). Writes data/shard_index.json."""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from remote_parquet import RangeFile, shard_url

index, total = [], 0
for i in range(15):
    f = RangeFile(shard_url(i))
    pf = pq.ParquetFile(f)
    md = pf.metadata
    names = pf.schema_arrow.names
    lab_col = names.index("label"); img_col = names.index("image")
    rgs = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        labels = pq.ParquetFile(f).read_row_group(g, columns=["label"]).column("label").to_pylist()
        rgs.append({"rows": rg.num_rows, "image_bytes": rg.column(img_col).total_compressed_size, "labels": labels})
    total += f.bytes_fetched
    index.append({"shard": i, "size": f.size, "row_groups": rgs})
    print(f"shard {i}: rows={md.num_rows} row_groups={md.num_row_groups} "
          f"mean_rg_img_MB={sum(r['image_bytes'] for r in rgs)/len(rgs)/1e6:.1f} fetched_so_far={total/1e6:.1f} MB", flush=True)
Path("data/shard_index.json").write_text(json.dumps(index))
print("schema:", names, "| total bytes fetched for index:", total)
