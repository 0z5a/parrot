import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

binaries = {"p1": "bench_optin_p1", "p2_default": "bench_optin_default", "p2_async": "bench_optin_async"}
with Path(sys.argv[1]).open("w") as output:
    for pair, arms in enumerate(itertools.permutations(binaries)):
        for mode in ("chain", "public"):
            for segments, width in ((128, 8), (2048, 128)):
                for arm in arms:
                    if mode == "public" and arm == "p2_async":
                        continue
                    cmd = [str(Path("builds") / binaries[arm]), mode, str(segments), str(width)]
                    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
                    row = json.loads(result.stdout)
                    assert row["correct"]
                    row.update(pair=pair, arm=arm)
                    output.write(json.dumps(row)+"\n")
                    output.flush()
        print(f"pair {pair} complete", flush=True)
print("60 arms passed", flush=True)
