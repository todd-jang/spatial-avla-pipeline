#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from src.core.bridge import GraphBridge


import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--out', required=True)
    ap.add_argument('--qwen-bits', type=int, choices=[4, 16], default=None, 
                    help='Simulate Qwen VLM forward pass (16=FP16, 4=4-bit)')
    args = ap.parse_args()

    if args.qwen_bits:
        os.environ["QWEN_BITS"] = str(args.qwen_bits)

    b = GraphBridge(fast_mode=True)
    lats = []
    for i in range(args.frames):
        r = b.invoke_frame(i, 1759500000000.0 + i * 33.333)
        lats.append(float(r['meta_exec'].get('lat_ms', 0.0)))
        assert lats[-1] < 330.0, f'lat >=330 at {i}: {lats[-1]}'

    raw = list(lats)
    lats.sort()
    p50 = lats[len(lats)//2]
    p95 = lats[int(len(lats)*0.95)] if lats else 0
    worst = max(raw)
    verdict = 'PASS' if (worst < 330.0 and p95 < 330.0) else 'FAIL'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = [f'frames={args.frames}',
            f'p50={p50}',
            f'p95={p95}',
            f'max={worst}',
            f'all<330=True',
            f'verdict={verdict}',
            'latencies_ms=' + ','.join(f'{v:.3f}' for v in raw)]
    out.write_text('\n'.join(body) + '\n')
    print(f'p50={p50} p95={p95} max={worst} verdict={verdict}')


if __name__ == '__main__':
    main()
