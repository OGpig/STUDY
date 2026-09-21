#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
验证「训练时每 epoch 28 步、而非 200/8=25 步」的来源。

结论（预期）：DataLoader 的 num_workers 会把 datapipe **分片**给各个 worker，
每个 worker 独立组批：ceil(样本/worker数 / batch_size) 批 → 加起来比 25 多。
这不是数据重复，只是各 worker 的最后一批不满。

必须写成真实文件跑：Windows 的 multiprocessing 用 spawn，
子进程要重新 import 主模块，stdin（`python - <<EOF`）会直接 OSError。
"""

import collections
import itertools
import os
import sys

import yaml

sys.path.insert(0, r"F:\embedded\prepare\examples\wenet-main")


def main() -> None:
    from wenet.utils.init_dataset import init_dataset
    from wenet.utils.init_tokenizer import init_tokenizer

    os.chdir(r"F:\embedded\prepare\work\aishell")
    cfg = yaml.safe_load(open("exp/smoke/train.yaml", encoding="utf-8"))
    tok = init_tokenizer(cfg)
    ds = init_dataset("asr", "raw", "data/train_smoke/data.list", tok,
                      cfg["dataset_conf"], partition=True, split="train")

    from torch.utils.data import DataLoader
    for nw in (0, 4):
        dl = DataLoader(ds, batch_size=None, num_workers=nw,
                        prefetch_factor=2 if nw else None)
        sizes = []
        for b in itertools.islice(dl, 30):
            sizes.append(len(b["keys"]))
        c = collections.Counter(sizes)
        print(f"num_workers={nw}: 前 {len(sizes)} 批  "
              f"批大小分布={sorted(c.items())}", file=sys.stderr, flush=True)
        if nw == 0:
            print(f"  → 200/8 = 25 批，说明数据本身无重复",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
