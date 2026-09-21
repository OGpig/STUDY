#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给 WeNet 打 3 组兼容性补丁：让它能在 torch 2.11 + 精简依赖的 Windows 环境跑起来。

【为什么需要】
    WeNet main 分支是按「依赖装全 + torch 较老」的假设写的。本机是
    torch 2.11.0+cu128（RTX 5060 的 sm_120 只能走 cu128 轮子，没得选），于是三处都炸：

    1. **`torch.nn.modules.conv` 不再带出 typing 的名字**
       `wenet/models/squeezeformer/conv2d.py`
         from torch.nn.modules.conv import _ConvNd, _size_2_t, Union, _pair, Tensor, Optional
       老版 torch 的 conv.py 里恰好 `from typing import ... Union ...`，
       所以能"顺带"import 出来；2.11 起不带了 → ImportError。
       而 `init_model.py` 会把**所有** backbone 都 import 一遍（哪怕你只用 conformer），
       所以这一行直接卡死整个训练。
       → 修法：`Union` 改成从 `typing` 正常导入。

    2. **`deepspeed` 被无条件顶层 import**
       `wenet/utils/train_utils.py` 开头就 `import deepspeed` + 3 个子模块。
       deepspeed 在 Windows 上编译困难，而本机用 `train_engine=torch_ddp` **根本不需要它**。
       → 修法：改成 try/except，加 `DEEPSPEED_AVAILABLE` 标志；
         只在真的选 deepspeed 引擎时才 assert 报错。

    3. **`whisper` 被无条件顶层 import**
       `wenet/utils/init_tokenizer.py` 顶部 `from wenet.text.whisper_tokenizer import WhisperTokenizer`，
       而那个模块会 `import whisper`（openai-whisper，与 numpy 2.x 冲突）。
       我们的 tokenizer 是 `char`，用不到 whisper。
       → 修法：把 import 挪进 `if tokenizer_type == "whisper":` 分支里（惰性导入）。

【用法】
    python scripts/patch_wenet.py --check     # 只看状态
    python scripts/patch_wenet.py             # 打补丁（自动备份 .orig）
    python scripts/patch_wenet.py --revert    # 还原

    重新拉取 WeNet 源码后需要重跑一次。

【与 patch_torch_libuv.py 的区别】
    这个是**精确文本替换**：匹配不上就报错退出，绝不"差不多就改"。
    因为每处替换的上下文都不同，硬套括号扫描反而危险。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WENET = ROOT / "examples" / "wenet-main"

# (相对路径, 原文, 新文, 说明)
PATCHES: list[tuple[str, str, str, str]] = [
    (
        "wenet/models/squeezeformer/conv2d.py",
        "from torch.nn.modules.conv import _ConvNd, _size_2_t, Union, _pair, Tensor, Optional",
        "from typing import Union\n\n"
        "from torch.nn.modules.conv import _ConvNd, _size_2_t, _pair, Tensor, Optional",
        "torch 2.11 的 conv 模块不再带出 Union（改为从 typing 导入）",
    ),
    (
        "wenet/utils/train_utils.py",
        "import deepspeed\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "import torch.optim as optim\n"
        "import yaml\n"
        "from deepspeed.runtime.zero.stage3 import \\\n"
        "    estimate_zero3_model_states_mem_needs_all_live\n"
        "from deepspeed.runtime.zero.stage_1_and_2 import \\\n"
        "    estimate_zero2_model_states_mem_needs_all_live\n"
        "from deepspeed.utils.zero_to_fp32 import \\\n"
        "    convert_zero_checkpoint_to_fp32_state_dict\n"
        "from tensorboardX import SummaryWriter\n",
        "# ★ 补丁(patch_wenet.py)：deepspeed 改为可选依赖。\n"
        "#   原版无条件 import，但本机用 train_engine=torch_ddp 根本不需要它，\n"
        "#   而 deepspeed 在 Windows 上编译困难。\n"
        "try:\n"
        "    import deepspeed\n"
        "    from deepspeed.runtime.zero.stage3 import \\\n"
        "        estimate_zero3_model_states_mem_needs_all_live\n"
        "    from deepspeed.runtime.zero.stage_1_and_2 import \\\n"
        "        estimate_zero2_model_states_mem_needs_all_live\n"
        "    from deepspeed.utils.zero_to_fp32 import \\\n"
        "        convert_zero_checkpoint_to_fp32_state_dict\n"
        "    DEEPSPEED_AVAILABLE = True\n"
        "except ImportError:  # pragma: no cover\n"
        "    deepspeed = None\n"
        "    estimate_zero3_model_states_mem_needs_all_live = None\n"
        "    estimate_zero2_model_states_mem_needs_all_live = None\n"
        "    convert_zero_checkpoint_to_fp32_state_dict = None\n"
        "    DEEPSPEED_AVAILABLE = False\n"
        "\n"
        "import torch\n"
        "import torch.distributed as dist\n"
        "import torch.optim as optim\n"
        "import yaml\n"
        "from tensorboardX import SummaryWriter\n",
        "deepspeed 顶层 import 改为 try/except 可选",
    ),
    (
        "wenet/dataset/datapipes.py",
        "            except Exception as ex:\n"
        "                if self.log_error:\n"
        "                    logging.warning(str(ex))\n",
        "            except Exception as ex:\n"
        "                if self.log_error:\n"
        "                    # ★ 补丁(patch_wenet.py)：原版只打一行 str(ex)，\n"
        "                    #   样本被静默丢弃时根本看不出是哪一步失败的 ——\n"
        "                    #   丢数据而不自知是最危险的情况（训练集会莫名变小）。\n"
        "                    #   这里补上完整 traceback。\n"
        "                    import traceback\n"
        "                    logging.warning(str(ex))\n"
        "                    logging.warning(traceback.format_exc())\n",
        "map_ignore_error 丢样本时补打 traceback（否则只能看到一行 str(ex)）",
    ),
    (
        "wenet/dataset/datapipes.py",
        "    def __init__(self, filenames, mode='r'):\n"
        "        super().__init__()\n"
        "        _dp = datapipes.iter.FileLister(filenames)\n"
        "        _dp = datapipes.iter.FileOpener(_dp, mode=mode)\n"
        "        self.dp = _dp\n",
        "    def __init__(self, filenames, mode='r'):\n"
        "        super().__init__()\n"
        "        _dp = datapipes.iter.FileLister(filenames)\n"
        "        # ★ 补丁(patch_wenet.py)：FileOpener 的 encoding 默认 None =\n"
        "        #   **系统默认编码**（中文 Windows 是 GBK）。而 data.list 里含中文\n"
        "        #   （txt 字段用 ensure_ascii=False 写的），于是逐行读时 UnicodeDecodeError\n"
        "        #   → 外面又包了 map_ignore_error → **每一条样本都被静默丢弃** →\n"
        "        #   最终表现成完全不相干的 `ZeroDivisionError`（在 executor.cv 里）。\n"
        "        #   强制 utf-8；二进制模式不能传 encoding。\n"
        "        _enc = None if 'b' in mode else 'utf-8'\n"
        "        _dp = datapipes.iter.FileOpener(_dp, mode=mode, encoding=_enc)\n"
        "        self.dp = _dp\n",
        "★ data.list 按系统编码(GBK)读取 → 含中文的行全部解码失败 → 样本被静默丢弃",
    ),
    (
        "wenet/bin/train.py",
        "    if final_epoch is not None and rank == 0:\n"
        "        final_model_path = os.path.join(args.model_dir, 'final.pt')\n"
        "        os.remove(final_model_path) if os.path.exists(\n"
        "            final_model_path) else None\n"
        "        os.symlink('{}.pt'.format(final_epoch), final_model_path)\n"
        "        writer.close()\n",
        "    if final_epoch is not None and rank == 0:\n"
        "        final_model_path = os.path.join(args.model_dir, 'final.pt')\n"
        "        # ★ 补丁(patch_wenet.py)：原版这里有两个真 bug\n"
        "        #\n"
        "        #   1) 清理用的是 os.path.exists —— 它对**断链**返回 False\n"
        "        #      （因为它会去 follow 链接）。于是 os.remove 被跳过，\n"
        "        #      同一个 model_dir 重跑必然 FileExistsError 退出。\n"
        "        #      必须用 os.path.lexists（只看链接本身在不在）。\n"
        "        #\n"
        "        #   2) 链接目标写成 '{}.pt'（如 '4.pt'），但 save_model 存的是\n"
        "        #      'epoch_4.pt' —— 所以 final.pt **天生就是断链**。\n"
        "        #      官方 run.sh 的 stage 5 用 `decode_checkpoint=$dir/final.pt`\n"
        "        #      去解码，拿到断链会直接失败。目标应带 'epoch_' 前缀。\n"
        "        #\n"
        "        #   顺带：Windows 上建符号链接需要开发者模式/管理员权限，\n"
        "        #   失败时退化成复制，保证 final.pt 一定指向真实权重。\n"
        "        #\n"
        "        #   另外：删除遗留的 final.pt 也可能失败（实测在「断链」这种\n"
        "        #   特殊状态下 Windows 会报 WinError 5 拒绝访问）——\n"
        "        #   所以先试 os.remove，不行就用 os.rename 挪走\n"
        "        #   （重命名只需要目录的写权限，通常能成功）。\n"
        "        if os.path.lexists(final_model_path):\n"
        "            try:\n"
        "                os.remove(final_model_path)\n"
        "            except OSError:\n"
        "                stale = final_model_path + '.stale'\n"
        "                if os.path.lexists(stale):\n"
        "                    try:\n"
        "                        os.remove(stale)\n"
        "                    except OSError:\n"
        "                        stale = '{}.stale.{}'.format(final_model_path,\n"
        "                                                     os.getpid())\n"
        "                os.rename(final_model_path, stale)\n"
        "        target = 'epoch_{}.pt'.format(final_epoch)\n"
        "        try:\n"
        "            os.symlink(target, final_model_path)\n"
        "        except OSError:\n"
        "            import shutil\n"
        "            shutil.copy2(os.path.join(args.model_dir, target),\n"
        "                         final_model_path)\n"
        "        writer.close()\n",
        "★ final.pt 是断链（目标名少 'epoch_' 前缀）+ 清理用 exists 而非 lexists",
    ),
    (
        "wenet/utils/train_utils.py",
        "    if int(os.environ.get('RANK', 0)) == 0:\n"
        "        saved_config_path = os.path.join(args.model_dir, 'train.yaml')\n"
        "        with open(saved_config_path, 'w') as fout:\n",
        "    if int(os.environ.get('RANK', 0)) == 0:\n"
        "        saved_config_path = os.path.join(args.model_dir, 'train.yaml')\n"
        "        # ★ 补丁(patch_wenet.py)：官方 run.sh 在启动 torchrun 之前有\n"
        "        #   `mkdir -p $dir`，手动跑极容易漏 → 这里自己保证目录存在；\n"
        "        #   顺便显式指定编码（Windows 默认 GBK）。\n"
        "        os.makedirs(os.path.dirname(saved_config_path) or '.', exist_ok=True)\n"
        "        with open(saved_config_path, 'w', encoding='utf-8') as fout:\n",
        "model_dir 不存在时自动创建（补齐官方 run.sh 的 mkdir -p）",
    ),
    (
        "wenet/utils/train_utils.py",
        "    if args.train_engine == \"torch_ddp\":  # native pytorch ddp\n"
        "        device = torch.device(args.device)\n"
        "        model.to(device)\n"
        "        model = torch.nn.parallel.DistributedDataParallel(\n"
        "            model, find_unused_parameters=not grad_ckpt)\n",
        "    if args.train_engine == \"torch_ddp\":  # native pytorch ddp\n"
        "        device = torch.device(args.device)\n"
        "        model.to(device)\n"
        "        # ★ 补丁(patch_wenet.py)：world_size == 1 时**不包 DDP**。\n"
        "        #   PyTorch 的 Windows gloo 是 CPU-only 构建（is_nccl_available()=False），\n"
        "        #   而 DDP 反向时要把 **CUDA** 梯度 all_reduce —— gloo 拿不到 CUDA tensor\n"
        "        #   → **直接段错误**（0xC0000005，Python 层没有任何异常）。\n"
        "        #   实测：裸模型 CUDA 前向+反向正常；同样代码包进 DDP 后 backward 立刻崩。\n"
        "        #   单进程本来没有任何梯度需要规约，DDP 纯属额外开销，跳过完全等价。\n"
        "        #   安全性：WeNet 训练循环与 save_model 都不依赖 .module（已全仓库确认），\n"
        "        #   且 save_model 走 state_dict()，两种包装都兼容。\n"
        "        if world_size > 1:\n"
        "            model = torch.nn.parallel.DistributedDataParallel(\n"
        "                model, find_unused_parameters=not grad_ckpt)\n",
        "★ 单卡（world_size==1）不包 DDP —— 修 GPU 训练段错误（gloo 不支持 CUDA）",
    ),
    (
        "wenet/utils/train_utils.py",
        "    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser\n"
        "    parser = deepspeed.add_config_arguments(parser)\n",
        "    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser\n"
        "    if DEEPSPEED_AVAILABLE:  # ★ 补丁(patch_wenet.py)\n"
        "        parser = deepspeed.add_config_arguments(parser)\n",
        "未装 deepspeed 时不注册它的命令行参数",
    ),
    (
        "wenet/utils/train_utils.py",
        '    elif args.train_engine == "deepspeed":\n'
        "        deepspeed.init_distributed(dist_backend=args.dist_backend)\n",
        '    elif args.train_engine == "deepspeed":\n'
        '        assert DEEPSPEED_AVAILABLE, "train_engine=deepspeed 需要先安装 deepspeed"\n'
        "        deepspeed.init_distributed(dist_backend=args.dist_backend)\n",
        "选 deepspeed 引擎时才检查依赖（给出清晰报错）",
    ),
    (
        "wenet/utils/common.py",
        "from whisper.tokenizer import LANGUAGES as WhiserLanguages\n"
        "\n"
        "WHISPER_LANGS = tuple(WhiserLanguages.keys())\n",
        "# ★ 补丁(patch_wenet.py)：whisper 改为可选导入。\n"
        "#   common.py 被**整个模型栈**依赖（encoder.py -> common.py），\n"
        "#   而 whisper 的语言表只用于多语言 token；本项目单语中文用不到，\n"
        "#   且 openai-whisper 会与 numpy 2.x 冲突。\n"
        "try:\n"
        "    from whisper.tokenizer import LANGUAGES as WhiserLanguages\n"
        "    WHISPER_LANGS = tuple(WhiserLanguages.keys())\n"
        "except ImportError:  # pragma: no cover\n"
        "    WhiserLanguages = None\n"
        "    WHISPER_LANGS = ()\n",
        "whisper 语言表改为可选导入（common.py 被整个模型栈依赖）",
    ),
    (
        "wenet/utils/init_tokenizer.py",
        "from wenet.text.sentencepiece_tokenizer import SentencepieceTokenizer\n"
        "from wenet.text.whisper_tokenizer import WhisperTokenizer\n",
        "from wenet.text.sentencepiece_tokenizer import SentencepieceTokenizer\n"
        "# ★ 补丁(patch_wenet.py)：whisper tokenizer 改为惰性导入。\n"
        "#   它的模块里会 `import whisper`（openai-whisper，与 numpy 2.x 冲突），\n"
        "#   而我们的 tokenizer 是 char，用不到。\n",
        "whisper_tokenizer 顶层导入移除",
    ),
    (
        "wenet/utils/init_tokenizer.py",
        '    if tokenizer_type == "whisper":\n'
        "        tokenizer = WhisperTokenizer(\n",
        '    if tokenizer_type == "whisper":\n'
        "        from wenet.text.whisper_tokenizer import WhisperTokenizer  # ★ 补丁\n"
        "        tokenizer = WhisperTokenizer(\n",
        "用到 whisper 时才导入",
    ),
]


def say(*a, **kw) -> None:
    print(*a, file=sys.stderr, **kw)


# ============================================================================
# 模式化补丁：同一处改动要应用到多个文件时用这个，避免重复抄。
# (旧文本, 新文本, [文件列表], 说明)
# ============================================================================
PATTERN_PATCHES: list[tuple[str, str, list[str], str]] = [
    (
        "with open(args.config, 'r') as fin:",
        "with open(args.config, 'r', encoding='utf-8') as fin:  "
        "# ★ 补丁(patch_wenet.py)",
        [
            "wenet/bin/train.py",
            "wenet/bin/recognize.py",
            "wenet/bin/export_jit.py",
            "wenet/bin/alignment.py",
            "wenet/bin/export_ipex.py",
            "wenet/bin/export_onnx_bpu.py",
            "wenet/bin/export_onnx_cpu.py",
            "wenet/bin/recognize_onnx_gpu.py",
        ],
        "yaml 配置读取未指定编码 → 中文注释在 GBK 环境下 UnicodeDecodeError",
    ),
    (
        'with open(args.config, "r") as fin:',
        'with open(args.config, "r", encoding="utf-8") as fin:  '
        "# ★ 补丁(patch_wenet.py)",
        ["wenet/bin/export_onnx_gpu.py"],
        "同上（这个文件用的是双引号写法）",
    ),
    (
        "with open(y, 'r') as f:",
        "with open(y, 'r', encoding='utf-8') as f:  # ★ 补丁(patch_wenet.py)",
        ["wenet/bin/average_model.py"],
        "读 {tag}.yaml 时未指定编码",
    ),
    (
        "with open(info_path, 'r') as fin:",
        "with open(info_path, 'r', encoding='utf-8') as fin:  "
        "# ★ 补丁(patch_wenet.py)",
        ["wenet/utils/checkpoint.py"],
        "读 checkpoint 的 yaml 元信息时未指定编码",
    ),
    (
        "with open(info_path, 'w') as fout:",
        "with open(info_path, 'w', encoding='utf-8') as fout:  "
        "# ★ 补丁(patch_wenet.py)",
        ["wenet/utils/checkpoint.py"],
        "写 checkpoint 的 yaml 元信息时未指定编码（Windows 上会写成 GBK）",
    ),
    (
        "with open(args.deepspeed_config, 'r') as fin:",
        "with open(args.deepspeed_config, 'r', encoding='utf-8') as fin:  "
        "# ★ 补丁(patch_wenet.py)",
        ["wenet/utils/train_utils.py"],
        "读 deepspeed 配置时未指定编码",
    ),
]


def all_patches() -> list[tuple[str, str, str, str]]:
    """把 PATCHES 与 PATTERN_PATCHES 展开成同一个列表。"""
    out = list(PATCHES)
    for old, new, files, why in PATTERN_PATCHES:
        for f in files:
            out.append((f, old, new, why))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="WeNet 兼容性补丁（torch 2.11 + 精简依赖）")
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    ap.add_argument("--revert", action="store_true", help="从 .orig 还原")
    args = ap.parse_args()

    if not WENET.is_dir():
        say(f"❌ 找不到 WeNet 源码: {WENET}")
        return 1

    patches = all_patches()

    if args.revert:
        for rel in dict.fromkeys(p for p, _, _, _ in patches):
            p = WENET / rel
            b = p.with_suffix(p.suffix + ".orig")
            if b.exists():
                shutil.copy2(b, p)
                say(f"↩️  已还原 {rel}")
            else:
                say(f"（无备份，跳过）{rel}")
        return 0

    applied = todo = failed = 0
    files_touched: set[Path] = set()
    missing_files: set[str] = set()

    for rel, old, new, why in patches:
        p = WENET / rel
        if not p.exists():
            missing_files.add(rel)
            failed += 1
            continue
        src = p.read_text(encoding="utf-8")

        if new in src:
            applied += 1
            continue
        if old not in src:
            # 已经打过「同一文件的同一处」时 new 会命中；到这里说明真的不匹配
            failed += 1
            say(f"❌ 匹配失败 {rel}  ← {why}")
            say(f"     找不到原文片段：{old[:70]}…")
            continue

        todo += 1
        if not args.check:
            backup = p.with_suffix(p.suffix + ".orig")
            if not backup.exists():
                shutil.copy2(p, backup)
            p.write_text(src.replace(old, new, 1), encoding="utf-8", newline="")
            files_touched.add(p)

    if missing_files:
        for m in sorted(missing_files):
            say(f"⚠️ 文件不存在（torch/WeNet 版本可能变了）: {m}")

    say("")
    if failed:
        say(f"❌ 有 {failed} 处失败 —— 请人工检查，别当作已修复")
        return 1
    if todo == 0:
        say(f"✅ 全部已打好（共 {len(patches)} 处），无需修改")
        return 0
    if args.check:
        say(f"⚠️ 还有 {todo} 处待打 —— 去掉 --check 执行")
        return 1
    say(f"✅ 补丁完成：新打 {todo} 处（共 {len(patches)} 处），改动文件 {len(files_touched)} 个")
    say("   备份为各自同目录的 *.py.orig（--revert 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
