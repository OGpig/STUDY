# 算法刷题区

> 面试手撕 + 底层功底的训练场。**代码自己写，AI 只负责出题、批改、对拍。**

## 目录

```
coding/
├── README.md          ← 本文件：编译方法 + 已做题索引
├── build.py           ← 一键编译运行脚本（自动配工具链）
├── problems/          ← 题面
├── solutions/         ← 你的代码
├── tests/             ← 测试输入（*.in）
└── build/             ← 编译产物（不入库）
```

---

## 怎么编译 / 怎么跑

**在 VSCode 里**：打开任意 `.cpp`，按 `Ctrl+Shift+B`（编译并运行）。
推荐装 `ms-vscode.cpptools` 扩展（已写进 `.vscode/extensions.json`）。

**在终端里**：

```powershell
# 编译 + 运行（自动喂 tests/<同名>_1.in）
F:\embedded\prepare\.venv\Scripts\python.exe F:\embedded\prepare\coding\build.py `
  F:\embedded\prepare\coding\solutions\P01_LIS.cpp

# 只编译不运行
... build.py coding\solutions\P01_LIS.cpp --no-run

# 换编译器（默认 gcc）
... build.py coding\solutions\P01_LIS.cpp --cc msvc

# 指定输入文件 / 手动敲输入 / 调试版
... build.py coding\solutions\P01_LIS.cpp --in coding\tests\P01_LIS_big.in
... build.py coding\solutions\P01_LIS.cpp --manual
... build.py coding\solutions\P01_LIS.cpp --dbg
```

脚本会自动探测并配好编译器环境，**不需要 Developer Command Prompt、不经过 cmd.exe**。

---

## 本机两套编译器

| 名 | 版本 | 位置 | 什么时候用 |
|---|---|---|---|
| **`gcc`（默认）** | MinGW-W64 **g++ 16.2.0** | `F:\data\toolchains\mingw64\bin` | 日常刷题。报错信息友好，最接近 Linux/评测机环境 |
| `msvc` | MSVC **14.51.36231** + SDK 10.0.26100.0 | VS 18 BuildTools | 测性能 / 写 AVX intrinsics / 需要 `dumpbin`、`cl /Qvec-report` 时 |

`gdb 17.2` 也在 `mingw64\bin` 下，可以单步调试（VSCode 里 F5 走 `launch.json` 的 cppdbg 配置）。

> 想在自己终端里直接用 `g++`？把 `F:\data\toolchains\mingw64\bin` 加到系统 PATH 即可。
> 不加重启也行——用上面的 `build.py` 就已经够了。

---

## 已做题索引

| # | 题目 | 考点 | 状态 | 题面 | 代码 |
|---|---|---|---|---|---|
| P01 | 最长递增子序列（LIS） | DP / 二分 / 贪心 / **路径还原** | ✅ 通过（12/12 用例 + 零警告） | [题面](problems/P01_LIS.md) | `solutions/P01_LIS.cpp` |

状态标记：⬜ 待做 ｜ 🟡 在做 ｜ ✅ 通过（附耗时） ｜ ❌ 有问题

---

# P01 审查记录（2026-09-18）

**一句话结论**：**算法逻辑 100% 正确，但有 2 个 bug 让它拿不到分** —— 一个是"stdout 第一行永远是 0"，
另一个是"最坏情况复杂度从 O(n²) 掉到 O(n³)，n=8000 就要 34 秒"。

## ✅ 先说对的部分（这些是真本事）

1. **两个二分版本都对** —— `lower_bound` 用得正确，正好命中追问 A
2. **还原路径的思路对**：用 `dp[i]` 记"这个元素被放进 `tail` 的第几个位置"，再**倒序**扫、
   遇到 `dp[i] == len-1` 就取到 `path` 里 —— 这是标准做法，你独立写对了
3. **`is_valid()` 写得对**：贪心匹配子序列 + 检查严格递增，两个条件都覆盖，
   空序列的退化情形也自然正确
4. **实测证据**：
   - **10/10 边界用例通过**（对比独立 Python 参考实现）：n=1、单调递增/递减、全相同、
     含负数跨 0、含 ±1e9 极值、重复头、两段递增、n=3000 随机
   - **n=100000 大数据：输出 614，与独立算出的真实 LIS 长度完全一致**，`is_valid() PASS`

## ❌ Bug 1（致命）：`return len;` —— 第一行永远输出 0

```cpp
int len=tail.size();          // L59
path.resize(len);
for(int i=a.size()-1;i>=0;i--){
    if(dp[i]==len-1){ path[len-1]=a[i]; len--; }   // 每取一个就 len--
}
return len;                   // ❌ L79：len 已经被减到 0 了
```

回溯循环结束时 `len == 0`，所以 `len_fast` **恒返回 0**。stdout 第一行永远是 `0`，
而第二行的序列是对的（`is_valid` 也是 PASS）。

**消融验证**：只把 `return len;` 换成 `return ans;`（先在 L59 存下原始长度），其余一行不动 →
两个样例立刻全绿（`path size = 4, expect 4 -> OK`、`is_valid() PASS`、`vs fast: OK`）。

## ❌ Bug 2（严重）：`dp[i]` 存了整段序列 → 空间 O(n²)、最坏时间 O(n³)

`std::vector<std::vector<ll>> dp(a.size());`  ← L28

`dp[i]` 本该是**一个整数**（以 `a[i]` 结尾的 LIS 长度），你存成了**整个序列**。后果：

**（a）每次"发现更优"要整段拷贝**（L33 `dp[i]=dp[j];`）。数一下：单调递增输入下，
对每个 `k`，`dp[k]` 会被反复覆盖 `k` 次、长度依次增长 →
**总拷贝元素数 = Σ_k k²/2 ≈ n³/6**。

**（b）实测（单调递增输入，DP 内部计时）**

| n | DP 耗时 | 相邻倍率 |
|---|---|---|
| 1000 | 20.1 ms | — |
| 2000 | 190.8 ms | 9.5× |
| 4000 | 3874 ms | 20.3× |
| 8000 | **33906 ms** | 8.8× |

**规模 8 倍，耗时 1685 倍；而 O(n²) 只该涨 64 倍。**

对照：**随机**输入下 n=4000 只要 **19.6 ms**（实测增长指数 ≈ 2.0）。
也就是说这个 DP 在被卡的时候（有序/近似有序输入）比正常情况慢 **200 倍以上**，
而且 **n=8000 就 34 秒** —— 远早于题面里说的"n=100000 才超时"。

**（c）空间**：Σ|dp[i]| 最坏 = n(n+1)/2 → n=8000 时 256 MB、n=20000 时 1.6 GB。

**修法**：`dp` 改成 `std::vector<int>`，转移写 `dp[i] = std::max(dp[i], dp[j] + 1)`；
要还原序列就再开一个 `pre[i]` 记前驱。

> 这正是题面 L22 那条注释（"把你的 `dp[i]` 定义写在这里"）存在的理由 ——
> **定义选错，后面全歪**。你把这条注释跳过没填，也就跳过了这个思考步骤。

## ⚠️ Bug 3：6 个 `-Wsign-compare` 警告（题面要求零警告）

```
L29  for(int i=0;i< a.size();i++)                   int vs size_t
L61  for(int i=0;i<a.size();i++)                    int vs size_t
L93  for(int i=0;i<a.size() && j<seq.size();i++)    ×2 处
L98  if(j==seq.size())
L99  for(int i=1;i<seq.size();i++)
```

修法：循环变量改 `std::size_t`，或统一写 `(int)a.size()`。

## ⚠️ Bug 4：n=0 直接段错误（题面不要求，但值得知道）

输入 `n=0` → 退出码 **3221225477 = 0xC0000005（ACCESS_VIOLATION）**。
原因：`lis_length_dp` 里 `ind` 初始为 0、循环不执行，最后 `return dp[ind].size()`
在空 vector 上 `dp[0]` 越界。

**阴险的地方：崩溃前 stdout / stderr 都正常打印完了**，只看输出会以为成功。
—— **这就是为什么刷题必须看退出码。** 评测机上这种直接判 RE。

## ⚠️ 未完成项：注释 TODO 全空

题面第六节把下面的列为验收标准，一条都没写：

| 行 | 要写 |
|---|---|
| L22 | `dp[i] = ...` ← **最关键**，就是 Bug 2 的根因 |
| L47 | `tail[k] = ...` |
| L50 | 追问 A：用的 `lower_bound` 还是 `upper_bound`，**为什么** |
| L53 | 追问 B：改成非严格递增该换哪个，**为什么** |

## 修复清单

| 行 | 改成 |
|---|---|
| L22 | 补上 `dp[i]` 的定义；L28 的 `vector<vector<ll>>` 换成 `vector<int>` |
| L32–L36 | `if (a[j] < a[i]) dp[i] = std::max(dp[i], dp[j] + 1);` 并记 `pre[i] = j` |
| L59 | 先存下 `const int ans = static_cast<int>(tail.size());`，用它当回溯游标 |
| L79 | `return ans;` |
| L29 / 61 / 93 / 98 / 99 | 消除 sign-compare |
| L47 / 50 / 53 | 补注释 |

## 验收目标

- [ ] stdout 第一行等于正确长度
- [ ] `-Wall -Wextra` 零警告
- [ ] **单调递增 n=8000 时 DP 从 33906 ms 降到 200 ms 以内**（这是 O(n³)→O(n²) 的实证）
- [ ] 4 条注释补齐，能口头答出追问 A/B

---

## P01 修正结果（AI 已应用，2026-09-18）

4 处 bug 全修，改动点用 `【修正N】` 标在源码里。全部验收通过：

| 项 | 修正前 | 修正后 |
|---|---|---|
| stdout 第 1 行 | 恒为 `0` | **正确**（样例 `4` / `1`；n=100000 → `614` = 独立参考值） |
| `-Wall -Wextra` 警告数 | 6 | **0** |
| n=0 | **段错误**（rc 3221225477 = 0xC0000005） | rc 0，正常输出空答案 |
| 边界用例 | — | **12/12 通过**（含 n=1、单调增减、全相同、±1e9、n=3000 随机、n=0） |
| n=100000 `O(n log n)` | 算法对但输出 0 | **2.9 ms**，序列 614 个，`is_valid PASS` |

**O(n³) → O(n²) 的实证**（单调递增输入，DP 内部计时）：

| n | 修正前 | 修正后 | 改善 |
|---|---|---|---|
| 1000 | 20.1 ms | 0.1 ms | 134× |
| 2000 | 190.8 ms | 0.6 ms | 323× |
| 4000 | 3874 ms | 2.3 ms | 1679× |
| 8000 | **33906 ms** | **9.5 ms** | **3582×** |

顺带修了 `coding/build.py` 的一个设计缺陷：脚本自己的横幅原本打在 stdout 上，
会污染被测程序的输出（想 `> out.txt` 对比答案就不行）。现在**脚本信息全部走 stderr、
被测程序 stdout 用原始字节透传**，`build.py xxx.cpp > out.txt` 拿到的就是干净的答案。

### 还剩两件事是「你的」

代码和注释里的技术性内容都补齐了，**只有这两处"为什么"我故意留空**——
它们是这题真正的面试考点，写上去就没意义了：

- 追问 A：用的是 `lower_bound`（第一个 `>= x`）。**为什么必须是它？换成 `upper_bound`
  会在哪种输入上算错？**
- 追问 B：改成「非严格递增」（最长不下降子序列）该换成哪个二分，**为什么**？

想通了写到源码里对应位置，或者口头跟我说，我帮你对。


### D1 热身记录（不是本目录的题，记一下）

| 题 | 考点 | 结果 |
|---|---|---|
| `lower_bound` / `upper_bound` | 二分边界 | ✅ 已手写并造了 5 组边界用例 |
| CTC 前向 α 递推 | DP + log 域 | ✅ 与 `torch.nn.CTCLoss` 逐样本一致（max\|Δ\| 9e-05） |
| 手写 Fbank | 传统特征 | ✅ 与 `torchaudio` bit-exact |

> P01 的正解就是 `lower_bound` 的实战用法，和 D1 热身直接接上。

---

## 刷题纪律（自己定的，别破例）

1. **先写对，再写快。** 每道题都先交一版暴力/朴素解拿到正确答案，再优化。
   没有「正确答案」当锚点，优化容易越改越错。
2. **自己造边界用例。** 题面给的样例只是最低要求。空输入、n=1、全相同、极值、负数——自己补。
3. **写完必须报复杂度**，并且**用实测数据佐证**（大数据跑一次，看是不是真的更快）。
   「我用了二分所以是 O(n log n)」这句话，没有实测数据支撑就不算数。
4. **诊断信息走 stderr，stdout 只放答案。** 评测机只读 stdout，调试信息混进去直接判 WA。
5. **不看题解超过 30 分钟没思路**，就先写暴力解，再回头看提示。硬耗是低效的。
6. 每道题的**面试追问**要口头过一遍（题面第七节），这是刷题真正的收益。
