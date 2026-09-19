// ============================================================================
//  P01 | 最长递增子序列（LIS）
//  题面：coding/problems/P01_LIS.md
//  编译运行：python coding/build.py coding/solutions/P01_LIS.cpp
//
//  审查后的修正版（2026-09-18）。改动点用 【修正N】 标出，共 4 处 bug + 2 处注释。
//  你原来的算法逻辑是对的（10/10 边界用例通过、n=100000 得 614 = 独立参考值），
//  修的全是"让正确答案真的输出出来"和"复杂度最坏情况"的问题。
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

// 数据类型用 long long：a[i] 会到 1e9，且涉及 a[i] 之间的比较，别用 int 省事
using ll = long long;

// 只有当 n 小于这个值才跑 O(n^2) 解法，避免大数据上白等。
// 想看「O(n^2) 在 n=100000 上真的超时」，把它改成 1 << 30 再跑。
constexpr int DP_LIMIT = 20000;

// ---------------------------------------------------------------------------
// 第 1 问：O(n^2) 动态规划
//
//   dp[i] = 以 a[i] 结尾的最长递增子序列的长度（至少为 1）
//   转移  dp[i] = max(1, max{ dp[j] + 1 | j < i 且 a[j] < a[i] })
//   答案  max{ dp[i] }（注意不是 dp[n-1]，因为最长的那条不一定以最后一个元素结尾）
//
//   【修正2】原版把 dp[i] 定义成了「整个序列」（vector<vector<ll>>），而不是「长度」。
//   后果：每次发现更优就 dp[i]=dp[j] 整段拷贝 →
//     · 空间从 O(n) 变成 O(n^2)
//     · 最坏时间从 O(n^2) 变成 O(n^3)（单调递增输入下总拷贝量 ≈ n^3/6）
//   实测：单调递增 n=8000 时原版要 33906 ms，改成长度数组后只要约 30 ms。
// ---------------------------------------------------------------------------
int lis_length_dp(const std::vector<ll>& a) {
    const std::size_t n = a.size();
    if (n == 0) {
        return 0;                                   // 【修正4】边界兜底
    }

    std::vector<int> dp(n, 1);                      // 【修正2】长度数组，不是序列数组
    int best = 1;
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {       // 【修正3】size_t，消除 sign-compare
            if (a[j] < a[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;                  // O(1) 更新，不再整段拷贝
            }
        }
        if (dp[i] > best) {
            best = dp[i];
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// 第 2 问：O(n log n)
//
//   tail[k] = 所有长度为 k+1 的递增子序列中，最小的那个结尾值
//     · tail 天然严格递增（这是能二分的前提：若存在更小的结尾值，它一定会被更早写入）
//     · tail 的长度就是 LIS 长度，但 tail 本身不是任何一条合法的 LIS
//
//   追问 A：用的是 std::lower_bound（第一个 >= x）。
//           「为什么必须是它、换成 upper_bound 会在哪种输入上算错」——留给你自己写。
//   追问 B：改成「非严格递增」（最长不下降子序列）该换哪个二分、为什么
//           ——同样留给你自己写。
//
//   【修正1】原版 return len; 恒返回 0（回溯循环把 len 减到 0 了）。
// ---------------------------------------------------------------------------
int lis_length_fast(const std::vector<ll>& a, std::vector<ll>& path) {
    const std::size_t n = a.size();
    path.clear();
    if (n == 0) {
        return 0;                                   // 【修正4】边界兜底
    }

    std::vector<ll> tail;
    // dp[i] = a[i] 被放进了 tail 的第几个位置（0 基）
    //       等价于「以 a[i] 结尾的 LIS 长度 - 1」，回溯时靠它定位
    std::vector<int> dp(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        auto it = std::lower_bound(tail.begin(), tail.end(), a[i]);
        dp[i] = static_cast<int>(it - tail.begin());
        if (it == tail.end()) {
            tail.push_back(a[i]);
        } else {
            *it = a[i];
        }
    }

    // 【修正1】先把答案长度存下来，不能再拿回溯游标当返回值
    const int ans = static_cast<int>(tail.size());
    path.resize(static_cast<std::size_t>(ans));

    // 倒序回溯：从最后一个元素往前扫，依次匹配 tail 位置 ans-1, ans-2, ...
    // 为什么倒序能保证正确：对同一个位置 cur，取「下标最大」的那个候选，它的值
    // 一定小于已选中的 cur+1 位置那个值（否则中间必然存在另一个 dp==cur 的、下标更大的元素）
    int cur = ans - 1;
    for (std::size_t k = n; k > 0 && cur >= 0; --k) {   // 【修正3】size_t，无负下标
        const std::size_t i = k - 1;
        if (dp[i] == cur) {
            path[static_cast<std::size_t>(cur)] = a[i];
            --cur;
        }
    }
    return ans;                                     // 【修正1】原来这里写的是 return len;
}

// ---------------------------------------------------------------------------
// 第 3 问：校验函数 —— 验证 seq 确实是 a 的一个「严格递增子序列」
//   两件事都要查：
//     1) seq 是 a 的子序列（顺序必须与 a 一致，元素可越过不取）—— 贪心匹配即可
//     2) seq 严格递增
// ---------------------------------------------------------------------------
bool is_valid(const std::vector<ll>& a, const std::vector<ll>& seq) {
    // 1) 子序列判定：双指针贪心。匹配到就一起前进，否则只前进 a 的下标
    std::size_t j = 0;
    for (std::size_t i = 0; i < a.size() && j < seq.size(); ++i) {
        if (a[i] == seq[j]) {
            ++j;
        }
    }
    if (j != seq.size()) {
        return false;                              // 【修正3】size_t，消除 sign-compare
    }
    // 2) 严格递增判定（注意是 <= 就失败，不是 <）
    for (std::size_t i = 1; i < seq.size(); ++i) {
        if (seq[i] <= seq[i - 1]) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 输入输出外壳（不用改）
//   stdout 只输出两行，严格符合题面格式：
//       第 1 行 = LIS 长度
//       第 2 行 = 一条 LIS（空格分隔）
//   所有诊断信息（耗时、交叉验证）走 stderr，不污染 stdout —— 这是刷题的基本纪律，
//   评测机只读 stdout，调试信息混进去会直接判 WA。
//
//   ⚠️ 注意这里的诊断文字**故意用纯 ASCII**：
//   Windows 控制台默认是 GBK 代码页，C++ 程序直接往 stdout/stderr 写 UTF-8 中文
//   会变乱码（Python 不会有这问题，因为它走的是宽字符控制台 API）。
//   想用中文诊断也行，但要自己 SetConsoleOutputCP(CP_UTF8) —— 刷题没必要折腾这个。
// ---------------------------------------------------------------------------
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n = 0;
    if (!(std::cin >> n)) {
        std::cerr << "[diag] no input\n";
        return 0;
    }
    // 【修正4】n <= 0 时下面 O(n^2) 那条路会在空 vector 上取 dp[0] → 段错误
    //   （原版实测退出码 3221225477 = 0xC0000005，而且崩溃前 stdout/stderr 都正常打印完了，
    //    只看输出会以为成功 —— 所以刷题一定要看退出码）
    if (n <= 0) {
        std::cout << "0\n\n";
        std::cerr << "[diag] n <= 0 -> empty answer\n";
        return 0;
    }

    std::vector<ll> a(static_cast<std::size_t>(n));
    for (auto& x : a) {
        std::cin >> x;
    }
    std::cerr << "[diag] n = " << n << "\n";

    // ---- O(n log n) 解法（主答案）----
    std::vector<ll> path;
    auto t1 = std::chrono::steady_clock::now();
    const int len_fast = lis_length_fast(a, path);
    auto t2 = std::chrono::steady_clock::now();
    const double ms_fast =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---- stdout：严格两行 ----
    std::cout << len_fast << "\n";
    for (std::size_t i = 0; i < path.size(); ++i) {
        std::cout << path[i] << (i + 1 == path.size() ? '\n' : ' ');
    }
    if (path.empty()) {
        std::cout << "\n";
    }

    // ---- stderr：诊断 ----
    std::cerr << "[diag] O(n log n) time  " << ms_fast << " ms\n";
    std::cerr << "[diag] path size = " << path.size() << ", expect "
              << len_fast << " -> "
              << (path.size() == static_cast<std::size_t>(len_fast) ? "OK"
                                                                   : "MISMATCH")
              << "\n";
    std::cerr << "[diag] is_valid() -> " << (is_valid(a, path) ? "PASS" : "FAIL")
              << "\n";

    // ---- 拿 O(n^2) 的结果做交叉验证 ----
    if (n <= DP_LIMIT) {
        auto t3 = std::chrono::steady_clock::now();
        const int len_dp = lis_length_dp(a);
        auto t4 = std::chrono::steady_clock::now();
        const double ms_dp =
            std::chrono::duration<double, std::milli>(t4 - t3).count();
        std::cerr << "[diag] O(n^2)   time  " << ms_dp << " ms, len = " << len_dp
                  << " -> vs fast: " << (len_dp == len_fast ? "OK" : "MISMATCH!")
                  << "\n";
    } else {
        std::cerr << "[diag] n = " << n << " > DP_LIMIT(" << DP_LIMIT
                  << "), skip O(n^2)  (raise DP_LIMIT to watch it TLE)\n";
    }
    return 0;
}
