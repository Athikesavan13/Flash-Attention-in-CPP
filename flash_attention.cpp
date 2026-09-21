// flash_attention.cpp
//
// Flash Attention — CPU reference implementation
// Demonstrates: tiling, online softmax, causal masking, multi-head attention,
// and benchmarking vs naive O(N^2) memory attention.
//
// Build:  g++ -O2 -std=c++17 flash_attention.cpp -o flash_attention
// Run:    ./flash_attention
//
// ─────────────────────────────────────────────────────────────────────────────
// WHY FLASH ATTENTION?
//
// Standard attention materializes an N×N score matrix:
//   S = Q K^T / sqrt(d)   [N×N floats → O(N^2) memory]
//   P = softmax(S)
//   O = P V
//
// For N=8192, d=64, fp32: S alone is 8192^2 * 4 bytes = 256 MB *per head*.
// That matrix must be written to HBM (GPU RAM) and read back — twice per layer.
// The bottleneck is not arithmetic (FLOPs) but memory bandwidth (IO).
//
// Flash Attention (Dao et al. 2022) keeps IO to O(N) by:
//   1. TILING: streaming K and V through SRAM in blocks, never storing S.
//   2. ONLINE SOFTMAX: maintaining a running (max, sum) so normalization is
//      exact even though we see only one tile at a time.
//
// IO complexity:
//   Naive:  O(N^2 * d) HBM reads+writes  (dominated by the score matrix)
//   Flash:  O(N * d)   HBM reads+writes  (only Q, K, V, O touch HBM)
//
// This is why Flash Attention is 2–4× faster in practice even though the
// FLOP count is identical — it is IO-bound, not compute-bound.
// ─────────────────────────────────────────────────────────────────────────────

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>
#include <chrono>
#include <string>
#include <functional>

using Matrix = std::vector<std::vector<float>>;

// ─── helpers ─────────────────────────────────────────────────────────────────

Matrix make_matrix(int rows, int cols, float init = 0.0f) {
    return Matrix(rows, std::vector<float>(cols, init));
}

Matrix random_matrix(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Matrix m = make_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i][j] = dist(rng);
    return m;
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
    return s;
}

float max_abs_diff(const Matrix& A, const Matrix& B) {
    float worst = 0.0f;
    for (size_t i = 0; i < A.size(); i++)
        for (size_t j = 0; j < A[0].size(); j++)
            worst = std::max(worst, std::abs(A[i][j] - B[i][j]));
    return worst;
}

// Returns elapsed milliseconds and stores result in `out`
double benchmark(std::function<Matrix()> fn, Matrix& out, int runs = 5) {
    double best = 1e18;
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        out = fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1) NAIVE ATTENTION  (ground truth, O(N^2) memory)
//
//    causal=true applies the autoregressive mask: position i can only attend
//    to positions j <= i, which is what every decoder-only LLM (GPT, Llama…)
//    needs.
// ─────────────────────────────────────────────────────────────────────────────

Matrix naive_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                       bool causal = false) {
    int N  = Q.size();
    int M  = K.size();
    int d  = Q[0].size();
    int dv = V[0].size();
    float scale = 1.0f / std::sqrt((float)d);
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    Matrix O = make_matrix(N, dv);

    for (int i = 0; i < N; i++) {
        std::vector<float> scores(M);
        float row_max = NEG_INF;
        for (int j = 0; j < M; j++) {
            // causal mask: future tokens get -inf score → softmax weight → 0
            scores[j] = (causal && j > i) ? NEG_INF : dot(Q[i], K[j]) * scale;
            if (scores[j] != NEG_INF) row_max = std::max(row_max, scores[j]);
        }

        float row_sum = 0.0f;
        for (int j = 0; j < M; j++) {
            scores[j] = (scores[j] == NEG_INF) ? 0.0f : std::exp(scores[j] - row_max);
            row_sum += scores[j];
        }
        for (int j = 0; j < M; j++) scores[j] /= row_sum;

        for (int j = 0; j < M; j++)
            for (int k = 0; k < dv; k++)
                O[i][k] += scores[j] * V[j][k];
    }
    return O;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) FLASH ATTENTION  (tiled + online softmax)
//
//    Per query row i, sweep K/V in tiles of width Bc:
//
//      state before tile t:  (m, l, acc)
//        m   = running max of all scores seen so far
//        l   = Σ exp(s_j - m)  for all j seen
//        acc = Σ exp(s_j - m) * V_j  for all j seen
//
//      on new tile [j0, j1):
//        s_j    = Q_i · K_j * scale          (masked to -inf if causal & j > i)
//        new_m  = max(m, max_j s_j)
//        corr   = exp(m - new_m)             (rescale old state to new baseline)
//        l      = l * corr  +  Σ exp(s_j - new_m)
//        acc    = acc * corr  +  Σ exp(s_j - new_m) * V_j
//        m      = new_m
//
//      after last tile:  O_i = acc / l
//
//    This is mathematically identical to naive softmax; correctness follows
//    from the identity:
//      exp(a - new_m) = exp(a - old_m) * exp(old_m - new_m)
// ─────────────────────────────────────────────────────────────────────────────

Matrix flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                       int Bc, bool causal = false) {
    int N  = Q.size();
    int M  = K.size();
    int d  = Q[0].size();
    int dv = V[0].size();
    float scale = 1.0f / std::sqrt((float)d);
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    Matrix O = make_matrix(N, dv);

    for (int i = 0; i < N; i++) {
        float m = NEG_INF;
        float l = 0.0f;
        std::vector<float> acc(dv, 0.0f);

        for (int j0 = 0; j0 < M; j0 += Bc) {
            int j1       = std::min(j0 + Bc, M);
            int tile_len = j1 - j0;

            // causal short-circuit: entire tile is in the future → skip it
            if (causal && j0 > i) break;

            // compute scores for this tile
            std::vector<float> s(tile_len);
            float tile_max = NEG_INF;
            for (int t = 0; t < tile_len; t++) {
                int j = j0 + t;
                s[t] = (causal && j > i) ? NEG_INF : dot(Q[i], K[j]) * scale;
                if (s[t] != NEG_INF) tile_max = std::max(tile_max, s[t]);
            }

            if (tile_max == NEG_INF) continue; // all masked, nothing to do

            // online softmax update
            float new_m  = std::max(m, tile_max);
            float corr   = std::exp(m - new_m);   // = 0 on first tile (m=-inf)
            l   *= corr;
            for (int k = 0; k < dv; k++) acc[k] *= corr;

            for (int t = 0; t < tile_len; t++) {
                if (s[t] == NEG_INF) continue;
                float p = std::exp(s[t] - new_m);
                l += p;
                for (int k = 0; k < dv; k++)
                    acc[k] += p * V[j0 + t][k];
            }
            m = new_m;
        }

        for (int k = 0; k < dv; k++) O[i][k] = acc[k] / l;
    }
    return O;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3) MULTI-HEAD FLASH ATTENTION
//
//    Real transformers split the d_model dimension into H independent heads,
//    each of width d_head = d_model / H.  Heads attend in parallel and are
//    concatenated before the output projection.
//
//    Here: Q/K/V are (N, H*d_head).  We slice out each head's sub-matrix,
//    run Flash Attention independently, then concatenate.
//
//    On a GPU you'd parallelize over heads (one thread block per head).
// ─────────────────────────────────────────────────────────────────────────────

// Slice columns [col_start, col_start+width) from a matrix
Matrix slice_cols(const Matrix& M, int col_start, int width) {
    int rows = M.size();
    Matrix out = make_matrix(rows, width);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < width; j++)
            out[i][j] = M[i][col_start + j];
    return out;
}

// Paste src into dst starting at column col_start
void paste_cols(Matrix& dst, const Matrix& src, int col_start) {
    int width = src[0].size();
    for (int i = 0; i < (int)src.size(); i++)
        for (int j = 0; j < width; j++)
            dst[i][col_start + j] = src[i][j];
}

Matrix multi_head_flash_attention(const Matrix& Q, const Matrix& K, const Matrix& V,
                                  int H, int Bc, bool causal = false) {
    int N      = Q.size();
    int d_model = Q[0].size();
    int d_head  = d_model / H;   // assumes d_model divisible by H

    Matrix O = make_matrix(N, d_model);

    for (int h = 0; h < H; h++) {
        int col = h * d_head;
        Matrix Qh = slice_cols(Q, col, d_head);
        Matrix Kh = slice_cols(K, col, d_head);
        Matrix Vh = slice_cols(V, col, d_head);

        Matrix Oh = flash_attention(Qh, Kh, Vh, Bc, causal);
        paste_cols(O, Oh, col);
    }
    return O;
}

// ─────────────────────────────────────────────────────────────────────────────
// main: correctness checks + benchmark
// ─────────────────────────────────────────────────────────────────────────────

void section(const std::string& title) {
    std::cout << "\n╔══ " << title << "\n";
}

void check(const std::string& label, const Matrix& ref, const Matrix& got) {
    float diff = max_abs_diff(ref, got);
    std::cout << "  " << std::left << std::setw(42) << label
              << " max_abs_diff = " << std::scientific << std::setprecision(2) << diff
              << (diff < 1e-5f ? "  ✓" : "  ✗ FAIL") << "\n";
}

int main() {
    std::mt19937 rng(42);

    // ── 1. Correctness: basic attention ──────────────────────────────────────
    section("Correctness — basic Flash Attention vs naive");
    {
        int N = 17, M = 23, d = 8;
        Matrix Q = random_matrix(N, d, rng);
        Matrix K = random_matrix(M, d, rng);
        Matrix V = random_matrix(M, d, rng);
        Matrix ref = naive_attention(Q, K, V);

        for (int Bc : {1, 4, 6, 16, 64})
            check("tile Bc=" + std::to_string(Bc),
                  ref, flash_attention(Q, K, V, Bc));
    }

    // ── 2. Correctness: causal mask ───────────────────────────────────────────
    section("Correctness — causal masking");
    {
        int N = 20, d = 16;
        Matrix Q = random_matrix(N, d, rng);
        Matrix K = random_matrix(N, d, rng);
        Matrix V = random_matrix(N, d, rng);
        Matrix ref = naive_attention(Q, K, V, /*causal=*/true);

        for (int Bc : {4, 7, 32})
            check("causal, tile Bc=" + std::to_string(Bc),
                  ref, flash_attention(Q, K, V, Bc, /*causal=*/true));
    }

    // ── 3. Correctness: multi-head ────────────────────────────────────────────
    section("Correctness — multi-head Flash Attention vs naive (head-by-head)");
    {
        int N = 16, H = 4, d_head = 8;
        int d_model = H * d_head;
        Matrix Q = random_matrix(N, d_model, rng);
        Matrix K = random_matrix(N, d_model, rng);
        Matrix V = random_matrix(N, d_model, rng);

        // Reference: run naive attention per head and concatenate
        Matrix ref = make_matrix(N, d_model);
        for (int h = 0; h < H; h++) {
            int col = h * d_head;
            Matrix ref_h = naive_attention(slice_cols(Q, col, d_head),
                                           slice_cols(K, col, d_head),
                                           slice_cols(V, col, d_head));
            paste_cols(ref, ref_h, col);
        }

        for (int Bc : {4, 16}) {
            Matrix got = multi_head_flash_attention(Q, K, V, H, Bc);
            check("H=" + std::to_string(H) + " heads, Bc=" + std::to_string(Bc),
                  ref, got);
        }

        // Also test with causal mask
        Matrix ref_causal = make_matrix(N, d_model);
        for (int h = 0; h < H; h++) {
            int col = h * d_head;
            Matrix ref_h = naive_attention(slice_cols(Q, col, d_head),
                                           slice_cols(K, col, d_head),
                                           slice_cols(V, col, d_head), true);
            paste_cols(ref_causal, ref_h, col);
        }
        Matrix got_causal = multi_head_flash_attention(Q, K, V, H, 4, true);
        check("H=" + std::to_string(H) + " heads, causal, Bc=4", ref_causal, got_causal);
    }

    // ── 4. Benchmark: Flash vs Naive, scaling sequence length ─────────────────
    section("Benchmark — wall-clock time vs sequence length  (d=64, H=1, Bc=64)");
    std::cout << "  " << std::setw(8) << "N"
              << std::setw(14) << "Naive (ms)"
              << std::setw(16) << "Flash (ms)"
              << std::setw(14) << "Speedup"
              << std::setw(22) << "Naive mem (MB, est.)"
              << "\n";

    int d = 64, Bc = 64;
    for (int N : {128, 256, 512, 1024, 2048}) {
        Matrix Q = random_matrix(N, d, rng);
        Matrix K = random_matrix(N, d, rng);
        Matrix V = random_matrix(N, d, rng);

        Matrix out_n, out_f;
        double t_naive = benchmark([&]{ return naive_attention(Q, K, V); },     out_n);
        double t_flash = benchmark([&]{ return flash_attention(Q, K, V, Bc); }, out_f);

        // Naive peak memory: N*N score matrix
        double naive_mb = (double)N * N * sizeof(float) / (1024 * 1024);

        std::cout << "  " << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << t_naive
                  << std::setw(16) << t_flash
                  << std::setw(13) << std::setprecision(2) << t_naive / t_flash << "x"
                  << std::setw(18) << std::setprecision(2) << naive_mb
                  << "\n";
    }

    std::cout << "\n  Note: on CPU the speedup comes from cache locality, not HBM\n"
                 "  bandwidth savings.  On a real GPU the gap is 2-4x at N=4096+.\n";

    return 0;
}
