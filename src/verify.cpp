// T1.4 — verifiers and reference tightness. See include/kiss/verify.h.
#include "kiss/verify.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

namespace kiss {

namespace {

std::string vec_str(const Vec& v) {
  std::string s = "(";
  for (std::size_t k = 0; k < static_cast<std::size_t>(DIM); ++k) {
    if (k) s += ' ';
    s += std::to_string(static_cast<int>(v[k]));
  }
  return s + ")";
}

}  // namespace

VerifyResult verify_independent(const Leech& L, const std::vector<Vec>& S) {
  return verify_independent(L, S, std::vector<long>{});
}

VerifyResult verify_independent(const Leech& L, const std::vector<Vec>& S,
                                const std::vector<long>& lines) {
  const auto name = [&](std::size_t k) {
    std::string s = "row " + std::to_string(k);
    if (k < lines.size()) s = "line " + std::to_string(lines[k]) + " (" + s + ")";
    return s;
  };
  const std::size_t n = S.size();
  VerifyResult r{false, n, ""};

  // 1. Norms.
  {
    std::size_t bad = 0, first = 0;
    int first_norm = 0;
    for (std::size_t k = 0; k < n; ++k) {
      const int nn = norm2(S[k]);
      if (nn != 32) {
        if (bad == 0) {
          first = k;
          first_norm = nn;
        }
        ++bad;
      }
    }
    if (bad) {
      r.message = "norm: " + name(first) + " has squared norm " + std::to_string(first_norm) +
                  " != 32: " + vec_str(S[first]) + " [" + std::to_string(bad) +
                  " offending row(s)]";
      return r;
    }
  }

  // 2. Membership in C.
  std::vector<uint32_t> idx(n);
  {
    std::size_t bad = 0, first = 0;
    for (std::size_t k = 0; k < n; ++k) {
      const int32_t i = L.index_of(S[k]);
      if (i < 0) {
        if (bad == 0) first = k;
        ++bad;
      } else {
        idx[k] = static_cast<uint32_t>(i);
      }
    }
    if (bad) {
      r.message = "membership: " + name(first) + " is not a Leech minimal vector: " +
                  vec_str(S[first]) + " [" + std::to_string(bad) + " offending row(s)]";
      return r;
    }
  }

  // 3. Distinctness (via the canonical indices; index_of is injective on C).
  {
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
    std::size_t bad = 0, first_a = 0, first_b = 0;
    for (std::size_t k = 1; k < n; ++k) {
      if (idx[order[k]] == idx[order[k - 1]]) {
        if (bad == 0) {
          first_a = std::min(order[k - 1], order[k]);
          first_b = std::max(order[k - 1], order[k]);
        }
        ++bad;
      }
    }
    if (bad) {
      r.message = "distinct: " + name(first_a) + " and " + name(first_b) +
                  " are the same vector: " + vec_str(S[first_a]) + " [" + std::to_string(bad) +
                  " duplicate row(s)]";
      return r;
    }
  }

  // 4. Gram off-diagonal ≤ 8 (no pair at 60°).
  {
    std::size_t bad = 0, first_i = 0, first_j = 0;
    int first_ip = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const int ip = dot(S[i], S[j]);
        if (ip > 8) {
          if (bad == 0) {
            first_i = i;
            first_j = j;
            first_ip = ip;
          }
          ++bad;
        }
      }
    }
    if (bad) {
      r.message = "gram: " + name(first_i) + " and " + name(first_j) + " have inner product " +
                  std::to_string(first_ip) + " > 8" + (first_ip == 16 ? " (60 degrees)" : "") +
                  ": " + vec_str(S[first_i]) + " . " + vec_str(S[first_j]) + " [" +
                  std::to_string(bad) + " offending pair(s)]";
      return r;
    }
  }

  r.ok = true;
  r.message = "ok";
  return r;
}

std::vector<uint16_t> tightness_cpu(const Leech& L, const std::vector<uint32_t>& S_idx) {
  const std::size_t n = L.C.size();
  if (S_idx.size() > 65535)
    throw std::length_error("tightness_cpu: |S| = " + std::to_string(S_idx.size()) + " > 65535");
  // Gather the members once (contiguous, cache-friendly) and bounds-check.
  std::vector<Vec> Sv;
  Sv.reserve(S_idx.size());
  for (uint32_t s : S_idx) {
    if (s >= n) throw std::out_of_range("tightness_cpu: index " + std::to_string(s) + " >= N");
    Sv.push_back(L.C[s]);
  }
  std::vector<uint16_t> out(n, 0);
  const std::size_t m = Sv.size();
  const long nn = static_cast<long>(n);
#pragma omp parallel for schedule(static)
  for (long v = 0; v < nn; ++v) {
    const Vec& x = L.C[static_cast<std::size_t>(v)];
    unsigned cnt = 0;
    for (std::size_t k = 0; k < m; ++k) cnt += (dot(x, Sv[k]) == 16) ? 1u : 0u;
    out[static_cast<std::size_t>(v)] = static_cast<uint16_t>(cnt);
  }
  return out;
}

std::vector<uint32_t> set_indices(const Leech& L, const std::vector<Vec>& S) {
  std::vector<uint32_t> idx;
  idx.reserve(S.size());
  for (std::size_t k = 0; k < S.size(); ++k) {
    const int32_t i = L.index_of(S[k]);
    if (i < 0)
      throw std::runtime_error("set_indices: row " + std::to_string(k) +
                               " is not a Leech minimal vector: " + vec_str(S[k]));
    idx.push_back(static_cast<uint32_t>(i));
  }
  return idx;
}

bool is_antipodal(const std::vector<Vec>& S) {
  const std::set<Vec> members(S.begin(), S.end());
  for (const Vec& v : S)
    if (members.count(negate(v)) == 0) return false;
  return true;
}

std::array<long, 8> gram_histogram(const std::vector<Vec>& S) {
  std::array<long, 8> h{};
  for (std::size_t i = 0; i < S.size(); ++i) {
    for (std::size_t j = i + 1; j < S.size(); ++j) {
      const int c = ip_class(dot(S[i], S[j]));
      ++h[c < 0 ? 7 : static_cast<std::size_t>(c)];
    }
  }
  return h;
}

std::vector<long> set_line_numbers(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("set_line_numbers: cannot open " + path.string());
  std::vector<long> out;
  std::string line;
  long lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    // Same rule as read_set: a line is a data row iff something other than
    // ' ', '\t', '\r' remains after stripping the comment.
    const bool has_data = line.find_first_not_of(" \t\r") != std::string::npos;
    if (has_data) out.push_back(lineno);
  }
  return out;
}

}  // namespace kiss
