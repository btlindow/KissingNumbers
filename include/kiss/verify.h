// T1.4 — exact verifiers for a candidate independent set S ⊂ C and the
// reference tightness computation (docs/design.md §2.3 / §5.1, README §2, §5).
//
// Everything here is integer arithmetic on the √8-scaled coordinates: a set S
// certifies K(25) ≥ 196560 + |S| iff every row has squared norm 32, is a Leech
// minimal vector, the rows are distinct and every off-diagonal Gram entry is
// ≤ 8 (i.e. no pair at 60°, ⟨x,y⟩ = 16).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "kiss/leech.h"
#include "kiss/types.h"

namespace kiss {

struct VerifyResult {
  bool ok;
  std::size_t size;      // number of rows in S (also reported on failure)
  std::string message;   // "ok" or a description of the first failure
};

// Certificate check (README §2). Checks run in the order
//   1. every row has squared norm 32,
//   2. every row is in C (Leech::index_of ≥ 0),
//   3. all rows are distinct,
//   4. every off-diagonal Gram entry ⟨S[i],S[j]⟩, i < j, is ≤ 8,
// and stop at the first failing check. The message names the first offending
// row (or pair, with its inner product) and how many offenders that check has.
// Rows are named "row k" (0-based index into S). The overload taking `lines`
// names them "line L (row k)" with L = lines[k], the 1-based file line as
// returned by set_line_numbers(). An empty S is valid (ok, size 0).
VerifyResult verify_independent(const Leech& L, const std::vector<Vec>& S);
VerifyResult verify_independent(const Leech& L, const std::vector<Vec>& S,
                                const std::vector<long>& lines);

// Reference tightness (the CPU oracle for the GPU kernel of T1.5):
//   out[v] = #{ s ∈ S_idx : ⟨C[v], C[s]⟩ == 16 }   for every vertex v in 0..N-1.
// out[v] == 0 and v ∉ S means v can be added to S ("free" vertex); an
// independent set is maximal iff no vertex outside S is free. OpenMP over v.
// Throws std::out_of_range if an index is ≥ N, std::length_error if
// |S_idx| > 65535 (would overflow uint16).
std::vector<uint16_t> tightness_cpu(const Leech& L, const std::vector<uint32_t>& S_idx);

// ---- additions beyond §2.3 (documented in docs/reports/T1.4.md) ------------

// Canonical indices of the rows of S. Throws std::runtime_error naming the
// first row that is not in C.
std::vector<uint32_t> set_indices(const Leech& L, const std::vector<Vec>& S);

// True iff S is closed under negation (−x ∈ S for every x ∈ S).
bool is_antipodal(const std::vector<Vec>& S);

// Histogram of ⟨S[i],S[j]⟩ over unordered pairs i < j by inner-product class
// (index 0..6 ↔ {-32,-16,-8,0,8,16,32}, see ip_class); [7] counts values
// outside the seven classes.
std::array<long, 8> gram_histogram(const std::vector<Vec>& S);

// 1-based file line number of each data row of a set file, in the order
// read_set() returns them (blank and comment-only lines are skipped exactly as
// read_set does). Throws std::runtime_error if the file cannot be opened.
std::vector<long> set_line_numbers(const std::filesystem::path& path);

}  // namespace kiss
