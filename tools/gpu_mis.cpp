// T3.2c — host driver of the GPU MIS engine (docs/design.md T3.2c, §5.5).
//
//   gpu_mis <config.json> [--set key=value]... [--resume] [--max-launches N]
//           [--minutes M] [--selfcheck [N]] [--run-name NAME] [--dry-run]
//
// Structure of a run (see docs/runbook_gpu_mis.md):
//
//   * Two engine states can coexist: one antipodal (moves apply to {v, neg v})
//     and one plain, with B split by `antipodal_fraction`. Each has its own
//     LSState/LSSearch, output slots and flag.
//   * The run advances in *launches* of K ILS iterations per chain
//     (kiss::cuda::lsk_run_chains). After each launch the host downloads the
//     per-chain statistics and the whole best_S block (one memcpy), appends a
//     CSV row, feeds newly seen sets into the elite pool, and drains the
//     record-custody / tightness-histogram queues.
//   * A launch is *not* an epoch boundary. An **epoch** starts with
//     lsa_init_from_sets + lss_init: every chain is (re)initialised from a host
//     set. Epoch boundaries happen (a) at the start, (b) when enough chains
//     have raised restart_req (they are re-seeded from the elite pool or the
//     seed mix, everybody else restarts from its own best set), and (c) at
//     every checkpoint. Because the device evolution inside an epoch is a
//     deterministic function of (initial sets, epoch seed, K, launch count),
//     a checkpoint taken at an epoch boundary resumes bit-identically —
//     that is the determinism contract of `--resume`.
//   * Custody chain (docs/design.md §2.4): the device sets a flag and copies S into an
//     output slot as soon as |S| >= target. The host writes the set to
//     runs/<run>/found/ FIRST, then runs tools/verify_s and
//     python/verify_S.py as subprocesses and logs RECORD only when both print
//     ok=1 with the same size; otherwise VERIFY-FAIL, loudly.
//
// This tool owns no CUDA kernels: everything device-side is T3.2a/b's API.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "adjacency_cuda.h"
#include "kiss/adjacency.h"
#include "kiss/group.h"
#include "kiss/io.h"
#include "kiss/leech.h"
#include "kiss/run_config.h"
#include "kiss/types.h"
#include "kiss/verify.h"
#include "kiss/version.h"
#include "kiss_cuda.h"
#include "ls_state.cuh"
#include "ls_search.cuh"
#include "kiss/platform.h"

namespace fs = std::filesystem;
using kiss::Adjacency;
using kiss::Leech;
using kiss::RunConfig;
using kiss::Vec;
using kiss::cuda::LSChainStats;
using kiss::cuda::LSFound;
using kiss::cuda::LSParams;
using kiss::cuda::LSSearch;
using kiss::cuda::LSSearchParams;
using kiss::cuda::LSState;
using kiss::cuda::SMAX;

namespace {

constexpr int kN = kiss::N;
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
std::string utc_stamp() {
  const std::time_t t = std::time(nullptr);
  const std::tm tm = kiss::gmtime_utc(t);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y%m%dT%H%M%SZ", &tm);
  return buf;
}

std::string gb(std::size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  return buf;
}

std::string hex8(uint32_t h) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "%08x", h);
  return buf;
}

double now_s(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Run a command, capture stdout+stderr, return the exit status (kiss/platform.h).
using kiss::run_capture;

// Parse `RESULT ok=<0|1> size=<n>` out of a verifier's output.
bool parse_result(const std::string& out, int* ok, long* size) {
  const std::size_t pos = out.rfind("RESULT ");
  if (pos == std::string::npos) return false;
  const std::string line = out.substr(pos, out.find('\n', pos) - pos);
  const std::size_t okp = line.find("ok=");
  const std::size_t szp = line.find("size=");
  if (okp == std::string::npos || szp == std::string::npos) return false;
  *ok = std::atoi(line.c_str() + okp + 3);
  *size = std::atol(line.c_str() + szp + 5);
  return true;
}

std::vector<Vec> to_vecs(const Leech& L, const std::vector<uint32_t>& S) {
  std::vector<Vec> v;
  v.reserve(S.size());
  for (uint32_t i : S) v.push_back(L.C[i]);
  return v;
}

std::vector<uint32_t> load_indices(const Leech& L, const fs::path& p) {
  std::vector<uint32_t> S;
  for (const Vec& r : kiss::read_set(p)) {
    const int32_t idx = L.index_of(r);
    if (idx < 0) throw std::runtime_error(p.string() + ": row is not a minimal vector");
    S.push_back(static_cast<uint32_t>(idx));
  }
  std::sort(S.begin(), S.end());
  return S;
}

uint32_t set_hash(const std::vector<uint32_t>& S) { return kiss::cuda::ls_set_hash_host(S.data(), S.size()); }

// Keep only the vertices whose antipode is also present (a set for an
// antipodal chain must be closed under negation).
std::vector<uint32_t> make_antipodal(const Leech& L, const std::vector<uint32_t>& S) {
  const std::unordered_set<uint32_t> in(S.begin(), S.end());
  std::vector<uint32_t> out;
  out.reserve(S.size());
  for (uint32_t v : S)
    if (in.count(L.neg[v])) out.push_back(v);
  return out;
}

// Random deletion of a fraction of the members (antipodal: whole pairs).
std::vector<uint32_t> delete_fraction(const Leech& L, const std::vector<uint32_t>& S, double frac, bool antipodal,
                                      std::mt19937_64& rng) {
  if (S.empty()) return S;
  if (!antipodal) {
    std::vector<uint32_t> s = S;
    std::shuffle(s.begin(), s.end(), rng);
    const std::size_t keep = static_cast<std::size_t>(
        std::max(1.0, std::floor(static_cast<double>(s.size()) * (1.0 - frac))));
    s.resize(std::min(keep, s.size()));
    std::sort(s.begin(), s.end());
    return s;
  }
  std::vector<uint32_t> reps;
  const std::unordered_set<uint32_t> in(S.begin(), S.end());
  for (uint32_t v : S)
    if (v < L.neg[v] && in.count(L.neg[v])) reps.push_back(v);
  std::shuffle(reps.begin(), reps.end(), rng);
  const std::size_t keep = static_cast<std::size_t>(
      std::max(1.0, std::floor(static_cast<double>(reps.size()) * (1.0 - frac))));
  std::vector<uint32_t> out;
  for (std::size_t i = 0; i < std::min(keep, reps.size()); ++i) {
    out.push_back(reps[i]);
    out.push_back(L.neg[reps[i]]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Random greedy maximal independent set (antipodal: adds pairs).
std::vector<uint32_t> greedy_set(const Leech& L, const Adjacency& adj, std::mt19937_64& rng, bool antipodal) {
  std::vector<uint32_t> order(static_cast<std::size_t>(kN));
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<uint8_t> blocked(static_cast<std::size_t>(kN), 0);
  std::vector<uint32_t> S;
  auto take = [&](uint32_t v) {
    blocked[v] = 1;
    S.push_back(v);
    const uint32_t* row = adj.row(v);
    for (int i = 0; i < kiss::DEG; ++i) blocked[row[i]] = 1;
  };
  for (uint32_t v : order) {
    if (blocked[v]) continue;
    if (antipodal) {
      const uint32_t nv = L.neg[v];
      if (blocked[nv] || nv == v) continue;
      take(v);
      take(nv);
    } else {
      take(v);
    }
    if (S.size() >= static_cast<std::size_t>(SMAX) - 1) break;
  }
  std::sort(S.begin(), S.end());
  return S;
}

// ---------------------------------------------------------------------------
// plateau atoms (runs/ls_search/plateau_atoms_496.json, T3.1 + T3.2b)
// ---------------------------------------------------------------------------
struct PlateauAtom {
  std::vector<uint32_t> rem, add;
};

std::vector<PlateauAtom> load_plateau_atoms(const fs::path& p) {
  std::vector<PlateauAtom> out;
  if (p.empty() || !fs::exists(p)) return out;
  const kiss::Json j = kiss::load_json(p);
  const kiss::Json* atoms = j.find("atoms");
  if (!atoms || !atoms->is_array()) throw std::runtime_error(p.string() + ": no \"atoms\" array");
  for (const kiss::Json& a : atoms->arr) {
    PlateauAtom at;
    const kiss::Json* r = a.find("remove_vertices");
    const kiss::Json* d = a.find("add_vertices");
    if (!r || !d || !r->is_array() || !d->is_array())
      throw std::runtime_error(p.string() + ": atom without remove_vertices/add_vertices");
    for (const kiss::Json& v : r->arr) at.rem.push_back(static_cast<uint32_t>(v.num));
    for (const kiss::Json& v : d->arr) at.add.push_back(static_cast<uint32_t>(v.num));
    out.push_back(std::move(at));
  }
  return out;
}

std::vector<uint32_t> plateau_image(const std::vector<uint32_t>& base, const std::vector<PlateauAtom>& atoms,
                                    uint64_t mask) {
  std::set<uint32_t> S(base.begin(), base.end());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (!((mask >> i) & 1u)) continue;
    for (uint32_t v : atoms[i].rem) S.erase(v);
    for (uint32_t v : atoms[i].add) S.insert(v);
  }
  return std::vector<uint32_t>(S.begin(), S.end());
}

// ---------------------------------------------------------------------------
// seed factory
// ---------------------------------------------------------------------------
enum SeedKind { SK_496 = 0, SK_488, SK_PLATEAU, SK_CO0_496, SK_CO0_488, SK_ELITE, SK_GREEDY, SK_COUNT };
const char* kSeedKindName[SK_COUNT] = {"s496", "s488", "plateau496", "co0_496", "co0_488", "elite", "greedy"};

struct EliteEntry {
  uint32_t size = 0, hash = 0;
  std::vector<uint32_t> S;
};

class SeedFactory {
 public:
  SeedFactory(const RunConfig& c, const Leech& L, const Adjacency& adj, std::vector<uint32_t> s496,
              std::vector<uint32_t> s488, std::vector<PlateauAtom> atoms, std::vector<kiss::IndexPerm> co0)
      : c_(c), L_(L), adj_(adj), s496_(std::move(s496)), s488_(std::move(s488)), atoms_(std::move(atoms)),
        co0_(std::move(co0)) {
    double acc = 0;
    const double w[SK_COUNT] = {c.mix.s496,   c.mix.s488,  c.mix.plateau496, c.mix.co0_496,
                               c.mix.co0_488, c.mix.elite, c.mix.greedy};
    for (int k = 0; k < SK_COUNT; ++k) {
      double ww = w[k];
      if ((k == SK_CO0_496 || k == SK_CO0_488) && co0_.empty()) ww = 0;
      if (k == SK_PLATEAU && atoms_.empty()) ww = 0;
      if (k == SK_488 && s488_.empty()) ww = 0;
      if ((k == SK_CO0_488) && s488_.empty()) ww = 0;
      acc += ww;
      cum_[k] = acc;
    }
    total_ = acc;
  }

  // Deterministic assignment of the seed kind to chain index `b` of `B`
  // (stratified: the kinds appear in blocks of the requested proportions).
  SeedKind kind_for(int b, int B) const {
    if (total_ <= 0) return SK_GREEDY;
    const double x = (static_cast<double>(b) + 0.5) / static_cast<double>(B) * total_;
    for (int k = 0; k < SK_COUNT; ++k)
      if (x < cum_[k]) return static_cast<SeedKind>(k);
    return SK_GREEDY;
  }
  SeedKind random_kind(std::mt19937_64& rng) const {
    if (total_ <= 0) return SK_GREEDY;
    const double x = std::uniform_real_distribution<double>(0.0, total_)(rng);
    for (int k = 0; k < SK_COUNT; ++k)
      if (x < cum_[k]) return static_cast<SeedKind>(k);
    return SK_GREEDY;
  }

  std::vector<uint32_t> make(SeedKind kind, bool antipodal, std::mt19937_64& rng,
                             const std::vector<EliteEntry>& elite) const {
    std::vector<uint32_t> base;
    switch (kind) {
      case SK_496: base = s496_; break;
      case SK_488: base = s488_.empty() ? s496_ : s488_; break;
      case SK_PLATEAU:
        base = atoms_.empty() ? s496_
                              : plateau_image(s496_, atoms_,
                                              1 + (rng() % ((1ull << atoms_.size()) - 1)));
        break;
      case SK_CO0_496: {
        const std::vector<uint32_t> img =
            atoms_.empty() ? s496_ : plateau_image(s496_, atoms_, rng() % (1ull << atoms_.size()));
        base = co0_.empty() ? img : kiss::apply(co0_[rng() % co0_.size()], img);
        break;
      }
      case SK_CO0_488: {
        const std::vector<uint32_t>& src = s488_.empty() ? s496_ : s488_;
        base = co0_.empty() ? src : kiss::apply(co0_[rng() % co0_.size()], src);
        break;
      }
      case SK_ELITE:
        if (elite.empty()) return make(SK_PLATEAU, antipodal, rng, elite);
        base = elite[rng() % elite.size()].S;
        break;
      case SK_GREEDY:
      default:
        return greedy_set(L_, adj_, rng, antipodal);
    }
    std::sort(base.begin(), base.end());
    if (antipodal) base = make_antipodal(L_, base);
    if (base.empty()) return greedy_set(L_, adj_, rng, antipodal);
    const double f = std::uniform_real_distribution<double>(c_.delete_frac_min, c_.delete_frac_max)(rng);
    std::vector<uint32_t> s = delete_fraction(L_, base, f, antipodal, rng);
    if (s.size() > static_cast<std::size_t>(SMAX)) s.resize(static_cast<std::size_t>(SMAX));
    return s;
  }

 private:
  const RunConfig& c_;
  const Leech& L_;
  const Adjacency& adj_;
  std::vector<uint32_t> s496_, s488_;
  std::vector<PlateauAtom> atoms_;
  std::vector<kiss::IndexPerm> co0_;
  double cum_[SK_COUNT] = {0};
  double total_ = 0;
};

// ---------------------------------------------------------------------------
// one engine state (antipodal or plain)
// ---------------------------------------------------------------------------
struct Engine {
  std::string name;
  bool antipodal = false;
  int B = 0;
  int nslots = 0;
  LSState st{};
  LSSearch ss{};
  uint32_t* d_flag = nullptr;
  uint32_t* d_out = nullptr;
  cudaStream_t stream = nullptr;         // the two states run concurrently
  std::vector<uint32_t> chain_best;      // host-side max-so-far per chain
  std::vector<uint8_t> reseed_pending;   // restart_req seen since the last epoch boundary
  int pending_count = 0;
  std::vector<uint32_t> h_best_S;        // [B][SMAX] staging buffer
  std::vector<uint32_t> h_best_size;      // [B] — ALWAYS downloaded together with h_best_S
  std::vector<LSChainStats> stats;
};

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
struct SeenSet {
  uint32_t size = 0;
  int launch = 0;
  double t = 0;
  std::string hist;
};

class Driver {
 public:
  Driver(RunConfig c, std::vector<std::string> argv_echo)
      : c_(std::move(c)), argv_echo_(std::move(argv_echo)) {}

  int run(bool resume, bool dry_run);

 private:
  // setup
  void open_run_dir(bool resume);
  void wait_for_vram();
  void build_fixtures();
  void alloc_engines();
  void seed_initial();
  void snapshot();

  // loop
  void sanitise_seeds(Engine& e, std::vector<std::vector<uint32_t>>& sets);
  void begin_epoch(Engine& e, std::vector<std::vector<uint32_t>>& sets, bool reset_stats);
  void launch_all();
  void collect(double launch_seconds);
  void download_best(Engine& e);
  void harvest_best(Engine& e);
  void handle_found(Engine& e);
  void reseed_pass(Engine& e);
  void checkpoint(bool final_exit);
  bool load_checkpoint();
  void watchdog(bool force);
  void selfcheck();
  void drain_hist_queue();
  void drain_verify_queue(bool all);
  void custody(const std::vector<uint32_t>& S, const std::string& origin, bool force_now);
  std::string tight_histogram(const std::vector<std::vector<uint32_t>>& sets, std::vector<std::string>& out);

  RunConfig c_;
  std::vector<std::string> argv_echo_;
  fs::path run_dir_, found_dir_, sets_dir_;
  std::ofstream log_csv_, sets_csv_, gpu_csv_, notes_;

  Leech L_;
  std::unique_ptr<Adjacency> adj_;
  kiss::cuda::DeviceLeech dl_{};
  uint32_t* d_adj_ = nullptr;
  std::vector<uint32_t> s496_, s488_;
  std::vector<PlateauAtom> atoms_;
  std::vector<kiss::IndexPerm> co0_;
  std::unique_ptr<SeedFactory> seeds_;
  std::vector<Engine> eng_;

  // histogram scratch
  int hist_batch_ = 16;
  uint32_t* d_hist_S_ = nullptr;
  uint32_t* d_hist_size_ = nullptr;
  uint16_t* d_hist_tight_ = nullptr;

  // run bookkeeping
  std::mt19937_64 rng_;
  std::chrono::steady_clock::time_point t0_;
  double elapsed_base_ = 0;
  int launch_ = 0, epoch_ = 0;
  long long iters_total_ = 0;
  uint32_t best_overall_ = 0;
  int reseeds_ = 0, found_written_ = 0, records_ = 0, verify_fail_ = 0, dup_found_ = 0, bad_seeds_ = 0;
  double last_ckpt_t_ = 0, last_watchdog_t_ = -1e9;
  std::vector<EliteEntry> elite_;
  std::unordered_map<uint32_t, uint32_t> elite_hash_;   // hash -> size
  std::unordered_map<uint32_t, SeenSet> seen_;          // distinct best sets >= log_min_size
  std::map<uint32_t, int> distinct_by_size_;
  std::unordered_set<uint32_t> found_hashes_;
  int set_files_ = 0;
  std::vector<std::pair<uint32_t, std::vector<uint32_t>>> hist_queue_;   // (hash, set)
  std::vector<std::pair<std::string, std::vector<uint32_t>>> verify_queue_;
  std::string gpu_last_ = "";
  double gpu_tmax_ = 0, gpu_pmax_ = 0, gpu_clk_min_ = 1e9, gpu_clk_sum_ = 0;
  int gpu_samples_ = 0;
};

void Driver::open_run_dir(bool resume) {
  if (c_.run_name.empty()) {
    fs::path stem = c_.source_file.empty() ? fs::path("run") : fs::path(c_.source_file).stem();
    c_.run_name = stem.string() + "_" + utc_stamp();
  }
  run_dir_ = fs::path(c_.out_dir) / c_.run_name;
  found_dir_ = run_dir_ / "found";
  sets_dir_ = run_dir_ / "sets";
  fs::create_directories(found_dir_);
  fs::create_directories(sets_dir_);
  fs::create_directories(run_dir_ / "checkpoints");

  kiss::Json eff = kiss::run_config_to_json(c_);
  eff.set("_git", kiss::Json::string(kiss::git_hash()));
  eff.set("_source_file", kiss::Json::string(c_.source_file));
  eff.set("_started_utc", kiss::Json::string(utc_stamp()));
  std::string cmdline;
  for (const std::string& a : argv_echo_) cmdline += (cmdline.empty() ? "" : " ") + a;
  eff.set("_argv", kiss::Json::string(cmdline));
  kiss::save_json(run_dir_ / "config_effective.json", eff);

  const bool exists = fs::exists(run_dir_ / "log.csv");
  log_csv_.open(run_dir_ / "log.csv", std::ios::app);
  if (!exists)
    log_csv_ << "t_s,launch,epoch,iter_per_chain,iters_total,best,mean_best,improved,restarts,reseeds,"
                "it_per_s,elite,distinct_ge_log,found,records,sizes_hash,gpu_temp_c,gpu_power_w,gpu_clk_mhz\n";
  const bool sexists = fs::exists(run_dir_ / "sets.csv");
  sets_csv_.open(run_dir_ / "sets.csv", std::ios::app);
  if (!sexists) sets_csv_ << "t_s,launch,state,chain,size,hash,file,tight_histogram\n";
  const bool gexists = fs::exists(run_dir_ / "gpu.csv");
  gpu_csv_.open(run_dir_ / "gpu.csv", std::ios::app);
  if (!gexists) gpu_csv_ << "t_s,temperature_c,power_w,clocks_sm_mhz,clocks_mem_mhz,util_pct\n";
  notes_.open(run_dir_ / "run.log", std::ios::app);
  notes_ << "\n==== " << utc_stamp() << " " << cmdline << (resume ? "  [resume]" : "") << "\n";
  notes_.flush();
  std::printf("run dir   : %s\n", run_dir_.string().c_str());
}

void Driver::wait_for_vram() {
  const std::size_t need = static_cast<std::size_t>(c_.vram_min_gb * 1024.0 * 1024.0 * 1024.0);
  std::size_t fr = 0, to = 0;
  for (int a = 0; a <= c_.vram_retries; ++a) {
    kiss::cuda::device_mem_info(&fr, &to);
    if (to == 0) throw std::runtime_error("no CUDA device visible (cudaMemGetInfo failed)");
    if (fr >= need) {
      std::printf("gpu       : %s free of %s (need %s)\n", gb(fr).c_str(), gb(to).c_str(), gb(need).c_str());
      return;
    }
    std::printf("gpu       : only %s of %s free (need %s) — retry %d/%d in %d s\n", gb(fr).c_str(), gb(to).c_str(),
                gb(need).c_str(), a + 1, c_.vram_retries, c_.vram_retry_seconds);
    std::fflush(stdout);
    if (a == c_.vram_retries) break;
    std::this_thread::sleep_for(std::chrono::seconds(c_.vram_retry_seconds));
  }
  throw std::runtime_error("the GPU is busy: only " + gb(fr) + " free of " + gb(to) + ", need " + gb(need) +
                           " (another process holds the card; retried " + std::to_string(c_.vram_retries) + " times)");
}

void Driver::build_fixtures() {
  const fs::path data_dir(c_.data_dir);
  L_ = kiss::load_leech(data_dir);
  const fs::path adj_path = data_dir / "adj.u32";
  if (!fs::exists(adj_path))
    throw std::runtime_error(adj_path.string() + " missing — run tools/build_adj " + data_dir.string());
  adj_ = std::make_unique<Adjacency>(adj_path);
  s496_ = load_indices(L_, data_dir / "S496.txt");
  s488_ = load_indices(L_, data_dir / "S488.txt");
  atoms_ = load_plateau_atoms(c_.plateau_atoms);
  std::printf("fixtures  : |S496|=%zu |S488|=%zu plateau atoms=%zu\n", s496_.size(), s488_.size(), atoms_.size());

  // Validate the plateau images once (cheap: 496 choose 2 integer dots each).
  if (!atoms_.empty()) {
    int bad = 0;
    const uint64_t nmask = 1ull << atoms_.size();
    for (uint64_t m = 0; m < nmask; ++m) {
      const std::vector<uint32_t> img = plateau_image(s496_, atoms_, m);
      const kiss::VerifyResult r = kiss::verify_independent(L_, to_vecs(L_, img));
      if (!r.ok || r.size != s496_.size()) ++bad;
    }
    std::printf("plateau   : %llu images of the 496 from %zu commuting atoms, %d invalid\n",
                static_cast<unsigned long long>(nmask), atoms_.size(), bad);
    if (bad) {
      notes_ << "WARNING: " << bad << " plateau images are not valid 496s — atoms disabled\n";
      atoms_.clear();
    }
  }

  const bool need_co0 = (c_.mix.co0_496 > 0 || c_.mix.co0_488 > 0) && c_.co0_count > 0;
  if (need_co0) {
    const fs::path gdir = c_.group_dir.empty() ? (data_dir / "group") : fs::path(c_.group_dir);
    const auto t = std::chrono::steady_clock::now();
    std::vector<kiss::IndexPerm> gens = kiss::co0_generator_perms(L_, gdir);
    kiss::ProductReplacement pr(gens, c_.co0_slots, c_.seed ^ 0x5DEECE66Dull);
    pr.burn_in(c_.co0_burnin);
    co0_.reserve(static_cast<std::size_t>(c_.co0_count));
    for (int i = 0; i < c_.co0_count; ++i) co0_.push_back(pr.next());
    std::printf("co0       : %d random Co_0 elements from %zu generators in %.1f s\n", c_.co0_count, gens.size(),
                now_s(t));
  }

  if (!c_.elite_in.empty() && fs::exists(c_.elite_in)) {
    for (const auto& de : fs::directory_iterator(c_.elite_in)) {
      if (!de.is_regular_file() || de.path().extension() != ".txt") continue;
      try {
        const std::vector<uint32_t> S = load_indices(L_, de.path());
        if (static_cast<int>(S.size()) < c_.elite_min_size) continue;
        const uint32_t h = set_hash(S);
        if (elite_hash_.count(h)) continue;
        elite_hash_[h] = static_cast<uint32_t>(S.size());
        elite_.push_back({static_cast<uint32_t>(S.size()), h, S});
      } catch (const std::exception&) {
      }
    }
    std::printf("elite_in  : preloaded %zu sets from %s\n", elite_.size(), c_.elite_in.c_str());
  }

  seeds_ = std::make_unique<SeedFactory>(c_, L_, *adj_, s496_, s488_, atoms_, co0_);
}

void Driver::alloc_engines() {
  const auto t = std::chrono::steady_clock::now();
  d_adj_ = kiss::cuda::adjacency_to_device(*adj_);
  std::size_t fr = 0;
  kiss::cuda::device_mem_info(&fr, nullptr);
  std::printf("adjacency : uploaded in %.1f s, %s free afterwards\n", now_s(t), gb(fr).c_str());

  int Ba = static_cast<int>(std::llround(static_cast<double>(c_.B) * c_.antipodal_fraction));
  Ba = std::max(0, std::min(c_.B, Ba));
  const int Bp = c_.B - Ba;
  struct Spec {
    const char* name;
    bool anti;
    int B;
  };
  const Spec specs[2] = {{"anti", true, Ba}, {"plain", false, Bp}};
  for (const Spec& s : specs) {
    if (s.B <= 0) continue;
    Engine e;
    e.name = s.name;
    e.antipodal = s.anti;
    e.B = s.B;
    LSParams p;
    p.B = s.B;
    p.antipodal = s.anti;
    p.FL = c_.engine.FL;
    p.TABU = c_.engine.TABU;
    p.force_cap = c_.engine.force_cap;
    e.st = kiss::cuda::lsa_alloc(p);
    LSSearchParams sp;
    sp.CL = c_.engine.CL;
    sp.tmin = c_.engine.tmin;
    sp.tmax = c_.engine.tmax;
    sp.list_tmax = c_.engine.list_tmax;
    sp.tenure = c_.engine.tenure;
    sp.tenure_jitter = c_.engine.tenure_jitter;
    sp.strength = c_.engine.strength;
    sp.select_scan = c_.engine.select_scan;
    sp.tournament_rounds = c_.engine.tournament_rounds;
    sp.greedy_pct = c_.engine.greedy_pct;
    sp.neigh_rounds = c_.engine.neigh_rounds;
    sp.uniform_rounds = c_.engine.uniform_rounds;
    sp.swap_enable = c_.engine.swap_enable;
    sp.max_x_tries = c_.engine.max_x_tries;
    sp.tabu_drain = c_.engine.tabu_drain;
    sp.max_drop = c_.engine.max_drop;
    sp.stall_limit = c_.engine.stall_limit;
    sp.restart_del = c_.engine.restart_del;
    sp.rescan_every = c_.engine.rescan_every;
    sp.accept_equal = c_.engine.accept_equal;
    sp.warps_per_block = c_.engine.warps_per_block;
    e.ss = kiss::cuda::lss_alloc(e.st, sp);
    e.nslots = c_.nslots > 0 ? std::min(c_.nslots, s.B) : s.B;
    KISS_CUDA_CHECK(cudaMalloc(&e.d_flag, sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaMemset(e.d_flag, 0, sizeof(uint32_t)));
    const std::size_t ow = kiss::cuda::ls_out_words(e.nslots);
    KISS_CUDA_CHECK(cudaMalloc(&e.d_out, ow * sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaMemset(e.d_out, 0, ow * sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaStreamCreate(&e.stream));
    e.chain_best.assign(static_cast<std::size_t>(s.B), 0);
    e.reseed_pending.assign(static_cast<std::size_t>(s.B), 0);
    e.h_best_S.assign(static_cast<std::size_t>(s.B) * SMAX, 0);
    e.h_best_size.assign(static_cast<std::size_t>(s.B), 0);
    eng_.push_back(std::move(e));
  }
  if (eng_.empty()) throw std::runtime_error("no chains: B and antipodal_fraction leave both states empty");

  if (c_.hist_per_launch > 0) {
    hist_batch_ = std::min(hist_batch_, std::max(1, c_.hist_per_launch));
    KISS_CUDA_CHECK(cudaMalloc(&d_hist_S_, static_cast<std::size_t>(hist_batch_) * SMAX * sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaMalloc(&d_hist_size_, static_cast<std::size_t>(hist_batch_) * sizeof(uint32_t)));
    KISS_CUDA_CHECK(cudaMalloc(&d_hist_tight_, static_cast<std::size_t>(hist_batch_) * kN * sizeof(uint16_t)));
  }
  kiss::cuda::device_mem_info(&fr, nullptr);
  std::printf("engines   : ");
  for (const Engine& e : eng_) std::printf("%s B=%d slots=%d  ", e.name.c_str(), e.B, e.nslots);
  std::printf("| %s free\n", gb(fr).c_str());
}

// Guard on the one operation that can inject a broken set into the engine.
// Every seed is sorted, de-duplicated and checked pairwise (dot != 16); a set
// that fails is replaced *deterministically* by the 496 and reported loudly —
// it can only mean the device state was already corrupt.
void Driver::sanitise_seeds(Engine& e, std::vector<std::vector<uint32_t>>& sets) {
  if (!c_.verify_seeds) return;
  const int B = static_cast<int>(sets.size());
  std::vector<int> bad(static_cast<std::size_t>(B), 0);
#pragma omp parallel for schedule(dynamic, 8)
  for (int b = 0; b < B; ++b) {
    std::vector<uint32_t>& S = sets[static_cast<std::size_t>(b)];
    std::sort(S.begin(), S.end());
    const std::size_t before = S.size();
    S.erase(std::unique(S.begin(), S.end()), S.end());
    if (S.size() != before) bad[static_cast<std::size_t>(b)] |= 1;
    if (S.size() > static_cast<std::size_t>(SMAX)) {
      S.resize(static_cast<std::size_t>(SMAX));
      bad[static_cast<std::size_t>(b)] |= 2;
    }
    for (std::size_t i = 0; i < S.size() && !(bad[static_cast<std::size_t>(b)] & 4); ++i) {
      if (S[i] >= static_cast<uint32_t>(kN)) { bad[static_cast<std::size_t>(b)] |= 8; break; }
      for (std::size_t j = i + 1; j < S.size(); ++j) {
        if (kiss::dot(L_.C[S[i]], L_.C[S[j]]) == 16) {
          bad[static_cast<std::size_t>(b)] |= 4;
          break;
        }
      }
    }
  }
  int nbad = 0;
  for (int b = 0; b < B; ++b) {
    if (!bad[static_cast<std::size_t>(b)]) continue;
    ++nbad;
    ++bad_seeds_;
    if (nbad <= 4)
      std::printf("!!! INVARIANT FAILURE: %s chain %d seed rejected (flags %d, |S| = %zu) — replaced by the 496\n",
                  e.name.c_str(), b, bad[static_cast<std::size_t>(b)], sets[static_cast<std::size_t>(b)].size());
    sets[static_cast<std::size_t>(b)] = e.antipodal ? make_antipodal(L_, s496_) : s496_;
  }
  if (nbad) {
    notes_ << "INVARIANT FAILURE " << e.name << " launch " << launch_ << ": " << nbad
           << " seed sets rejected and replaced\n";
    notes_.flush();
    std::fflush(stdout);
  }
}

void Driver::begin_epoch(Engine& e, std::vector<std::vector<uint32_t>>& sets, bool reset_stats) {
  sanitise_seeds(e, sets);
  const uint64_t base = c_.seed * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(epoch_) * 0xBF58476D1CE4E5B9ull +
                        (e.antipodal ? 0x1234567ull : 0x89ABCDEull);
  kiss::cuda::lsa_init_from_sets(e.st, sets, dl_, d_adj_, base);
  kiss::cuda::lss_init(e.ss, e.st, dl_, base * 7919ull + 17ull, reset_stats);
  std::fill(e.reseed_pending.begin(), e.reseed_pending.end(), 0);
  e.pending_count = 0;
}

void Driver::seed_initial() {
  for (Engine& e : eng_) {
    std::vector<std::vector<uint32_t>> sets(static_cast<std::size_t>(e.B));
    std::map<std::string, int> kinds;
    for (int b = 0; b < e.B; ++b) {
      const SeedKind k = seeds_->kind_for(b, e.B);
      sets[static_cast<std::size_t>(b)] = seeds_->make(k, e.antipodal, rng_, elite_);
      ++kinds[kSeedKindName[k]];
    }
    std::string desc;
    for (const auto& kv : kinds) desc += kv.first + "=" + std::to_string(kv.second) + " ";
    std::printf("seeds %-5s: %s\n", e.name.c_str(), desc.c_str());
    notes_ << "seeds " << e.name << ": " << desc << "\n";
    begin_epoch(e, sets, true);
    for (int b = 0; b < e.B; ++b)
      e.chain_best[static_cast<std::size_t>(b)] = static_cast<uint32_t>(sets[static_cast<std::size_t>(b)].size());
  }
  ++epoch_;
}

// Download the per-chain statistics and the best_S block without advancing the
// engine (so a checkpoint taken before the first launch is still complete).
void Driver::snapshot() {
  for (Engine& e : eng_) {
    e.stats = kiss::cuda::lss_download_stats(e.ss);
    download_best(e);
  }
}

// ---------------------------------------------------------------------------
// per-launch work
// ---------------------------------------------------------------------------
void Driver::launch_all() {
  for (Engine& e : eng_)
    kiss::cuda::lsk_run_chains(e.st, e.ss, d_adj_, dl_, c_.K, static_cast<uint32_t>(c_.target), e.d_flag, e.d_out,
                               e.nslots, e.stream);
  for (Engine& e : eng_) KISS_CUDA_CHECK(cudaStreamSynchronize(e.stream));
}

// best_size and best_S MUST be read in the same transaction: an epoch boundary
// (reseed pass / checkpoint) replaces both, and mixing a stale size with a fresh
// buffer produces a set with garbage members. That mismatch is what produced the
// bogus 497..568 candidates of the first production run (docs/reports/T3.2c.md §8).
void Driver::download_best(Engine& e) {
  KISS_CUDA_CHECK(cudaMemcpy(e.h_best_size.data(), e.st.best_size, e.h_best_size.size() * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
  KISS_CUDA_CHECK(cudaMemcpy(e.h_best_S.data(), e.st.best_S, e.h_best_S.size() * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
  for (int b = 0; b < e.B; ++b)
    e.h_best_size[static_cast<std::size_t>(b)] =
        std::min<uint32_t>(e.h_best_size[static_cast<std::size_t>(b)], static_cast<uint32_t>(SMAX));
}

// The set chain b would be re-initialised from (its best-ever set).
std::vector<uint32_t> chain_set(const Engine& e, int b) {
  const std::size_t bb = static_cast<std::size_t>(b);
  const uint32_t sz = e.h_best_size[bb];
  return std::vector<uint32_t>(e.h_best_S.begin() + static_cast<long>(bb * SMAX),
                               e.h_best_S.begin() + static_cast<long>(bb * SMAX + sz));
}

void Driver::harvest_best(Engine& e) {
  download_best(e);
  int budget = std::max(1, c_.hist_per_launch);   // new distinct sets fully processed this launch
  for (int b = 0; b < e.B; ++b) {
    const std::size_t bb = static_cast<std::size_t>(b);
    const uint32_t sz = e.h_best_size[bb];
    if (sz == 0 || sz > static_cast<uint32_t>(SMAX)) continue;
    best_overall_ = std::max(best_overall_, sz);
    if (static_cast<int>(sz) < c_.elite_min_size && static_cast<int>(sz) < c_.log_min_size) continue;
    // The hash is order-independent, so the (expensive) sort/copy only happens
    // for sets that are actually new.
    const uint32_t* raw = e.h_best_S.data() + bb * SMAX;
    const uint32_t h = kiss::cuda::ls_set_hash_host(raw, sz);
    const bool new_elite = static_cast<int>(sz) >= c_.elite_min_size && !elite_hash_.count(h);
    const bool new_distinct = static_cast<int>(sz) >= c_.log_min_size && !seen_.count(h);
    if (!new_elite && !new_distinct) continue;
    if (new_distinct && budget <= 0) {
      // Count it (the distinct-set census is the point) but skip the per-set
      // verification / file / histogram until a later launch has budget.
      SeenSet cheap;
      cheap.size = sz;
      cheap.launch = launch_;
      cheap.t = elapsed_base_ + now_s(t0_);
      cheap.hist = "(not sampled)";
      seen_[h] = cheap;
      ++distinct_by_size_[sz];
      continue;
    }
    std::vector<uint32_t> S(raw, raw + sz);
    std::sort(S.begin(), S.end());

    // elite pool (deduped by hash, capped). A corrupt set must never become a
    // seed for other chains, so everything entering the pool is verified.
    if (new_elite) {
      const kiss::VerifyResult evr = kiss::verify_independent(L_, to_vecs(L_, S));
      if (!evr.ok) {
        ++bad_seeds_;
        std::printf("!!! INVARIANT FAILURE: %s chain %d best_S (size %u) is not independent: %s\n", e.name.c_str(), b,
                    sz, evr.message.c_str());
        notes_ << "INVARIANT FAILURE elite " << e.name << "/" << b << ": " << evr.message << "\n";
        notes_.flush();
        continue;
      }
    }
    if (new_elite) {
      if (static_cast<int>(elite_.size()) < c_.elite_cap) {
        elite_hash_[h] = sz;
        elite_.push_back({sz, h, S});
      } else {
        // replace the smallest entry if this one is at least as good
        std::size_t worst = 0;
        for (std::size_t i = 1; i < elite_.size(); ++i)
          if (elite_[i].size < elite_[worst].size) worst = i;
        if (sz >= elite_[worst].size) {
          elite_hash_.erase(elite_[worst].hash);
          elite_hash_[h] = sz;
          elite_[worst] = {sz, h, S};
        }
      }
    }

    // distinct-set log (docs/design.md §5.5: every set >= log_min_size with its histogram)
    if (new_distinct) {
      --budget;
      const kiss::VerifyResult vr = kiss::verify_independent(L_, to_vecs(L_, S));
      if (!vr.ok) {
        std::printf("!!! INVARIANT FAILURE: chain %s/%d best_S of size %u is not independent: %s\n", e.name.c_str(), b,
                    sz, vr.message.c_str());
        notes_ << "INVARIANT FAILURE " << e.name << "/" << b << ": " << vr.message << "\n";
        notes_.flush();
        continue;
      }
      SeenSet ss;
      ss.size = sz;
      ss.launch = launch_;
      ss.t = elapsed_base_ + now_s(t0_);
      seen_[h] = ss;
      ++distinct_by_size_[sz];
      std::string file;
      if (set_files_ < c_.max_set_files) {
        const fs::path f = sets_dir_ / ("S_" + std::to_string(sz) + "_" + hex8(h) + ".txt");
        kiss::write_set(f, to_vecs(L_, S),
                        "gpu_mis run " + c_.run_name + " state " + e.name + " chain " + std::to_string(b) +
                            " launch " + std::to_string(launch_) + " hash " + hex8(h));
        file = f.string();
        ++set_files_;
      }
      sets_csv_ << ss.t << "," << launch_ << "," << e.name << "," << b << "," << sz << "," << hex8(h) << "," << file
                << "," << (c_.hist_per_launch > 0 ? "queued" : "") << "\n";
      sets_csv_.flush();
      if (c_.hist_per_launch > 0) hist_queue_.emplace_back(h, S);
    }
  }
}

void Driver::handle_found(Engine& e) {
  const std::vector<LSFound> found = kiss::cuda::lss_download_output(e.d_out, e.nslots, e.d_flag, true);
  for (const LSFound& f : found) {
    std::vector<uint32_t> S = f.S;
    std::sort(S.begin(), S.end());
    const uint32_t h = set_hash(S);
    if (!found_hashes_.insert(h).second) {
      ++dup_found_;
      continue;
    }
    custody(S, e.name + "/chain" + std::to_string(f.chain) + "/launch" + std::to_string(launch_),
            static_cast<int>(S.size()) >= 497);
  }
}

void Driver::custody(const std::vector<uint32_t>& S, const std::string& origin, bool force_now) {
  // 1. write FIRST — nothing else may happen before the set is on disk.
  const std::vector<Vec> V = to_vecs(L_, S);
  const fs::path f = found_dir_ / ("S_" + std::to_string(S.size()) + "_" + utc_stamp() + "_" + hex8(set_hash(S)) +
                                   ".txt");
  kiss::write_set(f, V, "gpu_mis run " + c_.run_name + " " + origin + " git " + kiss::git_hash());
  ++found_written_;
  std::printf("\n>>> candidate |S| = %zu (%s) written to %s\n", S.size(), origin.c_str(), f.string().c_str());
  std::fflush(stdout);
  notes_ << "candidate size=" << S.size() << " origin=" << origin << " file=" << f.string() << "\n";
  notes_.flush();
  if (force_now || static_cast<int>(verify_queue_.size()) < 4 * std::max(1, c_.verify_per_launch))
    verify_queue_.emplace_back(f.string(), S);
  if (force_now) drain_verify_queue(false);
}

void Driver::drain_verify_queue(bool all) {
  int budget = all ? static_cast<int>(verify_queue_.size()) : c_.verify_per_launch;
  while (budget-- > 0 && !verify_queue_.empty()) {
    const auto item = verify_queue_.front();
    verify_queue_.erase(verify_queue_.begin());
    const std::string& file = item.first;
    const std::size_t size = item.second.size();

    const std::string cmd1 = c_.verify_s + " " + file + " --data " + c_.data_dir;
    const std::string cmd2 = c_.python + " " + c_.verify_py + " " + file;
    std::string o1, o2;
    const int r1 = run_capture(cmd1, o1);
    const int r2 = run_capture(cmd2, o2);
    int ok1 = 0, ok2 = 0;
    long sz1 = -1, sz2 = -1;
    const bool p1 = parse_result(o1, &ok1, &sz1);
    const bool p2 = parse_result(o2, &ok2, &sz2);
    const bool good = (r1 == 0) && (r2 == 0) && p1 && p2 && ok1 == 1 && ok2 == 1 && sz1 == sz2 &&
                      sz1 == static_cast<long>(size);
    notes_ << "verify " << file << "\n  $ " << cmd1 << "\n  " << o1 << "  $ " << cmd2 << "\n  " << o2;
    if (good) {
      if (size >= 497) {
        ++records_;
        std::printf(
            "\n***************************************************************************\n"
            "*** RECORD |S| = %zu VERIFIED BY BOTH VERIFIERS: %s\n"
            "***************************************************************************\n",
            size, file.c_str());
        notes_ << "RECORD size=" << size << " file=" << file << "\n";
      } else {
        std::printf(">>> verified |S| = %zu (below 497, not a record): %s\n", size, file.c_str());
        notes_ << "verified size=" << size << " file=" << file << "\n";
      }
    } else {
      ++verify_fail_;
      std::printf(
          "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "!!! VERIFY-FAIL for %s (claimed |S| = %zu)\n"
          "!!!   %s -> exit %d, ok=%d size=%ld\n"
          "!!!   %s -> exit %d, ok=%d size=%ld\n"
          "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
          file.c_str(), size, cmd1.c_str(), r1, ok1, sz1, cmd2.c_str(), r2, ok2, sz2);
      notes_ << "VERIFY-FAIL " << file << "\n";
    }
    std::fflush(stdout);
    notes_.flush();
  }
}

std::string Driver::tight_histogram(const std::vector<std::vector<uint32_t>>& sets, std::vector<std::string>& out) {
  const int H = static_cast<int>(sets.size());
  std::vector<uint32_t> hS(static_cast<std::size_t>(H) * SMAX, 0), hsz(static_cast<std::size_t>(H), 0);
  for (int i = 0; i < H; ++i) {
    hsz[static_cast<std::size_t>(i)] = static_cast<uint32_t>(sets[static_cast<std::size_t>(i)].size());
    std::copy(sets[static_cast<std::size_t>(i)].begin(), sets[static_cast<std::size_t>(i)].end(),
              hS.begin() + static_cast<long>(i) * SMAX);
  }
  KISS_CUDA_CHECK(cudaMemcpy(d_hist_S_, hS.data(), hS.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  KISS_CUDA_CHECK(cudaMemcpy(d_hist_size_, hsz.data(), hsz.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
  kiss::cuda::tightness_full(dl_, d_hist_S_, d_hist_size_, H, d_hist_tight_);
  KISS_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> tight(static_cast<std::size_t>(H) * kN);
  KISS_CUDA_CHECK(cudaMemcpy(tight.data(), d_hist_tight_, tight.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  out.clear();
  for (int i = 0; i < H; ++i) {
    std::vector<char> inS(static_cast<std::size_t>(kN), 0);
    for (uint32_t v : sets[static_cast<std::size_t>(i)]) inS[v] = 1;
    std::map<int, long> hist;
    const uint16_t* row = tight.data() + static_cast<std::size_t>(i) * kN;
    for (int v = 0; v < kN; ++v)
      if (!inS[static_cast<std::size_t>(v)]) ++hist[row[v]];
    std::string s;
    for (const auto& kv : hist) {
      if (kv.first == 0 && kv.second == 0) continue;
      s += (s.empty() ? "" : "|") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
    }
    out.push_back(s);
  }
  return out.empty() ? std::string() : out[0];
}

void Driver::drain_hist_queue() {
  if (c_.hist_per_launch <= 0 || hist_queue_.empty()) return;
  int budget = c_.hist_per_launch;
  while (budget > 0 && !hist_queue_.empty()) {
    const int H = std::min({hist_batch_, budget, static_cast<int>(hist_queue_.size())});
    std::vector<std::vector<uint32_t>> batch;
    std::vector<uint32_t> hashes;
    for (int i = 0; i < H; ++i) {
      batch.push_back(hist_queue_[static_cast<std::size_t>(i)].second);
      hashes.push_back(hist_queue_[static_cast<std::size_t>(i)].first);
    }
    hist_queue_.erase(hist_queue_.begin(), hist_queue_.begin() + H);
    std::vector<std::string> hists;
    tight_histogram(batch, hists);
    for (int i = 0; i < H; ++i) {
      const uint32_t h = hashes[static_cast<std::size_t>(i)];
      auto it = seen_.find(h);
      if (it == seen_.end()) continue;
      it->second.hist = hists[static_cast<std::size_t>(i)];
      sets_csv_ << it->second.t << "," << it->second.launch << ",hist,-," << it->second.size << "," << hex8(h) << ",,"
                << hists[static_cast<std::size_t>(i)] << "\n";
    }
    sets_csv_.flush();
    budget -= H;
  }
}

void Driver::reseed_pass(Engine& e) {
  download_best(e);   // never build seeds from a stale snapshot
  std::vector<std::vector<uint32_t>> sets(static_cast<std::size_t>(e.B));
  int fresh = 0;
  for (int b = 0; b < e.B; ++b) {
    const std::size_t bb = static_cast<std::size_t>(b);
    if (e.reseed_pending[bb]) {
      ++fresh;
      const bool from_elite = !elite_.empty() &&
                              std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < c_.reseed_elite_prob;
      const SeedKind k = from_elite ? SK_ELITE : seeds_->random_kind(rng_);
      sets[bb] = seeds_->make(k, e.antipodal, rng_, elite_);
    } else {
      sets[bb] = chain_set(e, b);
      if (sets[bb].empty()) sets[bb] = seeds_->make(seeds_->random_kind(rng_), e.antipodal, rng_, elite_);
    }
  }
  ++epoch_;
  begin_epoch(e, sets, false);
  ++reseeds_;
  notes_ << "reseed " << e.name << " launch " << launch_ << ": " << fresh << " fresh seeds, epoch " << epoch_ << "\n";
}

void Driver::selfcheck() {
  for (Engine& e : eng_) {
    const std::vector<kiss::cuda::LSCheck> chk = kiss::cuda::lsa_check(e.st, dl_);
    long errs = 0, tightbad = 0, conf = 0, anti = 0;
    for (const kiss::cuda::LSCheck& c : chk) {
      errs += c.errors();
      tightbad += c.tight_mismatch;
      conf += c.conflict_pairs;
      anti += c.antipodal_bad;
    }
    std::printf("selfcheck : %-5s launch %d — errors=%ld (tight=%ld conflicts=%ld antipodal=%ld)\n", e.name.c_str(),
                launch_, errs, tightbad, conf, anti);
    notes_ << "selfcheck " << e.name << " launch " << launch_ << " errors=" << errs << " tight=" << tightbad
           << " conflicts=" << conf << " antipodal=" << anti << "\n";
    notes_.flush();
    if (errs) throw std::runtime_error("selfcheck failed on state " + e.name + " (" + std::to_string(errs) +
                                       " errors) — engine invariants broken");
  }
}

void Driver::watchdog(bool force) {
  if (c_.watchdog_seconds <= 0) return;
  const double t = elapsed_base_ + now_s(t0_);
  if (!force && t - last_watchdog_t_ < c_.watchdog_seconds) return;
  last_watchdog_t_ = t;
  std::string out;
  const int rc = run_capture(c_.nvidia_smi +
                                 " --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu"
                                 " --format=csv,noheader,nounits",
                             out);
  if (rc != 0 || out.empty()) return;
  std::string line = out.substr(0, out.find('\n'));
  // strip spaces
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
  gpu_csv_ << t << "," << line << "\n";
  gpu_csv_.flush();
  gpu_last_ = line;
  double temp = 0, pw = 0, clk = 0;
  if (std::sscanf(line.c_str(), "%lf,%lf,%lf", &temp, &pw, &clk) == 3) {
    gpu_tmax_ = std::max(gpu_tmax_, temp);
    gpu_pmax_ = std::max(gpu_pmax_, pw);
    gpu_clk_min_ = std::min(gpu_clk_min_, clk);
    gpu_clk_sum_ += clk;
    ++gpu_samples_;
  }
}

// ---------------------------------------------------------------------------
// checkpoints
// ---------------------------------------------------------------------------
void Driver::checkpoint(bool final_exit) {
  // Every checkpoint is an epoch boundary: the chain sets written here are
  // exactly the sets a resume re-initialises from, with epoch_ + 1 as the epoch
  // seed on both paths.
  for (Engine& e : eng_) download_best(e);
  const int next_epoch = epoch_ + 1;
  const fs::path tmp = run_dir_ / "checkpoint.tmp";
  {
    std::ofstream o(tmp, std::ios::binary);
    if (!o) throw std::runtime_error("cannot write " + tmp.string());
    o << "# gpu_mis checkpoint v1\n";
    o << "version 1\n";
    o << "run_name " << c_.run_name << "\n";
    o << "git " << kiss::git_hash() << "\n";
    o << "epoch " << next_epoch << "\n";
    o << "launch " << launch_ << "\n";
    o << "elapsed_s " << (elapsed_base_ + now_s(t0_)) << "\n";
    o << "iters_total " << iters_total_ << "\n";
    o << "best_overall " << best_overall_ << "\n";
    o << "counters " << reseeds_ << " " << found_written_ << " " << records_ << " " << verify_fail_ << " "
      << set_files_ << " " << dup_found_ << "\n";
    {
      std::ostringstream rs;
      rs << rng_;
      o << "rng " << rs.str() << "\n";
    }
    for (const Engine& e : eng_) {
      o << "state " << e.name << " " << (e.antipodal ? 1 : 0) << " " << e.B << "\n";
      for (int b = 0; b < e.B; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        const uint32_t sz = e.h_best_size[bb];
        o << "chain " << b << " " << sz << " " << e.chain_best[bb];
        for (uint32_t i = 0; i < sz; ++i) o << " " << e.h_best_S[bb * SMAX + i];
        o << "\n";
      }
    }
    o << "elite " << elite_.size() << "\n";
    for (const EliteEntry& en : elite_) {
      o << "e " << en.size;
      for (uint32_t v : en.S) o << " " << v;
      o << "\n";
    }
    o << "seen " << seen_.size() << "\n";
    for (const auto& kv : seen_) o << "s " << kv.first << " " << kv.second.size << " " << kv.second.launch << "\n";
    o << "end\n";
    if (!o) throw std::runtime_error("checkpoint write failed");
  }
  fs::rename(tmp, run_dir_ / "checkpoint.txt");
  last_ckpt_t_ = elapsed_base_ + now_s(t0_);
  std::printf("checkpoint: launch %d, epoch -> %d, %s\n", launch_, next_epoch,
              (run_dir_ / "checkpoint.txt").string().c_str());
  notes_ << "checkpoint launch " << launch_ << " epoch " << next_epoch << "\n";
  notes_.flush();

  if (final_exit) return;
  // Continue the run from exactly the state a resume would rebuild.
  epoch_ = next_epoch;
  for (Engine& e : eng_) {
    std::vector<std::vector<uint32_t>> sets(static_cast<std::size_t>(e.B));
    for (int b = 0; b < e.B; ++b) {
      const std::size_t bb = static_cast<std::size_t>(b);
      sets[bb] = chain_set(e, b);
      if (sets[bb].empty()) sets[bb] = seeds_->make(seeds_->random_kind(rng_), e.antipodal, rng_, elite_);
    }
    begin_epoch(e, sets, false);
  }
}

bool Driver::load_checkpoint() {
  const fs::path f = run_dir_ / "checkpoint.txt";
  if (!fs::exists(f)) throw std::runtime_error("--resume: " + f.string() + " does not exist");
  std::ifstream in(f, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + f.string());
  std::string line;
  std::map<std::string, std::vector<std::vector<uint32_t>>> state_sets;
  std::map<std::string, std::vector<uint32_t>> state_best;
  std::string cur;
  int ckpt_epoch = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string tag;
    is >> tag;
    if (tag == "epoch") is >> ckpt_epoch;
    else if (tag == "launch") is >> launch_;
    else if (tag == "elapsed_s") is >> elapsed_base_;
    else if (tag == "iters_total") is >> iters_total_;
    else if (tag == "best_overall") is >> best_overall_;
    else if (tag == "counters") is >> reseeds_ >> found_written_ >> records_ >> verify_fail_ >> set_files_ >> dup_found_;
    else if (tag == "rng") {
      std::string rest;
      std::getline(is, rest);
      std::istringstream rs(rest);
      rs >> rng_;
    } else if (tag == "state") {
      int anti = 0, B = 0;
      is >> cur >> anti >> B;
      state_sets[cur].assign(static_cast<std::size_t>(B), {});
      state_best[cur].assign(static_cast<std::size_t>(B), 0);
    } else if (tag == "chain") {
      int b = 0;
      uint32_t sz = 0, cb = 0;
      is >> b >> sz >> cb;
      std::vector<uint32_t> S(sz);
      for (uint32_t i = 0; i < sz; ++i) is >> S[i];
      if (b >= 0 && b < static_cast<int>(state_sets[cur].size())) {
        state_sets[cur][static_cast<std::size_t>(b)] = std::move(S);
        state_best[cur][static_cast<std::size_t>(b)] = cb;
      }
    } else if (tag == "e") {
      uint32_t sz = 0;
      is >> sz;
      std::vector<uint32_t> S(sz);
      for (uint32_t i = 0; i < sz; ++i) is >> S[i];
      const uint32_t h = set_hash(S);
      if (!elite_hash_.count(h)) {
        elite_hash_[h] = sz;
        elite_.push_back({sz, h, S});
      }
    } else if (tag == "s") {
      uint32_t h = 0, sz = 0;
      int lc = 0;
      is >> h >> sz >> lc;
      SeenSet ss;
      ss.size = sz;
      ss.launch = lc;
      seen_[h] = ss;
      ++distinct_by_size_[sz];
    }
  }
  epoch_ = ckpt_epoch;
  for (Engine& e : eng_) {
    auto it = state_sets.find(e.name);
    if (it == state_sets.end() || static_cast<int>(it->second.size()) != e.B)
      throw std::runtime_error("--resume: checkpoint has no state \"" + e.name + "\" with B = " +
                               std::to_string(e.B) + " (config changed since the checkpoint?)");
    for (int b = 0; b < e.B; ++b) {
      if (it->second[static_cast<std::size_t>(b)].empty())
        it->second[static_cast<std::size_t>(b)] = seeds_->make(seeds_->random_kind(rng_), e.antipodal, rng_, elite_);
      e.chain_best[static_cast<std::size_t>(b)] = state_best[e.name][static_cast<std::size_t>(b)];
    }
    begin_epoch(e, it->second, true);
  }
  std::printf("resume    : launch %d, epoch %d, elapsed %.1f s, elite %zu, distinct %zu\n", launch_, epoch_,
              elapsed_base_, elite_.size(), seen_.size());
  return true;
}

// ---------------------------------------------------------------------------
void Driver::collect(double launch_seconds) {
  uint32_t best = 0;
  double mean_sum = 0;
  int nchain = 0, improved = 0, restarts = 0;
  uint32_t sizes_hash = 0;
  for (Engine& e : eng_) {
    e.stats = kiss::cuda::lss_download_stats(e.ss);
    const std::vector<uint32_t> rr = kiss::cuda::lss_download_restart_req(e.ss, true);
    for (int b = 0; b < e.B; ++b) {
      const std::size_t bb = static_cast<std::size_t>(b);
      const uint32_t bs = e.stats[bb].best_size;
      best = std::max(best, bs);
      mean_sum += bs;
      ++nchain;
      if (bs > e.chain_best[bb]) {
        e.chain_best[bb] = bs;
        ++improved;
      }
      sizes_hash += kiss::cuda::ls_mix32(bs * 1000003u + e.stats[bb].size);
      if (rr[bb]) {
        if (!e.reseed_pending[bb]) {
          e.reseed_pending[bb] = 1;
          ++e.pending_count;
        }
        ++restarts;
      }
    }
    harvest_best(e);
    handle_found(e);
  }
  best_overall_ = std::max(best_overall_, best);

  const double t = elapsed_base_ + now_s(t0_);
  const double itps = launch_seconds > 0 ? static_cast<double>(c_.B) * c_.K / launch_seconds : 0;
  long distinct_ge = 0;
  for (const auto& kv : distinct_by_size_) distinct_ge += kv.second;
  if (launch_ % c_.log_every == 0) {
    watchdog(false);
    double temp = 0, pw = 0, clk = 0;
    if (!gpu_last_.empty()) std::sscanf(gpu_last_.c_str(), "%lf,%lf,%lf", &temp, &pw, &clk);
    log_csv_ << t << "," << launch_ << "," << epoch_ << "," << (static_cast<long long>(launch_) * c_.K) << ","
             << iters_total_ << "," << best << "," << (nchain ? mean_sum / nchain : 0) << "," << improved << ","
             << restarts << "," << reseeds_ << "," << itps << "," << elite_.size() << "," << distinct_ge << ","
             << found_written_ << "," << records_ << "," << hex8(sizes_hash) << "," << temp << "," << pw << "," << clk
             << "\n";
    log_csv_.flush();
    std::printf("[%7.1fs] launch %5d epoch %3d best %3u mean %6.2f impr %4d restart %4d it/s %8.0f elite %3zu "
                "distinct %4ld found %d rec %d hash %s%s\n",
                t, launch_, epoch_, best, nchain ? mean_sum / nchain : 0.0, improved, restarts, itps, elite_.size(),
                distinct_ge, found_written_, records_, hex8(sizes_hash).c_str(),
                gpu_last_.empty() ? "" : (" gpu " + gpu_last_).c_str());
    std::fflush(stdout);
  }
}

int Driver::run(bool resume, bool dry_run) {
  t0_ = std::chrono::steady_clock::now();
  rng_.seed(c_.seed);
  open_run_dir(resume);
  if (dry_run) {
    std::printf("dry-run   : configuration valid; run dir prepared\n");
    std::printf("RESULT ok=1 dry_run=1 run=%s\n", c_.run_name.c_str());
    return 0;
  }
  wait_for_vram();
  build_fixtures();
  dl_ = kiss::cuda::upload_leech(L_);
  alloc_engines();
  if (resume) load_checkpoint();
  else seed_initial();
  snapshot();
  watchdog(true);
  last_ckpt_t_ = elapsed_base_ + now_s(t0_);

  const double budget = c_.wall_clock_minutes * 60.0;
  std::string stop_reason = "wall_clock";
  for (;;) {
    if (g_stop) { stop_reason = "signal"; break; }
    if (budget > 0 && elapsed_base_ + now_s(t0_) >= budget) { stop_reason = "wall_clock"; break; }
    if (c_.max_launches > 0 && launch_ >= c_.max_launches) { stop_reason = "max_launches"; break; }

    const auto tl = std::chrono::steady_clock::now();
    launch_all();
    const double dt = now_s(tl);
    ++launch_;
    iters_total_ += static_cast<long long>(c_.B) * c_.K;
    collect(dt);
    drain_hist_queue();
    drain_verify_queue(false);

    if (c_.selfcheck_every > 0 && launch_ % c_.selfcheck_every == 0) selfcheck();

    for (Engine& e : eng_) {
      const int need = std::max(1, static_cast<int>(std::llround(c_.reseed_batch_frac * e.B)));
      if (e.pending_count >= need) reseed_pass(e);
    }

    const bool by_launch = c_.checkpoint_launches > 0 && launch_ % c_.checkpoint_launches == 0;
    const bool by_time = c_.checkpoint_minutes > 0 &&
                         (elapsed_base_ + now_s(t0_)) - last_ckpt_t_ >= c_.checkpoint_minutes * 60.0;
    const bool last = (c_.max_launches > 0 && launch_ >= c_.max_launches) ||
                      (budget > 0 && elapsed_base_ + now_s(t0_) >= budget);
    if ((by_launch || by_time) && !last) checkpoint(false);

    if (c_.stop_on_record && records_ > 0) { stop_reason = "record"; break; }
  }

  drain_hist_queue();
  drain_verify_queue(true);
  checkpoint(true);
  watchdog(true);

  // ---- summary -----------------------------------------------------------
  const double total = elapsed_base_ + now_s(t0_);
  std::ostringstream sm;
  sm << "run          " << c_.run_name << "\n"
     << "config       " << c_.source_file << "\n"
     << "git          " << kiss::git_hash() << "\n"
     << "stop_reason  " << stop_reason << "\n"
     << "elapsed_s    " << total << "\n"
     << "launches     " << launch_ << "  (K = " << c_.K << ", B = " << c_.B << ")\n"
     << "iterations   " << iters_total_ << "  (" << (total > 0 ? iters_total_ / total : 0) << " it/s aggregate)\n"
     << "best         " << best_overall_ << "\n"
     << "epochs       " << epoch_ << "  reseed passes " << reseeds_ << "\n"
     << "elite pool   " << elite_.size() << "\n"
     << "found files  " << found_written_ << "  (duplicates suppressed " << dup_found_ << ")\n"
     << "records      " << records_ << "  verify-fail " << verify_fail_ << "  rejected seeds " << bad_seeds_
     << "\n";
  sm << "distinct sets >= " << c_.log_min_size << " by size:";
  for (const auto& kv : distinct_by_size_) sm << " " << kv.first << ":" << kv.second;
  sm << "\n";
  if (gpu_samples_) {
    sm << "gpu          max_temp " << gpu_tmax_ << " C, max_power " << gpu_pmax_ << " W, mean_clk "
       << (gpu_clk_sum_ / gpu_samples_) << " MHz, min_clk " << gpu_clk_min_ << " MHz over " << gpu_samples_
       << " samples\n";
  }
  for (const Engine& e : eng_) {
    long long it = 0, fa = 0, sw = 0, rs = 0, pl = 0, im = 0;
    uint32_t bmax = 0;
    for (const LSChainStats& s : e.stats) {
      it += s.iterations;
      fa += s.force_adds;
      sw += s.swaps;
      rs += s.restarts;
      pl += s.plateau;
      im += s.improvements;
      bmax = std::max(bmax, s.best_size);
    }
    sm << "state " << e.name << "   B=" << e.B << " best=" << bmax << " iters=" << it << " force_adds=" << fa
       << " swaps=" << sw << " device_restarts=" << rs << " plateau=" << pl << " improvements=" << im << "\n";
  }
  const std::string summary = sm.str();
  std::fputs(summary.c_str(), stdout);
  notes_ << summary;
  notes_.flush();
  std::ofstream(run_dir_ / "summary.txt", std::ios::binary) << summary;

  std::printf("RESULT ok=%d run=%s best=%u launches=%d iterations=%lld records=%d verify_fail=%d found=%d "
              "distinct=%zu bad_seeds=%d elapsed_s=%.1f stop=%s\n",
              (verify_fail_ == 0 && bad_seeds_ == 0) ? 1 : 0, c_.run_name.c_str(), best_overall_, launch_, iters_total_, records_,
              verify_fail_, found_written_, seen_.size(), bad_seeds_, total, stop_reason.c_str());
  return (verify_fail_ == 0 && bad_seeds_ == 0) ? 0 : 1;
}

void usage() {
  std::fprintf(stderr,
               "usage: gpu_mis <config.json> [options]\n"
               "  --set key=value      override a config key (repeatable; \"engine.stall_limit=2000\")\n"
               "  --resume             continue from <out_dir>/<run_name>/checkpoint.txt\n"
               "  --run-name NAME      name of the run directory\n"
               "  --out-dir DIR        parent of the run directory\n"
               "  --max-launches N     stop after N launches (counted across resumes)\n"
               "  --minutes M          wall-clock budget in minutes (0 = unlimited)\n"
               "  --seed S             RNG seed\n"
               "  --target T           output-slot threshold (default 497)\n"
               "  --selfcheck [N]      run lsa_check every N launches (default 10)\n"
               "  --dry-run            validate the config, create the run dir, exit\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::vector<std::string> echo(argv, argv + argc);
  if (argc < 2) {
    usage();
    std::printf("RESULT ok=0 reason=\"usage\"\n");
    return 2;
  }
  std::string cfg_path;
  std::vector<std::string> overrides;
  bool resume = false, dry = false, selfcheck = false;
  int selfcheck_n = 10;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto need = [&](const char* what) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing argument after ") + what);
        return argv[++i];
      };
      if (a == "--set") overrides.push_back(need("--set"));
      else if (a == "--resume") resume = true;
      else if (a == "--dry-run") dry = true;
      else if (a == "--run-name") overrides.push_back("run_name=" + need("--run-name"));
      else if (a == "--out-dir") overrides.push_back("out_dir=" + need("--out-dir"));
      else if (a == "--max-launches") overrides.push_back("max_launches=" + need("--max-launches"));
      else if (a == "--minutes") overrides.push_back("wall_clock_minutes=" + need("--minutes"));
      else if (a == "--seed") overrides.push_back("seed=" + need("--seed"));
      else if (a == "--target") overrides.push_back("target=" + need("--target"));
      else if (a == "--selfcheck") {
        selfcheck = true;
        if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0]))) selfcheck_n = std::atoi(argv[++i]);
      } else if (a == "-h" || a == "--help") {
        usage();
        return 0;
      } else if (!a.empty() && a[0] == '-') {
        throw std::runtime_error("unknown option " + a);
      } else if (cfg_path.empty()) {
        cfg_path = a;
      } else {
        throw std::runtime_error("unexpected argument " + a);
      }
    }
    if (cfg_path.empty()) throw std::runtime_error("no config file given");

    RunConfig c = kiss::load_run_config(cfg_path, overrides);
    if (selfcheck) c.selfcheck_every = selfcheck_n;
    if (c.verify_s.empty()) {
      const std::string self = kiss::self_exe_path();
      c.verify_s = self.empty()
                       ? kiss::exe("verify_s")
                       : (fs::path(self).parent_path() / kiss::exe("verify_s")).string();
    }
    kiss::validate_run_config(c);
    Driver d(std::move(c), echo);
    return d.run(resume, dry);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "gpu_mis: %s\n", ex.what());
    std::printf("RESULT ok=0 reason=\"%s\"\n", ex.what());
    return 1;
  }
}
