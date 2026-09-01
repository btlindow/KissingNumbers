// T3.2c — run configuration of the GPU MIS driver (tools/gpu_mis) and a
// dependency-free JSON reader/writer (nlohmann/json is not installed here).
//
// The config is a JSON object with at most one level of nesting ("seed_mix"
// and "engine" are objects); every key has a default, unknown keys are an
// error (typos must not silently fall back to defaults). Command-line
// overrides use the dotted form "engine.stall_limit=2000" or "B=64".
//
// CPU-only (libkiss): no CUDA types here; tools/gpu_mis.cpp maps RunConfig
// onto kiss::cuda::LSParams / LSSearchParams.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace kiss {

// ---------------------------------------------------------------------------
// Minimal JSON value (null, bool, number, string, array, object). Objects keep
// insertion order (vector of pairs) so dumps are stable.
// ---------------------------------------------------------------------------
struct Json {
  enum Type { Null, Bool, Number, String, Array, Object };
  Type type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  static Json null() { return Json{}; }
  static Json boolean(bool v) { Json j; j.type = Bool; j.b = v; return j; }
  static Json number(double v) { Json j; j.type = Number; j.num = v; return j; }
  static Json string(const std::string& v) { Json j; j.type = String; j.str = v; return j; }
  static Json array() { Json j; j.type = Array; return j; }
  static Json object() { Json j; j.type = Object; return j; }

  bool is_null() const { return type == Null; }
  bool is_object() const { return type == Object; }
  bool is_array() const { return type == Array; }
  bool is_number() const { return type == Number; }
  bool is_string() const { return type == String; }
  bool is_bool() const { return type == Bool; }

  // Object access: nullptr if absent (or not an object).
  const Json* find(const std::string& key) const;
  Json& set(const std::string& key, Json value);   // insert or replace
  void push(Json value) { arr.push_back(std::move(value)); }

  // Convenience getters with a default; throw std::runtime_error on a type
  // mismatch (a string where a number is expected, ...).
  double get_number(const std::string& key, double dflt) const;
  int64_t get_int(const std::string& key, int64_t dflt) const;
  bool get_bool(const std::string& key, bool dflt) const;
  std::string get_string(const std::string& key, const std::string& dflt) const;
};

// Parser: throws std::runtime_error with the byte offset on a syntax error.
// Accepts standard JSON (plus "//" line comments, handy in config files).
Json parse_json(const std::string& text);
Json load_json(const std::filesystem::path& path);
// Serialiser: pretty-printed with `indent` spaces per level; numbers that are
// integral print without a decimal point.
std::string dump_json(const Json& j, int indent = 2);
void save_json(const std::filesystem::path& path, const Json& j);

// ---------------------------------------------------------------------------
// Run configuration
// ---------------------------------------------------------------------------
struct SeedMix {                 // fractions (normalised at use; unavailable kinds are skipped)
  double s496 = 0.30;            // the 496 minus a random 5–15 %
  double s488 = 0.10;            // the 488 likewise
  double plateau496 = 0.20;      // a random non-empty subset of the six plateau atoms applied to the 496, minus 5–15 %
  double co0_496 = 0.20;         // g · (random plateau image of the 496), g a random Co_0 element, minus 5–15 %
  double co0_488 = 0.05;         // g · 488 likewise
  double elite = 0.05;           // a random elite-pool set (previous best-ever) minus 5–15 %
  double greedy = 0.10;          // random greedy maximal set (~230), diversity only
};

struct EngineParams {            // mirrors kiss::cuda::LSSearchParams (T3.2b recommended defaults)
  int CL = 4096;
  int tmin = 1, tmax = 8, list_tmax = 8;
  int tenure = 8, tenure_jitter = 4;
  int strength = 1;
  int select_scan = 1, tournament_rounds = 4, greedy_pct = 100;
  int neigh_rounds = 2, uniform_rounds = 8;
  int swap_enable = 1, max_x_tries = 4, tabu_drain = 1;
  int max_drop = 24;
  int stall_limit = 2000;
  int restart_del = 8;
  int rescan_every = 8;
  int accept_equal = 1;
  int warps_per_block = 8;
  int FL = 4096, TABU = 32, force_cap = 64;   // T3.2a LSParams
};

struct RunConfig {
  // identity / paths
  std::string run_name;                    // "" -> "<config stem>_<utc>"
  std::string out_dir = "runs/gpu_mis";    // run directory = out_dir/run_name
  std::string data_dir = "data";
  std::string group_dir;                   // "" -> data_dir/group
  std::string plateau_atoms = "runs/ls_search/plateau_atoms_496.json";
  std::string co0_dir;                     // optional: load aut_*.u32 (tools/random_aut) instead of generating
  std::string elite_in;                    // optional: directory of set files to preload the elite pool
  std::string verify_s;                    // "" -> <dir of gpu_mis>/verify_s
  std::string python = ".venv/bin/python";
  std::string verify_py = "python/verify_S.py";
  std::string nvidia_smi = "nvidia-smi";

  // search size / budget
  uint64_t seed = 1;
  int B = 1024;                            // total chains (both modes)
  int K = 50;                              // ILS iterations per launch
  double antipodal_fraction = 0.75;        // share of chains in antipodal mode (two state objects if 0 < f < 1)
  int target = 497;                        // output-slot threshold (custody chain)
  double wall_clock_minutes = 30;          // 0 = unlimited
  int max_launches = 0;                    // 0 = unlimited (counts launches of the whole run, across resumes)
  double checkpoint_minutes = 5;           // 0 = only at exit
  int checkpoint_launches = 0;             // additionally every N launches (tests)
  int selfcheck_every = 0;                 // lsa_check every N launches (0 = off)
  bool verify_seeds = true;                // check every set handed to lsa_init_from_sets (distinct,
                                           // pairwise non-conflicting) — an epoch boundary must never
                                           // feed a broken set to the device
  int watchdog_seconds = 60;               // nvidia-smi sampling period (0 = off)
  int log_every = 1;                       // CSV line every N launches
  int nslots = 0;                          // output slots per state (0 = B of that state)
  bool stop_on_record = false;             // stop once a set >= target is verified by both verifiers

  // seeding / reseeding
  SeedMix mix;
  double delete_frac_min = 0.05, delete_frac_max = 0.15;
  int co0_count = 64, co0_slots = 10, co0_burnin = 200;
  int elite_cap = 256;
  int elite_min_size = 400;                // sets below this never enter the pool
  double reseed_elite_prob = 0.5;          // restart: elite pool (with deletions) vs the seed mix
  double reseed_batch_frac = 0.125;        // a host reseed pass (an "epoch boundary": every chain is
                                           // re-initialised from its best set, the stalled ones from the
                                           // elite pool / the seed mix) runs once this fraction of a
                                           // state's chains has raised restart_req
  int log_min_size = 490;                  // every distinct best set >= this is logged with its tightness histogram
  int max_set_files = 4096;                // ... and saved to <run>/sets/ (up to this many files)
  int hist_per_launch = 256;               // tightness histograms computed per launch (rest queued)
  int verify_per_launch = 4;               // custody-chain verifications per launch (rest queued; drained at exit)

  // GPU sharing
  double vram_min_gb = 5.5;
  int vram_retries = 12, vram_retry_seconds = 15;

  EngineParams engine;

  // Original JSON (for the effective-config dump) and the file it came from.
  std::string source_file;
};

// Parse a config file (JSON); unknown keys throw. `overrides` are "key=value"
// or "section.key=value" strings applied afterwards (CLI --set).
RunConfig load_run_config(const std::filesystem::path& path, const std::vector<std::string>& overrides = {});
RunConfig run_config_from_json(const Json& j, const std::vector<std::string>& overrides = {});
void apply_override(RunConfig& c, const std::string& kv);
Json run_config_to_json(const RunConfig& c);
// Throws std::runtime_error on an inconsistent config (B < 1, fractions out of range, ...).
void validate_run_config(const RunConfig& c);

}  // namespace kiss
