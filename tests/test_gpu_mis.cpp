// T3.2c — acceptance test of the host driver tools/gpu_mis and of the config
// reader src/run_config.cpp.
//
// Stages
//   json      : parser / dumper / override / validation unit tests (no GPU)
//   custody   : target = 490, seeds = 480-subsets of the 496, B = 64 — the output
//               flag must fire, a file must appear in <run>/found/, and BOTH
//               verifiers must accept it with the same size >= 490 (the test
//               re-runs them itself, it does not trust the driver's log)
//   resume    : --max-launches 4 + --resume up to 8 reproduces the sizes_hash of
//               an uninterrupted 8-launch run, launch by launch
//   soak      : a 3-minute run with --selfcheck must exit ok with no VERIFY-FAIL
//               and a well-formed CSV log
//
// Exit 0 on success, 77 (ctest SKIP_RETURN_CODE) without a CUDA device, without
// data/adj.u32 or when another process holds the card.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "adjacency_cuda.h"
#include "kiss/run_config.h"

namespace fs = std::filesystem;

namespace {

constexpr int kSkip = 77;
int g_failures = 0;
fs::path g_exe_dir, g_tool, g_root;

void fail(const std::string& msg) {
  std::printf("FAIL: %s\n", msg.c_str());
  std::fflush(stdout);
  ++g_failures;
}
void check(bool cond, const std::string& msg) {
  if (!cond) fail(msg);
}

std::string gbstr(std::size_t b) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f GB", static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0));
  return buf;
}

int run_capture(const std::string& cmd, std::string& out) {
  out.clear();
  FILE* p = popen((cmd + " 2>&1").c_str(), "r");
  if (!p) return -1;
  char buf[4096];
  while (std::fgets(buf, sizeof buf, p)) out += buf;
  const int rc = pclose(p);
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

std::string last_result_line(const std::string& out) {
  const std::size_t pos = out.rfind("RESULT ");
  if (pos == std::string::npos) return "";
  const std::size_t nl = out.find('\n', pos);
  return out.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
}

long field(const std::string& line, const std::string& key, long dflt = -1) {
  const std::size_t p = line.find(key + "=");
  if (p == std::string::npos) return dflt;
  return std::atol(line.c_str() + p + key.size() + 1);
}

// ---------------------------------------------------------------------------
// stage 1: the JSON reader / config
// ---------------------------------------------------------------------------
void stage_json() {
  std::printf("\n== stage json ==\n");
  // round trip
  const std::string txt = R"({
    // a comment
    "a": 1, "b": -2.5, "c": "x\ny", "d": [1, 2, {"e": true}], "f": null
  })";
  const kiss::Json j = kiss::parse_json(txt);
  check(j.is_object(), "json: top level is an object");
  check(j.get_int("a", 0) == 1, "json: a == 1");
  check(j.get_number("b", 0) == -2.5, "json: b == -2.5");
  check(j.get_string("c", "") == "x\ny", "json: escapes");
  const kiss::Json* d = j.find("d");
  check(d && d->is_array() && d->arr.size() == 3, "json: array size");
  check(d && d->arr[2].get_bool("e", false), "json: nested object");
  const kiss::Json* f = j.find("f");
  check(f && f->is_null(), "json: null");
  const kiss::Json j2 = kiss::parse_json(kiss::dump_json(j));
  check(j2.get_int("a", 0) == 1 && j2.get_string("c", "") == "x\ny", "json: dump/parse round trip");

  bool threw = false;
  try {
    kiss::parse_json("{\"a\": }");
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "json: syntax error throws");

  // config: defaults, unknown key, overrides, validation
  const fs::path tmp = fs::temp_directory_path() / "kiss_t32c_cfg.json";
  {
    std::ofstream o(tmp);
    o << "{ \"B\": 128, \"K\": 7, \"engine\": { \"stall_limit\": 1234 } }\n";
  }
  const kiss::RunConfig c = kiss::load_run_config(tmp, {});
  check(c.B == 128 && c.K == 7 && c.engine.stall_limit == 1234, "config: values read");
  check(c.target == 497, "config: target defaults to 497");
  check(c.engine.tmax == 8 && c.engine.CL == 4096, "config: engine defaults preserved");

  const kiss::RunConfig c2 =
      kiss::load_run_config(tmp, {"B=64", "engine.stall_limit=2000", "seed_mix.greedy=0.5", "run_name=zz"});
  check(c2.B == 64 && c2.engine.stall_limit == 2000 && c2.mix.greedy == 0.5 && c2.run_name == "zz",
        "config: overrides (incl. seed_mix. alias)");

  threw = false;
  try {
    std::ofstream o(tmp);
    o << "{ \"Bee\": 1 }\n";
    o.close();
    kiss::load_run_config(tmp, {});
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "config: unknown key is an error");

  threw = false;
  try {
    kiss::load_run_config(tmp, {});
  } catch (const std::exception&) {
    threw = true;
  }
  (void)threw;
  {
    std::ofstream o(tmp);
    o << "{ \"engine\": { \"nope\": 1 } }\n";
  }
  threw = false;
  try {
    kiss::load_run_config(tmp, {});
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "config: unknown nested key is an error");

  {
    std::ofstream o(tmp);
    o << "{ \"antipodal_fraction\": 1.5 }\n";
  }
  threw = false;
  try {
    kiss::validate_run_config(kiss::load_run_config(tmp, {}));
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "config: validation rejects antipodal_fraction = 1.5");

  {
    std::ofstream o(tmp);
    o << "{}\n";
  }
  threw = false;
  try {
    kiss::validate_run_config(kiss::load_run_config(tmp, {"engine.TABU=33"}));
  } catch (const std::exception&) {
    threw = true;
  }
  check(threw, "config: validation rejects TABU = 33");
  fs::remove(tmp);

  // the shipped configs must load and validate
  for (const char* name : {"configs/default.json", "configs/record_neighbourhood.json", "configs/soak_10min.json"}) {
    try {
      kiss::RunConfig cc = kiss::load_run_config(name, {});
      kiss::validate_run_config(cc);
      std::printf("config    : %s ok (B=%d K=%d antipodal=%.2f target=%d stall=%d)\n", name, cc.B, cc.K,
                  cc.antipodal_fraction, cc.target, cc.engine.stall_limit);
    } catch (const std::exception& e) {
      fail(std::string(name) + ": " + e.what());
    }
  }
}

// ---------------------------------------------------------------------------
// stage 2: custody chain
// ---------------------------------------------------------------------------
void stage_custody() {
  std::printf("\n== stage custody (target 490, 480-subsets of the 496, B = 64) ==\n");
  const fs::path run = fs::path("runs/gpu_mis/test_custody");
  fs::remove_all(run);
  const std::string cmd = g_tool.string() +
                          " configs/soak_10min.json --run-name test_custody --set B=64 --set K=20"
                          " --target 490 --set delete_frac_min=0.032 --set delete_frac_max=0.033"
                          " --set seed_mix.s496=1 --set seed_mix.s488=0 --set seed_mix.plateau496=0"
                          " --set seed_mix.elite=0 --set seed_mix.greedy=0 --set checkpoint_minutes=0"
                          " --set hist_per_launch=8 --set verify_per_launch=4 --set watchdog_seconds=0"
                          " --max-launches 3 --minutes 0";
  std::printf("$ %s\n", cmd.c_str());
  std::string out;
  const int rc = run_capture(cmd, out);
  std::fputs(out.c_str(), stdout);
  const std::string res = last_result_line(out);
  check(rc == 0, "custody: gpu_mis exit code 0 (got " + std::to_string(rc) + ")");
  check(field(res, "ok") == 1, "custody: RESULT ok=1");
  check(field(res, "verify_fail") == 0, "custody: no VERIFY-FAIL");
  check(field(res, "found") >= 1, "custody: at least one set written to found/");
  check(out.find("candidate |S|") != std::string::npos, "custody: candidate banner printed");

  // The test re-runs both verifiers itself on every file in found/.
  int files = 0;
  for (const auto& de : fs::directory_iterator(run / "found")) {
    if (!de.is_regular_file()) continue;
    ++files;
    std::string o1, o2;
    const int r1 = run_capture((g_exe_dir / ".." / "tools" / "verify_s").string() + " " + de.path().string() +
                                   " --data data",
                               o1);
    const int r2 = run_capture(".venv/bin/python python/verify_S.py " + de.path().string(), o2);
    const std::string l1 = last_result_line(o1), l2 = last_result_line(o2);
    std::printf("  %s\n    verify_s : %s\n    verify_S.py: %s\n", de.path().filename().c_str(), l1.c_str(), l2.c_str());
    check(r1 == 0 && field(l1, "ok") == 1, "custody: verify_s accepts " + de.path().string());
    check(r2 == 0 && field(l2, "ok") == 1, "custody: verify_S.py accepts " + de.path().string());
    check(field(l1, "size") == field(l2, "size"), "custody: both verifiers report the same size");
    check(field(l1, "size") >= 490, "custody: size >= 490");
  }
  check(files >= 1, "custody: found/ is non-empty");
  check(fs::exists(run / "sets.csv"), "custody: sets.csv written");
  // the tightness histogram of the reference 496 must be the known one (see docs/reports/)
  std::ifstream sc(run / "sets.csv");
  std::string line;
  bool saw_hist = false, saw_ref = false;
  while (std::getline(sc, line)) {
    if (line.find(",hist,") == std::string::npos) continue;
    saw_hist = true;
    if (line.find("4:80|6:640|7:256|8:2704|9:8064|10:31424|11:52672|12:49552|13:31360|14:13440|15:2560|16:1480|"
                  "17:256|18:704|19:64|20:528|22:128|24:152") != std::string::npos)
      saw_ref = true;
  }
  check(saw_hist, "custody: tightness histograms logged");
  check(saw_ref, "custody: the reference 496's tightness histogram matches the documented reference histogram");
}

// ---------------------------------------------------------------------------
// stage 3: checkpoint / resume determinism
// ---------------------------------------------------------------------------
std::vector<std::string> log_column(const fs::path& csv, const std::string& col) {
  std::ifstream in(csv);
  std::string line;
  std::vector<std::string> out;
  int idx = -1;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::istringstream is(line);
    std::string tok;
    while (std::getline(is, tok, ',')) f.push_back(tok);
    if (idx < 0) {
      for (std::size_t i = 0; i < f.size(); ++i)
        if (f[i] == col) idx = static_cast<int>(i);
      continue;
    }
    if (idx >= 0 && idx < static_cast<int>(f.size())) out.push_back(f[static_cast<std::size_t>(idx)]);
  }
  return out;
}

void stage_resume() {
  std::printf("\n== stage resume (--max-launches 4 + --resume vs an uninterrupted run) ==\n");
  const std::string common =
      " configs/soak_10min.json --set B=64 --set K=20 --set checkpoint_minutes=0 --set checkpoint_launches=4"
      " --set watchdog_seconds=0 --set hist_per_launch=0 --set selfcheck_every=0 --minutes 0";
  fs::remove_all("runs/gpu_mis/test_detA");
  fs::remove_all("runs/gpu_mis/test_detB");
  std::string o;
  int rc = run_capture(g_tool.string() + common + " --run-name test_detA --max-launches 8", o);
  check(rc == 0, "resume: reference run exit 0");
  rc = run_capture(g_tool.string() + common + " --run-name test_detB --max-launches 4", o);
  check(rc == 0, "resume: interrupted run exit 0");
  check(fs::exists("runs/gpu_mis/test_detB/checkpoint.txt"), "resume: checkpoint.txt written");
  rc = run_capture(g_tool.string() + common + " --run-name test_detB --max-launches 8 --resume", o);
  check(rc == 0, "resume: resumed run exit 0");

  const std::vector<std::string> a = log_column("runs/gpu_mis/test_detA/log.csv", "sizes_hash");
  const std::vector<std::string> b = log_column("runs/gpu_mis/test_detB/log.csv", "sizes_hash");
  const std::vector<std::string> am = log_column("runs/gpu_mis/test_detA/log.csv", "mean_best");
  const std::vector<std::string> bm = log_column("runs/gpu_mis/test_detB/log.csv", "mean_best");
  check(a.size() == 8 && b.size() == 8, "resume: 8 CSV rows each (got " + std::to_string(a.size()) + "/" +
                                            std::to_string(b.size()) + ")");
  bool same = a.size() == b.size();
  for (std::size_t i = 0; same && i < a.size(); ++i) same = (a[i] == b[i]) && (am[i] == bm[i]);
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    std::printf("  launch %zu  A %s / %s   B %s / %s%s\n", i + 1, a[i].c_str(), am[i].c_str(), b[i].c_str(),
                bm[i].c_str(), a[i] == b[i] && am[i] == bm[i] ? "" : "   <-- DIFFERS");
  check(same, "resume: the resumed run reproduces every launch's sizes_hash and mean_best");
  // the first launches must be identical too — that is the fixed-seed determinism check
  check(a.size() >= 1 && b.size() >= 1 && a[0] == b[0], "determinism: identical first-launch sizes");
}

// ---------------------------------------------------------------------------
// stage 4: soak
// ---------------------------------------------------------------------------
void stage_soak(double minutes) {
  std::printf("\n== stage soak (%.1f min, --selfcheck) ==\n", minutes);
  fs::remove_all("runs/gpu_mis/test_soak");
  char mins[32];
  std::snprintf(mins, sizeof mins, "%g", minutes);
  const std::string cmd = g_tool.string() + " configs/soak_10min.json --run-name test_soak --minutes " + mins +
                          " --selfcheck 20 --set checkpoint_minutes=1 --set watchdog_seconds=20";
  std::printf("$ %s\n", cmd.c_str());
  std::string out;
  const int rc = run_capture(cmd, out);
  // print the tail only
  std::size_t start = out.size() > 3000 ? out.size() - 3000 : 0;
  std::printf("...%s\n", out.substr(start).c_str());
  const std::string res = last_result_line(out);
  check(rc == 0, "soak: exit 0");
  check(field(res, "ok") == 1, "soak: RESULT ok=1");
  check(field(res, "verify_fail") == 0, "soak: no VERIFY-FAIL");
  check(out.find("VERIFY-FAIL") == std::string::npos, "soak: no VERIFY-FAIL banner");
  check(out.find("INVARIANT FAILURE") == std::string::npos, "soak: no invariant failure");
  check(out.find("selfcheck :") != std::string::npos, "soak: --selfcheck ran");
  check(out.find("errors=0") != std::string::npos, "soak: selfcheck reported 0 errors");
  check(field(res, "launches") > 0, "soak: at least one launch");

  const fs::path run("runs/gpu_mis/test_soak");
  check(fs::exists(run / "log.csv") && fs::exists(run / "summary.txt") && fs::exists(run / "checkpoint.txt") &&
            fs::exists(run / "config_effective.json"),
        "soak: log.csv / summary.txt / checkpoint.txt / config_effective.json written");
  const std::vector<std::string> t = log_column(run / "log.csv", "t_s");
  const std::vector<std::string> best = log_column(run / "log.csv", "best");
  check(!t.empty(), "soak: CSV has rows");
  bool mono = true;
  for (std::size_t i = 1; i < t.size(); ++i)
    if (std::atof(t[i].c_str()) < std::atof(t[i - 1].c_str())) mono = false;
  check(mono, "soak: CSV timestamps are non-decreasing");
  long bmax = 0;
  for (const std::string& s : best) bmax = std::max(bmax, std::atol(s.c_str()));
  std::printf("soak      : %zu CSV rows, best %ld, %ld launches\n", t.size(), bmax, field(res, "launches"));
  check(bmax >= 400, "soak: reaches at least 400 (seeds contain the records)");
  const std::vector<std::string> gpu = log_column(run / "gpu.csv", "temperature_c");
  check(!gpu.empty(), "soak: watchdog sampled nvidia-smi");
}

}  // namespace

int main(int argc, char** argv) {
  double soak_minutes = 3.0;
  bool small = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--soak-minutes" && i + 1 < argc) soak_minutes = std::atof(argv[++i]);
    else if (a == "--small") { small = true; soak_minutes = 0.25; }
  }
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) {
    std::printf("RESULT ok=0 reason=\"cannot locate own executable\"\n");
    return 1;
  }
  buf[n] = '\0';
  g_exe_dir = fs::path(buf).parent_path();
  g_tool = g_exe_dir / ".." / "tools" / "gpu_mis";
  g_root = fs::current_path();
  std::printf("test_gpu_mis: tool %s, cwd %s, soak %.2f min%s\n", g_tool.c_str(), g_root.c_str(), soak_minutes,
              small ? " (small)" : "");

  stage_json();

  if (!fs::exists(g_tool)) {
    std::printf("test_gpu_mis: %s not built — skipping the GPU stages\n", g_tool.c_str());
    std::printf("RESULT ok=%d skipped=1 reason=no_tool\n", g_failures == 0 ? 1 : 0);
    return g_failures == 0 ? kSkip : 1;
  }
  if (!fs::exists("data/adj.u32")) {
    std::printf("test_gpu_mis: data/adj.u32 missing (run tools/build_adj) — skipping\n");
    std::printf("RESULT ok=%d skipped=1 reason=no_adjacency\n", g_failures == 0 ? 1 : 0);
    return g_failures == 0 ? kSkip : 1;
  }
  {
    const std::size_t need = static_cast<std::size_t>(5.5 * 1024 * 1024 * 1024);
    std::size_t fr = 0, to = 0;
    for (int a = 0; a < 12; ++a) {
      kiss::cuda::device_mem_info(&fr, &to);
      if (fr >= need) break;
      std::printf("test_gpu_mis: only %s of %s free (need 5.5 GB) — retry %d/12 in 15 s\n", gbstr(fr).c_str(),
                  gbstr(to).c_str(), a + 1);
      std::fflush(stdout);
      std::this_thread::sleep_for(std::chrono::seconds(15));
    }
    if (fr < need) {
      std::printf("test_gpu_mis: %s free of %s — skipping (another process holds the GPU)\n", gbstr(fr).c_str(),
                  gbstr(to).c_str());
      std::printf("RESULT ok=%d skipped=1 reason=vram\n", g_failures == 0 ? 1 : 0);
      return g_failures == 0 ? kSkip : 1;
    }
  }

  stage_custody();
  stage_resume();
  stage_soak(soak_minutes);

  std::printf("\nRESULT ok=%d failures=%d soak_minutes=%.2f\n", g_failures == 0 ? 1 : 0, g_failures, soak_minutes);
  return g_failures == 0 ? 0 : 1;
}
