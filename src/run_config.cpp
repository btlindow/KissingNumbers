// T3.2c — hand-written JSON reader/writer and the run configuration of
// tools/gpu_mis (include/kiss/run_config.h). nlohmann/json is not installed on
// this machine and the config surface is small, so the parser is written here.
//
// Deliberate strictness: unknown keys throw (a typo in a config file must not
// silently fall back to a default), and every numeric field is range-checked in
// validate_run_config().

#include "kiss/run_config.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kiss {
namespace {

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error("run_config: " + msg); }

// ---------------------------------------------------------------------------
// JSON parser
// ---------------------------------------------------------------------------
class Parser {
 public:
  explicit Parser(const std::string& t) : t_(t) {}

  Json parse() {
    skip();
    Json j = value();
    skip();
    if (i_ != t_.size()) err("trailing characters after the top-level value");
    return j;
  }

 private:
  const std::string& t_;
  std::size_t i_ = 0;

  [[noreturn]] void err(const std::string& msg) const {
    die("JSON syntax error at byte " + std::to_string(i_) + ": " + msg);
  }
  bool eof() const { return i_ >= t_.size(); }
  char peek() const { return eof() ? '\0' : t_[i_]; }

  void skip() {
    for (;;) {
      while (!eof() && (t_[i_] == ' ' || t_[i_] == '\t' || t_[i_] == '\n' || t_[i_] == '\r')) ++i_;
      if (i_ + 1 < t_.size() && t_[i_] == '/' && t_[i_ + 1] == '/') {
        while (!eof() && t_[i_] != '\n') ++i_;
        continue;
      }
      if (i_ + 1 < t_.size() && t_[i_] == '/' && t_[i_ + 1] == '*') {
        i_ += 2;
        while (i_ + 1 < t_.size() && !(t_[i_] == '*' && t_[i_ + 1] == '/')) ++i_;
        if (i_ + 1 >= t_.size()) err("unterminated /* comment");
        i_ += 2;
        continue;
      }
      return;
    }
  }

  void expect(char c) {
    if (peek() != c) err(std::string("expected '") + c + "'");
    ++i_;
  }

  Json value() {
    skip();
    if (eof()) err("unexpected end of input");
    const char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') return Json::string(string_lit());
    if (c == 't') { literal("true"); return Json::boolean(true); }
    if (c == 'f') { literal("false"); return Json::boolean(false); }
    if (c == 'n') { literal("null"); return Json::null(); }
    return number();
  }

  void literal(const char* lit) {
    const std::size_t n = std::char_traits<char>::length(lit);
    if (t_.compare(i_, n, lit) != 0) err(std::string("expected ") + lit);
    i_ += n;
  }

  std::string string_lit() {
    expect('"');
    std::string out;
    while (true) {
      if (eof()) err("unterminated string");
      const char c = t_[i_++];
      if (c == '"') break;
      if (c != '\\') { out.push_back(c); continue; }
      if (eof()) err("unterminated escape");
      const char e = t_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (i_ + 4 > t_.size()) err("short \\u escape");
          unsigned cp = 0;
          for (int k = 0; k < 4; ++k) {
            const char h = t_[i_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
            else err("bad hex digit in \\u escape");
          }
          // UTF-8 encode (no surrogate-pair handling: config files are ASCII).
          if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
          } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: err("unknown escape");
      }
    }
    return out;
  }

  Json number() {
    const std::size_t start = i_;
    if (peek() == '-' || peek() == '+') ++i_;
    bool any = false;
    while (!eof() && std::isdigit(static_cast<unsigned char>(t_[i_]))) { ++i_; any = true; }
    if (!eof() && t_[i_] == '.') {
      ++i_;
      while (!eof() && std::isdigit(static_cast<unsigned char>(t_[i_]))) { ++i_; any = true; }
    }
    if (!any) err("expected a value");
    if (!eof() && (t_[i_] == 'e' || t_[i_] == 'E')) {
      ++i_;
      if (!eof() && (t_[i_] == '-' || t_[i_] == '+')) ++i_;
      while (!eof() && std::isdigit(static_cast<unsigned char>(t_[i_]))) ++i_;
    }
    return Json::number(std::strtod(t_.substr(start, i_ - start).c_str(), nullptr));
  }

  Json array() {
    expect('[');
    Json j = Json::array();
    skip();
    if (peek() == ']') { ++i_; return j; }
    for (;;) {
      j.arr.push_back(value());
      skip();
      if (peek() == ',') { ++i_; continue; }
      if (peek() == ']') { ++i_; return j; }
      err("expected ',' or ']'");
    }
  }

  Json object() {
    expect('{');
    Json j = Json::object();
    skip();
    if (peek() == '}') { ++i_; return j; }
    for (;;) {
      skip();
      const std::string key = string_lit();
      skip();
      expect(':');
      j.obj.emplace_back(key, value());
      skip();
      if (peek() == ',') { ++i_; continue; }
      if (peek() == '}') { ++i_; return j; }
      err("expected ',' or '}'");
    }
  }
};

std::string escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

std::string num_to_string(double v) {
  if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(v));
    return buf;
  }
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.10g", v);
  return buf;
}

void dump_into(const Json& j, int indent, int level, std::string& out) {
  const std::string pad(static_cast<std::size_t>(indent * level), ' ');
  const std::string pad1(static_cast<std::size_t>(indent * (level + 1)), ' ');
  switch (j.type) {
    case Json::Null: out += "null"; break;
    case Json::Bool: out += j.b ? "true" : "false"; break;
    case Json::Number: out += num_to_string(j.num); break;
    case Json::String: out += '"' + escape(j.str) + '"'; break;
    case Json::Array:
      if (j.arr.empty()) { out += "[]"; break; }
      out += "[\n";
      for (std::size_t k = 0; k < j.arr.size(); ++k) {
        out += pad1;
        dump_into(j.arr[k], indent, level + 1, out);
        out += (k + 1 < j.arr.size()) ? ",\n" : "\n";
      }
      out += pad + "]";
      break;
    case Json::Object:
      if (j.obj.empty()) { out += "{}"; break; }
      out += "{\n";
      for (std::size_t k = 0; k < j.obj.size(); ++k) {
        out += pad1 + '"' + escape(j.obj[k].first) + "\": ";
        dump_into(j.obj[k].second, indent, level + 1, out);
        out += (k + 1 < j.obj.size()) ? ",\n" : "\n";
      }
      out += pad + "}";
      break;
  }
}

// ---------------------------------------------------------------------------
// typed field access with "unknown key" bookkeeping
// ---------------------------------------------------------------------------
class Reader {
 public:
  explicit Reader(const Json& j, std::string where) : j_(j), where_(std::move(where)) {
    if (!j.is_object()) die(where_ + ": expected a JSON object");
  }
  ~Reader() = default;

  void get(const char* key, double& dst) {
    if (const Json* v = take(key)) {
      if (!v->is_number()) die(where_ + "." + key + ": expected a number");
      dst = v->num;
    }
  }
  void get(const char* key, int& dst) {
    double d = dst;
    get(key, d);
    dst = static_cast<int>(d);
  }
  void get(const char* key, uint64_t& dst) {
    if (const Json* v = take(key)) {
      if (!v->is_number()) die(where_ + "." + key + ": expected a number");
      if (v->num < 0) die(where_ + "." + key + ": must be >= 0");
      dst = static_cast<uint64_t>(v->num);
    }
  }
  void get(const char* key, bool& dst) {
    if (const Json* v = take(key)) {
      if (v->is_bool()) dst = v->b;
      else if (v->is_number()) dst = v->num != 0;
      else die(where_ + "." + key + ": expected true/false");
    }
  }
  void get(const char* key, std::string& dst) {
    if (const Json* v = take(key)) {
      if (!v->is_string()) die(where_ + "." + key + ": expected a string");
      dst = v->str;
    }
  }
  const Json* sub(const char* key) {
    const Json* v = take(key);
    if (!v) return nullptr;
    if (!v->is_object()) die(where_ + "." + key + ": expected an object");
    return v;
  }
  void finish() const {
    for (const auto& kv : j_.obj) {
      if (!seen_.count(kv.first)) {
        die(where_ + ": unknown key \"" + kv.first + "\"");
      }
    }
  }

 private:
  const Json* take(const char* key) {
    seen_.insert(key);
    return j_.find(key);
  }
  const Json& j_;
  std::string where_;
  std::set<std::string> seen_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Json members
// ---------------------------------------------------------------------------
const Json* Json::find(const std::string& key) const {
  if (type != Object) return nullptr;
  for (const auto& kv : obj)
    if (kv.first == key) return &kv.second;
  return nullptr;
}

Json& Json::set(const std::string& key, Json value) {
  type = Object;
  for (auto& kv : obj) {
    if (kv.first == key) {
      kv.second = std::move(value);
      return kv.second;
    }
  }
  obj.emplace_back(key, std::move(value));
  return obj.back().second;
}

double Json::get_number(const std::string& key, double dflt) const {
  const Json* v = find(key);
  if (!v || v->is_null()) return dflt;
  if (!v->is_number()) die("key \"" + key + "\": expected a number");
  return v->num;
}
int64_t Json::get_int(const std::string& key, int64_t dflt) const {
  return static_cast<int64_t>(get_number(key, static_cast<double>(dflt)));
}
bool Json::get_bool(const std::string& key, bool dflt) const {
  const Json* v = find(key);
  if (!v || v->is_null()) return dflt;
  if (v->is_bool()) return v->b;
  if (v->is_number()) return v->num != 0;
  die("key \"" + key + "\": expected true/false");
}
std::string Json::get_string(const std::string& key, const std::string& dflt) const {
  const Json* v = find(key);
  if (!v || v->is_null()) return dflt;
  if (!v->is_string()) die("key \"" + key + "\": expected a string");
  return v->str;
}

Json parse_json(const std::string& text) { return Parser(text).parse(); }

Json load_json(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  try {
    return parse_json(ss.str());
  } catch (const std::exception& e) {
    die(path.string() + ": " + e.what());
  }
}

std::string dump_json(const Json& j, int indent) {
  std::string out;
  dump_into(j, indent, 0, out);
  out.push_back('\n');
  return out;
}

void save_json(const std::filesystem::path& path, const Json& j) {
  std::ofstream out(path, std::ios::binary);
  if (!out) die("cannot write " + path.string());
  out << dump_json(j);
  if (!out) die("write failed: " + path.string());
}

// ---------------------------------------------------------------------------
// RunConfig
// ---------------------------------------------------------------------------
RunConfig run_config_from_json(const Json& j, const std::vector<std::string>& overrides) {
  RunConfig c;
  Reader r(j, "config");
  r.get("run_name", c.run_name);
  r.get("out_dir", c.out_dir);
  r.get("data_dir", c.data_dir);
  r.get("group_dir", c.group_dir);
  r.get("plateau_atoms", c.plateau_atoms);
  r.get("co0_dir", c.co0_dir);
  r.get("elite_in", c.elite_in);
  r.get("verify_s", c.verify_s);
  r.get("python", c.python);
  r.get("verify_py", c.verify_py);
  r.get("nvidia_smi", c.nvidia_smi);

  r.get("seed", c.seed);
  r.get("B", c.B);
  r.get("K", c.K);
  r.get("antipodal_fraction", c.antipodal_fraction);
  r.get("target", c.target);
  r.get("wall_clock_minutes", c.wall_clock_minutes);
  r.get("max_launches", c.max_launches);
  r.get("checkpoint_minutes", c.checkpoint_minutes);
  r.get("checkpoint_launches", c.checkpoint_launches);
  r.get("selfcheck_every", c.selfcheck_every);
  r.get("verify_seeds", c.verify_seeds);
  r.get("watchdog_seconds", c.watchdog_seconds);
  r.get("log_every", c.log_every);
  r.get("nslots", c.nslots);
  r.get("stop_on_record", c.stop_on_record);

  r.get("delete_frac_min", c.delete_frac_min);
  r.get("delete_frac_max", c.delete_frac_max);
  r.get("co0_count", c.co0_count);
  r.get("co0_slots", c.co0_slots);
  r.get("co0_burnin", c.co0_burnin);
  r.get("elite_cap", c.elite_cap);
  r.get("elite_min_size", c.elite_min_size);
  r.get("reseed_elite_prob", c.reseed_elite_prob);
  r.get("reseed_batch_frac", c.reseed_batch_frac);
  r.get("log_min_size", c.log_min_size);
  r.get("max_set_files", c.max_set_files);
  r.get("hist_per_launch", c.hist_per_launch);
  r.get("verify_per_launch", c.verify_per_launch);

  r.get("vram_min_gb", c.vram_min_gb);
  r.get("vram_retries", c.vram_retries);
  r.get("vram_retry_seconds", c.vram_retry_seconds);

  if (const Json* m = r.sub("seed_mix")) {
    Reader rm(*m, "config.seed_mix");
    rm.get("s496", c.mix.s496);
    rm.get("s488", c.mix.s488);
    rm.get("plateau496", c.mix.plateau496);
    rm.get("co0_496", c.mix.co0_496);
    rm.get("co0_488", c.mix.co0_488);
    rm.get("elite", c.mix.elite);
    rm.get("greedy", c.mix.greedy);
    rm.finish();
  }
  if (const Json* e = r.sub("engine")) {
    Reader re(*e, "config.engine");
    re.get("CL", c.engine.CL);
    re.get("tmin", c.engine.tmin);
    re.get("tmax", c.engine.tmax);
    re.get("list_tmax", c.engine.list_tmax);
    re.get("tenure", c.engine.tenure);
    re.get("tenure_jitter", c.engine.tenure_jitter);
    re.get("strength", c.engine.strength);
    re.get("select_scan", c.engine.select_scan);
    re.get("tournament_rounds", c.engine.tournament_rounds);
    re.get("greedy_pct", c.engine.greedy_pct);
    re.get("neigh_rounds", c.engine.neigh_rounds);
    re.get("uniform_rounds", c.engine.uniform_rounds);
    re.get("swap_enable", c.engine.swap_enable);
    re.get("max_x_tries", c.engine.max_x_tries);
    re.get("tabu_drain", c.engine.tabu_drain);
    re.get("max_drop", c.engine.max_drop);
    re.get("stall_limit", c.engine.stall_limit);
    re.get("restart_del", c.engine.restart_del);
    re.get("rescan_every", c.engine.rescan_every);
    re.get("accept_equal", c.engine.accept_equal);
    re.get("warps_per_block", c.engine.warps_per_block);
    re.get("FL", c.engine.FL);
    re.get("TABU", c.engine.TABU);
    re.get("force_cap", c.engine.force_cap);
    re.finish();
  }
  r.finish();

  for (const std::string& kv : overrides) apply_override(c, kv);
  return c;
}

RunConfig load_run_config(const std::filesystem::path& path, const std::vector<std::string>& overrides) {
  RunConfig c = run_config_from_json(load_json(path), overrides);
  c.source_file = path.string();
  return c;
}

namespace {
double to_num(const std::string& key, const std::string& v) {
  try {
    std::size_t pos = 0;
    const double d = std::stod(v, &pos);
    while (pos < v.size() && std::isspace(static_cast<unsigned char>(v[pos]))) ++pos;
    if (pos != v.size()) throw std::invalid_argument("trailing");
    return d;
  } catch (const std::exception&) {
    die("override " + key + "=" + v + ": not a number");
  }
}
bool to_bool(const std::string& key, const std::string& v) {
  if (v == "true" || v == "1" || v == "yes" || v == "on") return true;
  if (v == "false" || v == "0" || v == "no" || v == "off") return false;
  die("override " + key + "=" + v + ": not a boolean");
}
}  // namespace

void apply_override(RunConfig& c, const std::string& kv) {
  const std::size_t eq = kv.find('=');
  if (eq == std::string::npos) die("override \"" + kv + "\": expected key=value");
  const std::string k = kv.substr(0, eq);
  const std::string v = kv.substr(eq + 1);

#define KISS_STR(name)                    \
  if (k == #name) {                       \
    c.name = v;                           \
    return;                               \
  }
#define KISS_NUM(name)                            \
  if (k == #name) {                               \
    c.name = static_cast<decltype(c.name)>(to_num(k, v)); \
    return;                                       \
  }
#define KISS_BOOL(name)      \
  if (k == #name) {          \
    c.name = to_bool(k, v);  \
    return;                  \
  }
#define KISS_SUBNUM(sec, name)                                    \
  if (k == #sec "." #name) {                                      \
    c.sec.name = static_cast<decltype(c.sec.name)>(to_num(k, v)); \
    return;                                                       \
  }

  KISS_STR(run_name) KISS_STR(out_dir) KISS_STR(data_dir) KISS_STR(group_dir)
  KISS_STR(plateau_atoms) KISS_STR(co0_dir) KISS_STR(elite_in) KISS_STR(verify_s)
  KISS_STR(python) KISS_STR(verify_py) KISS_STR(nvidia_smi)

  KISS_NUM(seed) KISS_NUM(B) KISS_NUM(K) KISS_NUM(antipodal_fraction) KISS_NUM(target)
  KISS_NUM(wall_clock_minutes) KISS_NUM(max_launches) KISS_NUM(checkpoint_minutes)
  KISS_NUM(checkpoint_launches) KISS_NUM(selfcheck_every) KISS_NUM(watchdog_seconds)
  KISS_NUM(log_every) KISS_NUM(nslots)
  KISS_BOOL(stop_on_record) KISS_BOOL(verify_seeds)

  KISS_NUM(delete_frac_min) KISS_NUM(delete_frac_max) KISS_NUM(co0_count) KISS_NUM(co0_slots)
  KISS_NUM(co0_burnin) KISS_NUM(elite_cap) KISS_NUM(elite_min_size) KISS_NUM(reseed_elite_prob)
  KISS_NUM(reseed_batch_frac) KISS_NUM(log_min_size) KISS_NUM(max_set_files)
  KISS_NUM(hist_per_launch) KISS_NUM(verify_per_launch)
  KISS_NUM(vram_min_gb) KISS_NUM(vram_retries) KISS_NUM(vram_retry_seconds)

  KISS_SUBNUM(mix, s496) KISS_SUBNUM(mix, s488) KISS_SUBNUM(mix, plateau496)
  KISS_SUBNUM(mix, co0_496) KISS_SUBNUM(mix, co0_488) KISS_SUBNUM(mix, elite) KISS_SUBNUM(mix, greedy)
  KISS_SUBNUM(engine, CL) KISS_SUBNUM(engine, tmin) KISS_SUBNUM(engine, tmax)
  KISS_SUBNUM(engine, list_tmax) KISS_SUBNUM(engine, tenure) KISS_SUBNUM(engine, tenure_jitter)
  KISS_SUBNUM(engine, strength) KISS_SUBNUM(engine, select_scan) KISS_SUBNUM(engine, tournament_rounds)
  KISS_SUBNUM(engine, greedy_pct) KISS_SUBNUM(engine, neigh_rounds) KISS_SUBNUM(engine, uniform_rounds)
  KISS_SUBNUM(engine, swap_enable) KISS_SUBNUM(engine, max_x_tries) KISS_SUBNUM(engine, tabu_drain)
  KISS_SUBNUM(engine, max_drop) KISS_SUBNUM(engine, stall_limit) KISS_SUBNUM(engine, restart_del)
  KISS_SUBNUM(engine, rescan_every) KISS_SUBNUM(engine, accept_equal) KISS_SUBNUM(engine, warps_per_block)
  KISS_SUBNUM(engine, FL) KISS_SUBNUM(engine, TABU) KISS_SUBNUM(engine, force_cap)

#undef KISS_STR
#undef KISS_NUM
#undef KISS_BOOL
#undef KISS_SUBNUM

  // "seed_mix.x" is the JSON spelling of the C++ member "mix.x".
  if (k.rfind("seed_mix.", 0) == 0) {
    apply_override(c, "mix." + kv.substr(std::string("seed_mix.").size()));
    return;
  }
  die("override \"" + k + "\": unknown key");
}

Json run_config_to_json(const RunConfig& c) {
  Json j = Json::object();
  auto S = [&](const char* k, const std::string& v) { j.set(k, Json::string(v)); };
  auto Nu = [&](const char* k, double v) { j.set(k, Json::number(v)); };
  S("run_name", c.run_name);
  S("out_dir", c.out_dir);
  S("data_dir", c.data_dir);
  S("group_dir", c.group_dir);
  S("plateau_atoms", c.plateau_atoms);
  S("co0_dir", c.co0_dir);
  S("elite_in", c.elite_in);
  S("verify_s", c.verify_s);
  S("python", c.python);
  S("verify_py", c.verify_py);
  S("nvidia_smi", c.nvidia_smi);
  Nu("seed", static_cast<double>(c.seed));
  Nu("B", c.B);
  Nu("K", c.K);
  Nu("antipodal_fraction", c.antipodal_fraction);
  Nu("target", c.target);
  Nu("wall_clock_minutes", c.wall_clock_minutes);
  Nu("max_launches", c.max_launches);
  Nu("checkpoint_minutes", c.checkpoint_minutes);
  Nu("checkpoint_launches", c.checkpoint_launches);
  Nu("selfcheck_every", c.selfcheck_every);
  j.set("verify_seeds", Json::boolean(c.verify_seeds));
  Nu("watchdog_seconds", c.watchdog_seconds);
  Nu("log_every", c.log_every);
  Nu("nslots", c.nslots);
  j.set("stop_on_record", Json::boolean(c.stop_on_record));
  Nu("delete_frac_min", c.delete_frac_min);
  Nu("delete_frac_max", c.delete_frac_max);
  Nu("co0_count", c.co0_count);
  Nu("co0_slots", c.co0_slots);
  Nu("co0_burnin", c.co0_burnin);
  Nu("elite_cap", c.elite_cap);
  Nu("elite_min_size", c.elite_min_size);
  Nu("reseed_elite_prob", c.reseed_elite_prob);
  Nu("reseed_batch_frac", c.reseed_batch_frac);
  Nu("log_min_size", c.log_min_size);
  Nu("max_set_files", c.max_set_files);
  Nu("hist_per_launch", c.hist_per_launch);
  Nu("verify_per_launch", c.verify_per_launch);
  Nu("vram_min_gb", c.vram_min_gb);
  Nu("vram_retries", c.vram_retries);
  Nu("vram_retry_seconds", c.vram_retry_seconds);

  Json m = Json::object();
  m.set("s496", Json::number(c.mix.s496));
  m.set("s488", Json::number(c.mix.s488));
  m.set("plateau496", Json::number(c.mix.plateau496));
  m.set("co0_496", Json::number(c.mix.co0_496));
  m.set("co0_488", Json::number(c.mix.co0_488));
  m.set("elite", Json::number(c.mix.elite));
  m.set("greedy", Json::number(c.mix.greedy));
  j.set("seed_mix", m);

  Json e = Json::object();
  e.set("CL", Json::number(c.engine.CL));
  e.set("tmin", Json::number(c.engine.tmin));
  e.set("tmax", Json::number(c.engine.tmax));
  e.set("list_tmax", Json::number(c.engine.list_tmax));
  e.set("tenure", Json::number(c.engine.tenure));
  e.set("tenure_jitter", Json::number(c.engine.tenure_jitter));
  e.set("strength", Json::number(c.engine.strength));
  e.set("select_scan", Json::number(c.engine.select_scan));
  e.set("tournament_rounds", Json::number(c.engine.tournament_rounds));
  e.set("greedy_pct", Json::number(c.engine.greedy_pct));
  e.set("neigh_rounds", Json::number(c.engine.neigh_rounds));
  e.set("uniform_rounds", Json::number(c.engine.uniform_rounds));
  e.set("swap_enable", Json::number(c.engine.swap_enable));
  e.set("max_x_tries", Json::number(c.engine.max_x_tries));
  e.set("tabu_drain", Json::number(c.engine.tabu_drain));
  e.set("max_drop", Json::number(c.engine.max_drop));
  e.set("stall_limit", Json::number(c.engine.stall_limit));
  e.set("restart_del", Json::number(c.engine.restart_del));
  e.set("rescan_every", Json::number(c.engine.rescan_every));
  e.set("accept_equal", Json::number(c.engine.accept_equal));
  e.set("warps_per_block", Json::number(c.engine.warps_per_block));
  e.set("FL", Json::number(c.engine.FL));
  e.set("TABU", Json::number(c.engine.TABU));
  e.set("force_cap", Json::number(c.engine.force_cap));
  j.set("engine", e);
  return j;
}

void validate_run_config(const RunConfig& c) {
  auto req = [](bool cond, const std::string& msg) {
    if (!cond) die("invalid config: " + msg);
  };
  req(c.B >= 1, "B must be >= 1");
  req(c.K >= 1, "K must be >= 1");
  req(c.target >= 1, "target must be >= 1");
  req(c.antipodal_fraction >= 0.0 && c.antipodal_fraction <= 1.0, "antipodal_fraction must be in [0,1]");
  req(c.wall_clock_minutes >= 0, "wall_clock_minutes must be >= 0");
  req(c.max_launches >= 0, "max_launches must be >= 0");
  req(c.checkpoint_minutes >= 0, "checkpoint_minutes must be >= 0");
  req(c.checkpoint_launches >= 0, "checkpoint_launches must be >= 0");
  req(c.log_every >= 1, "log_every must be >= 1");
  req(c.nslots >= 0, "nslots must be >= 0");
  req(c.delete_frac_min >= 0.0 && c.delete_frac_min <= 0.9, "delete_frac_min must be in [0,0.9]");
  req(c.delete_frac_max >= c.delete_frac_min && c.delete_frac_max <= 0.9,
      "delete_frac_max must be in [delete_frac_min, 0.9]");
  req(c.elite_cap >= 1, "elite_cap must be >= 1");
  req(c.elite_min_size >= 0, "elite_min_size must be >= 0");
  req(c.reseed_elite_prob >= 0.0 && c.reseed_elite_prob <= 1.0, "reseed_elite_prob must be in [0,1]");
  req(c.reseed_batch_frac > 0.0 && c.reseed_batch_frac <= 1.0, "reseed_batch_frac must be in (0,1]");
  req(c.co0_count >= 0, "co0_count must be >= 0");
  req(c.co0_slots >= 2, "co0_slots must be >= 2");
  req(c.hist_per_launch >= 0, "hist_per_launch must be >= 0");
  req(c.verify_per_launch >= 0, "verify_per_launch must be >= 0");
  req(c.vram_min_gb >= 0, "vram_min_gb must be >= 0");

  const double sum = c.mix.s496 + c.mix.s488 + c.mix.plateau496 + c.mix.co0_496 + c.mix.co0_488 +
                     c.mix.elite + c.mix.greedy;
  req(sum > 0, "seed_mix: all weights are zero");
  req(c.mix.s496 >= 0 && c.mix.s488 >= 0 && c.mix.plateau496 >= 0 && c.mix.co0_496 >= 0 &&
          c.mix.co0_488 >= 0 && c.mix.elite >= 0 && c.mix.greedy >= 0,
      "seed_mix: weights must be >= 0");

  const EngineParams& e = c.engine;
  req(e.CL >= 32, "engine.CL must be >= 32");
  req(e.tmin >= 1 && e.tmax >= e.tmin, "engine.tmin/tmax");
  req(e.list_tmax >= 1 && e.list_tmax <= 16, "engine.list_tmax must be in [1,16]");
  req(e.tenure >= 0 && e.tenure_jitter >= 0, "engine.tenure/tenure_jitter must be >= 0");
  req(e.strength >= 1, "engine.strength must be >= 1");
  req(e.greedy_pct >= 0 && e.greedy_pct <= 100, "engine.greedy_pct must be in [0,100]");
  req(e.max_drop >= 0, "engine.max_drop must be >= 0");
  req(e.stall_limit >= 1, "engine.stall_limit must be >= 1");
  req(e.restart_del >= 0, "engine.restart_del must be >= 0");
  req(e.rescan_every >= 1, "engine.rescan_every must be >= 1");
  req(e.warps_per_block >= 1 && e.warps_per_block <= 16, "engine.warps_per_block must be in [1,16]");
  req(e.FL >= 64, "engine.FL must be >= 64");
  req(e.TABU >= 32 && e.TABU % 32 == 0, "engine.TABU must be a positive multiple of 32");
  req(e.force_cap >= 1 && e.force_cap <= 64, "engine.force_cap must be in [1,64]");
}

}  // namespace kiss
