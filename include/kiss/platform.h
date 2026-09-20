// The OS services this project asks for outside the C++ standard library:
// running a child process and capturing its output, locating our own
// executable, a process id for temp-file names, and a thread-safe UTC
// breakdown of a time_t.
//
// They exist because of the custody chain in docs/design.md §2.4: the search
// driver writes a found set to disk and then re-verifies it by *running*
// tools/verify_s and python/verify_S.py as subprocesses, which means it has to
// find its own directory and read a child's stdout.
//
// Implemented for POSIX and for Win32. Header-only; <windows.h> is pulled in
// only on Windows and only with the lean macros set, because several
// translation units that need this also define `min`/`max`-shaped names.
#pragma once

#include <cstdio>
#include <ctime>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kiss {

// Run `cmd` through the shell with stderr folded into stdout, collect
// everything it prints into `out`, and return its exit status (-1 if the
// child could not be started or did not exit normally).
inline int run_capture(const std::string& cmd, std::string& out) {
  out.clear();
#if defined(_WIN32)
  // cmd.exe needs the whole command wrapped in an extra pair of quotes when it
  // begins with a quoted path, which ours does (build trees live under paths
  // with spaces, e.g. C:\Program Files\...).
  FILE* p = _popen(("\"" + cmd + " 2>&1\"").c_str(), "r");
#else
  FILE* p = popen((cmd + " 2>&1").c_str(), "r");
#endif
  if (!p) return -1;
  char buf[4096];
  while (std::fgets(buf, sizeof buf, p)) out += buf;
#if defined(_WIN32)
  return _pclose(p);  // already the child's exit code
#else
  const int rc = pclose(p);
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
#endif
}

// Absolute path of the running executable, or "" if it cannot be determined.
inline std::string self_exe_path() {
  char buf[4096];
#if defined(_WIN32)
  const DWORD n = GetModuleFileNameA(nullptr, buf, static_cast<DWORD>(sizeof buf));
  if (n == 0 || n >= sizeof buf) return "";
  return std::string(buf, n);
#else
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "";
  return std::string(buf, static_cast<std::size_t>(n));
#endif
}

// Process id, used only to keep concurrent test runs off each other's temp files.
inline long process_id() {
#if defined(_WIN32)
  return static_cast<long>(::_getpid());
#else
  return static_cast<long>(::getpid());
#endif
}

// Thread-safe UTC breakdown of `t` (gmtime_r / gmtime_s). Run directories are
// named from this (docs/design.md §2.2), so it must be UTC, not local time.
inline std::tm gmtime_utc(std::time_t t) {
  std::tm tm{};
#if defined(_WIN32)
  ::gmtime_s(&tm, &t);   // note: Win32 argument order is the reverse of POSIX
#else
  ::gmtime_r(&t, &tm);
#endif
  return tm;
}

// Basename of a helper executable as it must be spelled to run it.
// `exe("verify_s")` is "verify_s" on POSIX and "verify_s.exe" on Windows.
inline std::string exe(const std::string& stem) {
#if defined(_WIN32)
  return stem + ".exe";
#else
  return stem;
#endif
}

}  // namespace kiss
