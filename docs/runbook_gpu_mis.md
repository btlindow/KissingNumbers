# Runbook — `gpu_mis`, the GPU MIS search driver (T3.2c)

`tools/gpu_mis` drives the T3.2a/T3.2b CUDA engine: it seeds the chains, launches the ILS kernel in
bursts, logs, re-seeds stalled chains from an elite pool, checkpoints, and puts anything the device
reports through the record custody chain of docs/design.md §2.4.

```
build/release/tools/gpu_mis configs/default.json
build/release/tools/gpu_mis configs/record_neighbourhood.json --minutes 30
build/release/tools/gpu_mis configs/soak_10min.json --run-name soak --selfcheck
```

---

## 1. Prerequisites

| Needed | How |
|---|---|
| `data/leech_min.i8`, `data/neg.u32` | `build/release/tools/gen_leech data/` |
| `data/adj.u32` (3.62 GB) | `build/release/tools/build_adj data/` |
| `data/S496.txt`, `data/S488.txt` | committed (T1.3) |
| `runs/ls_search/plateau_atoms_496.json` | T3.2b output; optional (the plateau seeds are skipped without it) |
| `data/group/{m24_generators.txt,xi.txt,sextet.txt}` | committed (T4.1); needed only when `seed_mix.co0_*` > 0 |
| `.venv` with numpy | `scripts/setup_venv.sh` — the second verifier is `python/verify_S.py` |
| ~5.5 GB free VRAM | the driver polls `cudaMemGetInfo` and retries `vram_retries` × `vram_retry_seconds` before failing loudly |

Memory: adjacency 3.62 GB + **483,945 bytes per chain** (T3.2a state 442,609 + T3.2b search 41,336).
B = 2048 → 4.6 GB, B = 4096 → 5.6 GB. Both engine states (antipodal / plain) share the one adjacency
table.

---

## 2. Command line

```
gpu_mis <config.json> [options]
  --set key=value      override a config key (repeatable): "B=64", "engine.stall_limit=4000",
                       "seed_mix.greedy=0.3"
  --resume             continue from <out_dir>/<run_name>/checkpoint.txt
  --run-name NAME      name of the run directory (default "<config stem>_<utc>")
  --out-dir DIR        parent of the run directory (default runs/gpu_mis)
  --max-launches N     stop after N launches, counted across resumes (0 = unlimited)
  --minutes M          wall-clock budget (0 = unlimited)
  --seed S             RNG seed
  --target T           output-slot threshold (default 497)
  --selfcheck [N]      run lsa_check every N launches (default 10)
  --dry-run            validate the config, create the run directory, exit
```

The last line of stdout is machine readable (docs/design.md §2.4):

```
RESULT ok=1 run=<name> best=496 launches=1729 iterations=177049600 records=0 verify_fail=0
       found=1 distinct=8134 elapsed_s=1800.4 stop=wall_clock
```

`ok=0` **only** when a verification of a written candidate failed (`verify_fail > 0`) — that is a
loud, unmissable condition; everything else exits `ok=1`.

---

## 3. Configuration

JSON with `//` comments, at most one level of nesting (`seed_mix`, `engine`). Every key has a default
in `include/kiss/run_config.h`; **unknown keys are an error** (a typo must not silently fall back).
`configs/default.json` (diverse mix), `configs/record_neighbourhood.json` (B = 2048, 75 % antipodal,
seeds from the 496 / its plateau images / Co₀ images) and `configs/soak_10min.json` (small, self-check
on) are the three shipped configurations.

| Group | Keys |
|---|---|
| size / budget | `B`, `K`, `antipodal_fraction`, `target`, `wall_clock_minutes`, `max_launches`, `log_every`, `nslots`, `stop_on_record` |
| seeding | `seed`, `seed_mix.{s496,s488,plateau496,co0_496,co0_488,elite,greedy}`, `delete_frac_min/max`, `co0_count/slots/burnin`, `elite_cap`, `elite_min_size`, `elite_in`, `reseed_elite_prob`, `reseed_batch_frac` |
| logging | `log_min_size`, `max_set_files`, `hist_per_launch`, `verify_per_launch`, `watchdog_seconds` |
| durability | `checkpoint_minutes`, `checkpoint_launches`, `selfcheck_every` |
| paths | `out_dir`, `run_name`, `data_dir`, `group_dir`, `plateau_atoms`, `verify_s`, `python`, `verify_py`, `nvidia_smi` |
| GPU sharing | `vram_min_gb`, `vram_retries`, `vram_retry_seconds` |
| engine | `engine.*` — a 1:1 image of `kiss::cuda::LSSearchParams` plus T3.2a's `FL`, `TABU`, `force_cap` |

### Seed mix

Weights are normalised; a kind whose inputs are missing (no plateau atoms, no Co₀ generators, empty
elite pool) drops out of the mix. Each seed is the base set minus a random
`delete_frac_min … delete_frac_max` fraction (whole antipodal pairs in antipodal mode).

| kind | base set |
|---|---|
| `s496`, `s488` | `data/S496.txt`, `data/S488.txt` |
| `plateau496` | a random **non-empty** subset of the six commuting plateau moves applied to the 496 (64 images, all verified at start-up) |
| `co0_496` | `g · (random plateau image)`, `g` a random Co₀ element (product replacement, T4.1) |
| `co0_488` | `g · 488` |
| `elite` | a random member of the elite pool |
| `greedy` | a fresh random greedy maximal set (~230–240; diversity only) |

`antipodal_fraction` splits `B` into an antipodal state and a plain state; both run concurrently on
their own CUDA streams (≈ 20 % faster than sequential launches).

---

## 4. What a run does, launch by launch

```
launch  : lsk_run_chains(K) on each state (separate streams) -> stream-synchronise
collect : lss_download_stats + lss_download_restart_req + one memcpy of the whole best_S block
          -> CSV row -> elite pool -> distinct-set log -> lss_download_output -> custody chain
histogram / verification queues drained (bounded per launch)
selfcheck  every `selfcheck_every` launches (lsa_check: tightness recompute, independence,
           antipodal closure, free-list; any error aborts the run)
reseed     when `reseed_batch_frac`·B chains of a state have raised restart_req
checkpoint every `checkpoint_minutes` / `checkpoint_launches`
```

### Epochs and determinism

A launch is **not** a state boundary. An **epoch** begins with `lsa_init_from_sets` + `lss_init`,
which re-initialises every chain of a state from a host-supplied set. Epoch boundaries happen

1. at the start of the run (seed mix),
2. at a **reseed pass**: chains that raised `restart_req` get a fresh seed (elite pool with random
   deletions with probability `reseed_elite_prob`, otherwise the seed mix); every other chain restarts
   from its own best-ever set,
3. at every **checkpoint**.

Inside an epoch the device evolution is a deterministic function of (initial sets, epoch seed, K,
launch count), so a checkpoint written at an epoch boundary resumes bit-identically. That is why the
checkpoint is always followed by an epoch boundary — and why `--resume` reproduces an uninterrupted
run's per-launch sizes exactly (`tests/test_gpu_mis` stage `resume` asserts it launch by launch).
Wall-clock checkpoints (`checkpoint_minutes`) are of course not reproducible *timing*-wise; use
`checkpoint_launches` when you want a byte-reproducible experiment.

### Record custody (docs/design.md §2.4)

The device raises a flag and copies `S` into an output slot whenever a chain's `|S| ≥ target` exceeds
what that chain reported before. The host then, **in this order**:

1. `write_set` to `runs/<run>/found/S_<size>_<utc>_<hash>.txt` — before anything else;
2. runs `tools/verify_s <file> --data <data_dir>` and `.venv/bin/python python/verify_S.py <file>` as
   subprocesses;
3. logs `RECORD` only when both exit 0, both print `ok=1`, and both report the same size ≥ 497.
   Otherwise it prints a `VERIFY-FAIL` banner, counts it, and the run exits with `ok=0`.

Candidates below 497 (a lower `--target`, e.g. the test's 490) take the same path and are logged as
`verified |S| = n (below 497, not a record)`. Identical sets (same order-independent hash) are written
once; the duplicate count is in the summary.

---

## 5. Run directory

```
runs/gpu_mis/<run_name>/
  config_effective.json   the full resolved configuration + git hash + argv
  log.csv                 one row per launch (see below)
  sets.csv                every distinct best set >= log_min_size, and its tightness histogram
  gpu.csv                 nvidia-smi watchdog samples (log only, never acts)
  run.log                 human-readable event log incl. the full verifier transcripts
  summary.txt             the end-of-run summary (same text as stdout)
  checkpoint.txt          resumable state (chain sets, elite pool, host RNG, counters, epoch)
  found/                  candidates >= target, written before verification
  sets/                   up to max_set_files of the distinct sets >= log_min_size
```

`log.csv` columns: `t_s, launch, epoch, iter_per_chain, iters_total, best, mean_best, improved,
restarts, reseeds, it_per_s, elite, distinct_ge_log, found, records, sizes_hash, gpu_temp_c,
gpu_power_w, gpu_clk_mhz`. `sizes_hash` is a hash of all chains' (best size, size) — the determinism
fingerprint of a launch.

`sets.csv` columns: `t_s, launch, state, chain, size, hash, file, tight_histogram`; the histogram is
computed on the GPU (`tightness_full`) a launch or two later and appended as a second row with
`state = hist`. At most `hist_per_launch` new distinct sets get the full treatment (verify + file +
histogram) per launch; the rest are still counted (they carry `(not sampled)`), so the distinct-set
census stays exact while the host cost stays bounded.

---

## 6. Watchdog and thermals

`watchdog_seconds` (default 60) samples
`nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu` into
`gpu.csv` and into the per-launch console line. **It only logs** — no throttling decisions are made,
and results are correct regardless of clocks (docs/design.md §6.2).

---

## 7. Recipes

```bash
# 30 minutes in the record neighbourhood, 2048 chains, mostly antipodal
build/release/tools/gpu_mis configs/record_neighbourhood.json --minutes 30

# the diverse mix, unattended, big batch
build/release/tools/gpu_mis configs/default.json --set B=4096 --set K=100 --minutes 240 \
    --run-name overnight

# resume it after a reboot / an interruption
build/release/tools/gpu_mis configs/default.json --run-name overnight --resume --minutes 240

# reproduce an experiment exactly (no wall-clock checkpoints)
build/release/tools/gpu_mis configs/soak_10min.json --run-name exp --seed 42 \
    --set checkpoint_minutes=0 --set checkpoint_launches=100 --max-launches 500 --minutes 0

# regression: does the engine still refill the 496 from 480-subsets?
build/release/tools/gpu_mis configs/soak_10min.json --run-name refill --target 490 \
    --set seed_mix.s496=1 --set seed_mix.s488=0 --set seed_mix.plateau496=0 \
    --set seed_mix.elite=0 --set seed_mix.greedy=0 \
    --set delete_frac_min=0.032 --set delete_frac_max=0.033 --max-launches 3 --minutes 0

# start from the sets a previous run found
build/release/tools/gpu_mis configs/default.json --set elite_in=runs/gpu_mis/overnight/sets
```

Stopping: `SIGINT`/`SIGTERM` (Ctrl-C) finishes the current launch, drains the verification queue,
writes a final checkpoint and the summary. `--resume` then picks the run up exactly where it stopped.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `the GPU is busy: only … free` | another process holds the card; the driver already retried `vram_retries` times. Lower `B` or wait. |
| `data/adj.u32 missing` | `tools/build_adj data/` (~15 s, 3.62 GB) |
| `--resume: checkpoint has no state "anti" with B = …` | `B` or `antipodal_fraction` changed since the checkpoint; resume with the same config. |
| `config: unknown key "…"` | a typo — every valid key is listed in `include/kiss/run_config.h`. |
| `selfcheck failed on state …` | an engine invariant broke (T3.2a/b bug): the run aborts on purpose. Re-run under `compute-sanitizer` with a small `B`. |
| `VERIFY-FAIL` | a set the device claimed is not accepted by both verifiers. **Do not claim anything.** The file is in `found/`, the full transcript in `run.log`. |
| throughput far below the table in §9 | another process on the GPU, or `hist_per_launch` / `selfcheck_every` set too aggressively. |

---

## 9. Measured throughput (RTX 3070 Laptop, this repo, T3.2c)

| configuration | aggregate ILS iterations/s |
|---|---|
| B = 512, 50 % antipodal, K = 50 | ≈ 1.0 × 10⁵ |
| B = 2048, 75 % antipodal, K = 50 | ≈ 1.2 × 10⁵ |

Host overhead per launch (stats + best_S block + elite/distinct bookkeeping + queues) is a few percent;
a reseed pass or a checkpoint costs one `lsa_init_from_sets` (≈ 0.4 s at B = 2048, dominated by the
`tightness_full` recompute).
