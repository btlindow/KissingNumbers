#!/usr/bin/env bash
# Create/refresh the project virtualenv at <repo>/.venv (idempotent).
#   scripts/setup_venv.sh && source .venv/bin/activate
#
# This machine's python3 (3.10.12) has no `ensurepip` (python3-venv is not
# installed), so `python3 -m venv` alone produces a venv without pip. Fallback
# chain: venv → virtualenv → venv --without-pip + bootstrap pip via the system
# pip's `--python` option (pip ≥ 22.3).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
REQ="${ROOT}/python/requirements.txt"
PY="${PYTHON:-python3}"

create_venv() {
  if "${PY}" -m venv "${VENV}" >/dev/null 2>&1; then
    echo "setup_venv: created ${VENV} with '${PY} -m venv'"
  elif rm -rf "${VENV}" && "${PY}" -m virtualenv -q "${VENV}" 2>/dev/null; then
    echo "setup_venv: created ${VENV} with '${PY} -m virtualenv'"
  else
    rm -rf "${VENV}"
    "${PY}" -m venv --without-pip "${VENV}"
    echo "setup_venv: created ${VENV} with '${PY} -m venv --without-pip'"
  fi
}

if [[ -x "${VENV}/bin/python" ]]; then
  echo "setup_venv: ${VENV} already exists"
else
  create_venv
fi

if ! "${VENV}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "setup_venv: bootstrapping pip into ${VENV} via system pip"
  "${PY}" -m pip --quiet --python "${VENV}/bin/python" install pip
fi

"${VENV}/bin/python" -m pip install --quiet --upgrade pip
"${VENV}/bin/python" -m pip install --quiet -r "${REQ}"

"${VENV}/bin/python" - <<'PYEOF'
import sys, numpy, scipy, sympy, pytest
print(f"python {sys.version.split()[0]} numpy {numpy.__version__} scipy {scipy.__version__} "
      f"sympy {sympy.__version__} pytest {pytest.__version__}")
print(f"RESULT ok=1 venv={sys.prefix}")
PYEOF
