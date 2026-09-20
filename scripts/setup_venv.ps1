# Create/refresh the project virtualenv at <repo>\.venv (idempotent).
# The Windows counterpart of scripts/setup_venv.sh.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
#   .\.venv\Scripts\Activate.ps1
#
# Unlike the Linux box of docs/design.md §1, the python.org/Store builds of
# Python here do ship ensurepip, so `py -m venv` is enough and there is no
# virtualenv fallback chain. The layout differs too: Scripts\python.exe, not
# bin/python.
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$req  = Join-Path $root 'python\requirements.txt'
$vpy  = Join-Path $venv 'Scripts\python.exe'

# `py` is the launcher; fall back to whatever python is on PATH.
$py = if (Get-Command py.exe -ErrorAction SilentlyContinue) { 'py' }
      elseif (Get-Command python.exe -ErrorAction SilentlyContinue) { 'python' }
      else { throw 'setup_venv: no Python found on PATH' }

if (Test-Path $vpy) {
    Write-Host "setup_venv: $venv already exists"
} else {
    & $py -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "setup_venv: '$py -m venv' failed" }
    Write-Host "setup_venv: created $venv with '$py -m venv'"
}

& $vpy -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'setup_venv: pip upgrade failed' }
& $vpy -m pip install --quiet -r $req
if ($LASTEXITCODE -ne 0) { throw 'setup_venv: requirements install failed' }

# Same RESULT line the shell script prints, so either can be checked the same way.
# Fed on stdin, not via -c: a multi-line string handed to -c loses its newlines
# crossing the PowerShell/Win32 argv boundary and python then sees one broken line.
$check = @'
import sys, numpy, scipy, sympy, pytest
print(f"python {sys.version.split()[0]} numpy {numpy.__version__} scipy {scipy.__version__} "
      f"sympy {sympy.__version__} pytest {pytest.__version__}")
print(f"RESULT ok=1 venv={sys.prefix}")
'@
$check | & $vpy -
