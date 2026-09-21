# Advance the submodule external/kravatskiy (A. Kravatskiy's repository) to its upstream head and
# say what moved: old and new commit, and whether RESULTS.md changed (with the changed rows).
# A plain `git pull` in this repository does NOT do this. Pinned verifications are unaffected:
# they read detached checkouts made by tools/kravatskiy/pinned.py.
#   powershell -ExecutionPolicy Bypass -File scripts\update_kravatskiy.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path 'external/kravatskiy/.git')) { git submodule update --init external/kravatskiy | Out-Null }
$old = (git -C external/kravatskiy rev-parse HEAD).Trim()
git submodule update --remote external/kravatskiy
$new = (git -C external/kravatskiy rev-parse HEAD).Trim()
Write-Output "external/kravatskiy: $old -> $new"
if ($old -eq $new) {
    Write-Output 'no new commits; RESULTS.md unchanged'
} else {
    $n = (git -C external/kravatskiy rev-list --count "$old..$new").Trim()
    git -C external/kravatskiy diff --quiet $old $new -- RESULTS.md
    if ($LASTEXITCODE -eq 0) {
        Write-Output "$n new commit(s); RESULTS.md unchanged"
    } else {
        Write-Output "$n new commit(s); RESULTS.md CHANGED. Changed rows (- old, + new):"
        git -C external/kravatskiy diff -U0 $old $new -- RESULTS.md | Where-Object { $_ -match '^[-+]\|' } | ForEach-Object { $_.Substring(0, [Math]::Min(160, $_.Length)) }
    }
}
Write-Output 'to record the new submodule commit in this repository: git add external/kravatskiy; git commit'
