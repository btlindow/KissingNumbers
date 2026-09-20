# Put the MSVC toolchain on PATH in the *current* PowerShell session, so that
# `cmake --preset windows-release` can find cl.exe (the Ninja generator has no
# way to locate it by itself, unlike the Visual Studio generator).
#
#   . scripts\dev_shell.ps1          # note the leading dot: it must be sourced
#   cmake --preset windows-release
#   cmake --build build/windows-release
#
# Idempotent: re-sourcing it in a session that already has cl.exe does nothing.
# Afterwards the session reports the CUDA toolkit and GPU it will build for,
# which is the docs/design.md section 1 envelope for this machine.
#
# ASCII only, deliberately: Windows PowerShell 5.1 decodes an unsigned .ps1 as
# the active code page, so a stray non-ASCII character here is a parse error.

# PATH as the registry has it, so a toolchain installed after this console was
# opened is still found. Machine first, then user, then whatever we inherited.
$env:Path = @(
    [Environment]::GetEnvironmentVariable('Path', 'Machine'),
    [Environment]::GetEnvironmentVariable('Path', 'User'),
    $env:Path
) -join ';'

if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    Write-Host 'dev_shell: MSVC already on PATH' -ForegroundColor DarkGray
} else {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw 'dev_shell: vswhere.exe not found. Is Visual Studio installed?'
    }

    $vs = & $vswhere -latest -products * `
                     -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                     -property installationPath
    if (-not $vs) {
        throw 'dev_shell: no Visual Studio install carries the C++ toolset (VC.Tools.x86.x64)'
    }

    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvars)) { throw "dev_shell: $vcvars not found" }

    # vcvars64.bat only edits the environment of the cmd.exe that runs it, so
    # run it, dump the resulting environment, and replay it into this session.
    $cmdline = 'call "' + $vcvars + '" >nul 2>&1 & set'
    & cmd.exe /c $cmdline | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path ('Env:' + $Matches[1]) -Value $Matches[2]
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw 'dev_shell: vcvars64.bat ran but cl.exe is still not on PATH'
    }
    Write-Host "dev_shell: MSVC from $vs" -ForegroundColor Green
}

# CUDA. The toolkit installer puts nvcc on PATH; report which one, because
# CMakePresets.json pins an exact path and a mismatch is only a warning.
$nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if ($nvcc) {
    $m = (& nvcc --version | Select-String 'release ([\d.]+)')
    Write-Host ("dev_shell: nvcc {0} at {1}" -f $m.Matches.Groups[1].Value, $nvcc.Source)
} else {
    Write-Warning 'dev_shell: nvcc not on PATH. Install the CUDA Toolkit, or set CMAKE_CUDA_COMPILER.'
}

foreach ($t in 'cmake', 'ninja', 'ctest') {
    if (-not (Get-Command "$t.exe" -ErrorAction SilentlyContinue)) {
        Write-Warning "dev_shell: $t not on PATH"
    }
}

# The GPU this build targets (docs/design.md section 1).
if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
    Write-Host ('dev_shell: GPU ' +
        (& nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader))
}
