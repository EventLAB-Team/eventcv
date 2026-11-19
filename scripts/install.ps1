# Installs the standalone `eventcv` command from the GitHub release binaries:
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/EventLAB-Team/eventcv/releases/latest/download/install.ps1 | iex"
#
# $env:EVENTCV_VERSION pins a release (default: the latest), $env:EVENTCV_INSTALL_DIR picks where the
# binary goes (default: %LOCALAPPDATA%\Programs\eventcv\bin), and $env:EVENTCV_DOWNLOAD_URL replaces
# the release URL entirely (a mirror, or a local folder — how CI tests this script). Upgrading is
# running it again; uninstalling is deleting that folder and removing it from the user PATH.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # the progress bar makes Invoke-WebRequest crawl

$repo = 'https://github.com/EventLAB-Team/eventcv'
# x64 only: Windows on ARM runs it under emulation.
$archive = 'eventcv-x86_64-pc-windows-msvc.zip'

if ($env:EVENTCV_DOWNLOAD_URL) {
    $base = $env:EVENTCV_DOWNLOAD_URL
} elseif ($env:EVENTCV_VERSION) {
    $base = "$repo/releases/download/v$($env:EVENTCV_VERSION.TrimStart('v'))"
} else {
    $base = "$repo/releases/latest/download"
}
$installDir = if ($env:EVENTCV_INSTALL_DIR) { $env:EVENTCV_INSTALL_DIR } else {
    Join-Path $env:LOCALAPPDATA 'Programs\eventcv\bin'
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "downloading $base/$archive"
    # A local folder is copied rather than fetched; everything else is a URL.
    if (Test-Path $base) {
        Copy-Item (Join-Path $base $archive) $tmp
        Copy-Item (Join-Path $base "$archive.sha256") $tmp
    } else {
        Invoke-WebRequest "$base/$archive" -OutFile (Join-Path $tmp $archive) -UseBasicParsing
        Invoke-WebRequest "$base/$archive.sha256" -OutFile (Join-Path $tmp "$archive.sha256") -UseBasicParsing
    }

    $expected = ((Get-Content (Join-Path $tmp "$archive.sha256") -Raw) -split '\s+')[0]
    $actual = (Get-FileHash (Join-Path $tmp $archive) -Algorithm SHA256).Hash
    if ($expected -ne $actual) {
        throw "checksum mismatch for $archive (expected $expected, got $actual)"
    }

    Expand-Archive (Join-Path $tmp $archive) -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    Copy-Item (Join-Path $tmp 'eventcv.exe') $installDir -Force
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$exe = Join-Path $installDir 'eventcv.exe'
Write-Host "installed $((& $exe --version)[0]) to $exe"

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (($userPath -split ';') -contains $installDir)) {
    [Environment]::SetEnvironmentVariable('Path', "$installDir;$userPath", 'User')
    Write-Host "added $installDir to your user PATH; open a new terminal to use ``eventcv``"
}
$found = Get-Command eventcv -ErrorAction SilentlyContinue | Select-Object -First 1
# A pip-installed `eventcv` earlier on PATH would quietly answer instead of this one.
if ($found -and ($found.Source -ne $exe)) {
    Write-Host "note: $($found.Source) comes first on PATH, so ``eventcv`` runs that one, not this"
}
