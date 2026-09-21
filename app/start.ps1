param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $AppRoot
$FrontendRoot = Join-Path $AppRoot "frontend"

if (-not $SkipBuild) {
    Push-Location $FrontendRoot
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "node_modules"))) {
            npm install
        }
        npm run build
    }
    finally {
        Pop-Location
    }
}

if (-not $env:VINO_LOCAL_FILES_ONLY) {
    $env:VINO_LOCAL_FILES_ONLY = "1"
}

Set-Location $ProjectRoot
uv run vino-api
