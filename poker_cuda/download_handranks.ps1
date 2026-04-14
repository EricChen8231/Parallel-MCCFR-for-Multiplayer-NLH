# =============================================================================
# download_handranks.ps1 — Download the Two Plus Two handranks.dat table
# Run with: powershell -ExecutionPolicy Bypass -File download_handranks.ps1
# =============================================================================

$outPath = "$PSScriptRoot\data\handranks.dat"

if (Test-Path $outPath) {
    Write-Host "handranks.dat already exists at $outPath" -ForegroundColor Green
    exit 0
}

New-Item -ItemType Directory -Force -Path "$PSScriptRoot\data" | Out-Null

# Try to download from known mirrors
$urls = @(
    "https://github.com/b-g-goodell/two-plus-two-hand-evaluator/raw/master/HandRanks.dat",
    "https://raw.githubusercontent.com/HenryRLee/PokerHandEvaluator/master/resources/HandRanks.dat"
)

$success = $false
foreach ($url in $urls) {
    Write-Host "Trying: $url"
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $outPath)
        $size = (Get-Item $outPath).Length
        if ($size -gt 100MB) {
            Write-Host "Downloaded successfully: $([math]::Round($size/1MB, 0)) MB" -ForegroundColor Green
            $success = $true
            break
        } else {
            Write-Host "File too small ($size bytes), trying next..." -ForegroundColor Yellow
            Remove-Item $outPath -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Host "Failed: $_" -ForegroundColor Yellow
    }
}

if (-not $success) {
    Write-Host ""
    Write-Host "Automatic download failed. Please download manually:" -ForegroundColor Red
    Write-Host "  1. Go to: https://github.com/b-g-goodell/two-plus-two-hand-evaluator"
    Write-Host "  2. Download HandRanks.dat (should be ~130 MB)"
    Write-Host "  3. Save it to: $outPath"
}
