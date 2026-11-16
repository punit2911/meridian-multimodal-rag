param(
    [string]$BaseUrl = "http://127.0.0.1:8001"
)

$ErrorActionPreference = "Stop"
$corpusDir = Join-Path (Split-Path -Parent $PSScriptRoot) "demo_corpus_stress"
$files = @(Get-ChildItem -LiteralPath $corpusDir -File | Where-Object { $_.Extension -in @(".txt", ".md") })
if ($files.Count -eq 0) {
    throw "No .txt or .md stress documents found in $corpusDir."
}

$apiBase = $BaseUrl.TrimEnd("/")
$health = $null
$deadline = (Get-Date).AddMinutes(5)
while (-not $health -and (Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod -Uri "$apiBase/api/health" -TimeoutSec 5
    } catch {
        Start-Sleep -Seconds 2
    }
}
if (-not $health) {
    throw "Meridian did not become ready at $apiBase."
}

$results = @()
$totalBatches = [Math]::Ceiling($files.Count / 20)
for ($start = 0; $start -lt $files.Count; $start += 20) {
    $end = [Math]::Min($start + 19, $files.Count - 1)
    $batch = @($files[$start..$end])
    Write-Host "Uploading batch $([Math]::Floor($start / 20) + 1) of $totalBatches ($($batch.Count) files)..."
    $curlArgs = @("--fail", "--silent", "--show-error", "--max-time", "1800")
    foreach ($file in $batch) {
        $curlArgs += @("-F", "file=@$($file.FullName)")
    }
    $response = & curl.exe @curlArgs "$apiBase/api/documents"
    if ($LASTEXITCODE -ne 0) {
        throw "Stress corpus upload failed for batch starting at file $($start + 1) (curl exit code $LASTEXITCODE)."
    }

    $body = ($response -join [Environment]::NewLine) | ConvertFrom-Json
    $results += @($body.results)
}
if ($results.Count -ne $files.Count) {
    throw "Expected $($files.Count) upload results, received $($results.Count)."
}
$errors = @($results | Where-Object { $_.status -eq "error" })
$added = @($results | Where-Object { $_.status -in @("added", "replaced") }).Count
$duplicates = @($results | Where-Object { $_.status -eq "duplicate" }).Count
Write-Host "Stress documents: $added added/replaced, $duplicates already present, $($errors.Count) errors."
if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Host "$($_.file): $($_.error)" }
    exit 1
}

$health = Invoke-RestMethod -Uri "$apiBase/api/health" -TimeoutSec 15
Write-Host "Knowledge base now has $($health.documents) documents and $($health.chunks) chunks."
