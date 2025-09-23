<#
PowerShell wrapper to discover & download CSV/XLS files from football-data.co.uk/data.php
Usage:
  .\download_football_data.ps1 -OutDir ..\data\football-data -Filter 2025 -DryRun
#>
param(
    [string]$Entry = "https://www.football-data.co.uk/data.php",
    [string]$OutDir = "..\data\football-data",
    [switch]$DryRun,
    [string]$Filter = $null
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outFull = Resolve-Path -Path (Join-Path $scriptRoot $OutDir) -ErrorAction SilentlyContinue
if (-not $outFull) { New-Item -ItemType Directory -Path (Join-Path $scriptRoot $OutDir) -Force | Out-Null; $outFull = Resolve-Path -Path (Join-Path $scriptRoot $OutDir) }
$outFull = $outFull.Path

Write-Host "Fetching $Entry ..."
try {
    $r = Invoke-WebRequest -Uri $Entry -UseBasicParsing -TimeoutSec 30
} catch {
    Write-Error "Failed to fetch $Entry : $_"
    exit 1
}

$links = @()
foreach ($a in $r.Links) {
    if ($a.href -match '\.(csv|xls|xlsx)$') {
        $uri = [Uri]::new($Entry, $a.href)
        $links += $uri.AbsoluteUri
    }
}
$links = $links | Sort-Object -Unique
if ($Filter) { $links = $links | Where-Object { $_ -like "*$Filter*" } }
Write-Host "Found $($links.Count) files"

foreach ($url in $links) {
    $file = [IO.Path]::GetFileName(([Uri]$url).AbsolutePath)
    $outFile = Join-Path $outFull $file
    if (Test-Path $outFile) {
        Write-Host "Skipping $file (exists)"
        continue
    }
    Write-Host "Downloading $file ..."
    if (-not $DryRun) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $outFile -TimeoutSec 60
            Start-Sleep -Seconds 1
        } catch {
            Write-Warning "Failed $url : $_"
        }
    }
}
Write-Host "Done. Files saved to $outFull"
