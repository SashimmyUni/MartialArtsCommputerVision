param(
    [int]$ExamplesPerAngle = 2,
    [int]$NumVideoSequenceSamples = 150,
    [double]$RefMinScoreGate = 55.0,
    [double]$RefMinMotionEnergy = 0.10,
    [double]$RefMinReturnClosure = 0.0,
    [int]$ReferenceSearchMaxFrames = 260,
    [string]$ReferenceDir = "reference_poses",
    [string]$Weights = "yolo26n-pose.pt"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$repoRoot = Resolve-Path (Join-Path $projectRoot "..\..\..\..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$planPath = Join-Path $projectRoot "reference_poses/generated_capture_plan_all_labels.csv"
if (-not (Test-Path $planPath)) {
    throw "Plan file not found: $planPath"
}

$rows = Import-Csv $planPath | Where-Object { $_.technique -eq "jab" -and $_.command_ready -eq "yes" }
if (-not $rows -or $rows.Count -eq 0) {
    throw "No jab rows found with command_ready=yes in generated_capture_plan_all_labels.csv"
}

$runTag = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $projectRoot "data/runs/jab_seed_batch_$runTag"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$saved = 0
$skipped = 0
$failed = 0

foreach ($row in $rows) {
    $angle = ($row.angle).Trim()
    $urls = @($row.source_url_1, $row.source_url_2, $row.source_url_3, $row.source_url_4) |
        Where-Object { $_ -and $_.Trim() -ne "" } |
        Select-Object -Unique

    if (-not $urls -or $urls.Count -eq 0) {
        Write-Host "[skip] jab/$angle has no source URLs"
        $failed++
        continue
    }

    for ($i = 1; $i -le $ExamplesPerAngle; $i++) {
        $idx = "{0:D2}" -f $i
        $recordKey = "jab__${angle}_seedgate_${idx}"

        $techniqueDir = Join-Path $projectRoot "$ReferenceDir/jab"
        New-Item -ItemType Directory -Force -Path $techniqueDir | Out-Null
        $outPath = Join-Path $techniqueDir "${angle}_seedgate_${idx}.npy"

        if (Test-Path $outPath) {
            Write-Host "[skip] exists: $outPath"
            $skipped++
            continue
        }

        $source = $urls[($i - 1) % $urls.Count]
        $jobLog = Join-Path $logDir "jab_${angle}_${idx}.log"
        Write-Host "[run] $recordKey"

        $args = @(
            "action_recognition.py",
            "--weights", $Weights,
            "--source", $source,
            "--record-reference", $recordKey,
            "--reference-capture-mode", "best_window",
            "--target-technique", "_capture_only",
            "--reference-dir", $ReferenceDir,
            "--num-video-sequence-samples", "$NumVideoSequenceSamples",
            "--skip-frame", "1",
            "--record-reference-max-saves", "1",
            "--reference-capture-cooldown-frames", "24",
            "--disable-video-classifier",
            "--no-display",
            "--auto-exit-after-reference",
            "--reference-search-max-frames", "$ReferenceSearchMaxFrames",
            "--person-selection-mode", "most_motion",
            "--disable-structured-storage",
            "--ref-min-motion-energy", "$RefMinMotionEnergy",
            "--ref-min-return-closure", "$RefMinReturnClosure",
            "--ref-min-score-gate", "$RefMinScoreGate"
        )

        & $python @args *> $jobLog

        if (Test-Path $outPath) {
            Write-Host "[ok] $outPath"
            $saved++
        }
        else {
            Write-Host "[fail] $recordKey (log: $jobLog)"
            $failed++
        }
    }
}

$summary = "SUMMARY saved=$saved skipped=$skipped failed=$failed runTag=$runTag"
$summaryPath = Join-Path $logDir "summary.txt"
$summary | Set-Content $summaryPath
Write-Host $summary
Write-Host "summary file: $summaryPath"
