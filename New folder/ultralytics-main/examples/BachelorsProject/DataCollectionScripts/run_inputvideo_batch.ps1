param(
    [int]$Jobs = 1
)

$ErrorActionPreference = "Continue"

# action_recognition.py lives in BachelorsProject/, one level above this script
# (DataCollectionScripts/). Previously this script did `Set-Location $PSScriptRoot`
# and then invoked `python.exe action_recognition.py ...` with a path relative to
# that (wrong) directory, which python.exe cannot resolve — the batch never
# actually ran action_recognition.py. Every $PSScriptRoot-based path below is
# fixed to $projectRoot for the same reason: action_recognition.py's own
# PROJECT_ROOT-relative resolution (InputVideo/, data/runs/<run_id>/metrics.csv,
# etc.) is anchored to BachelorsProject/, not DataCollectionScripts/.
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$batchTag = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $projectRoot ("data/runs/inputvideo_final_" + $batchTag)
$outDir = Join-Path $projectRoot "output_input_batch"
$inputDir = Join-Path $projectRoot "InputVideo"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$videos = Get-ChildItem $inputDir -File | Where-Object { $_.Extension.ToLower() -in @('.mp4','.mov','.avi','.mkv','.webm','.m4v') } | Sort-Object Name

function Invoke-OneVideo {
    param($video, $projectRoot, $pythonExe, $logDir, $outDir)

    $name = [System.IO.Path]::GetFileNameWithoutExtension($video.Name)
    $tech = $name -replace '_Input$', ''
    $log = Join-Path $logDir ($name + '.log')
    $overlay = Join-Path $outDir ($name + '_overlay.mp4')

    Write-Host "Running $($video.Name) -> technique=$tech"
    Push-Location $projectRoot
    try {
        & $pythonExe action_recognition.py --source $video.FullName --target-technique $tech --reference-dir reference_poses --output-path $overlay --disable-video-classifier --skip-frame 2 --no-display *> $log
        $rc = $LASTEXITCODE
    } finally {
        Pop-Location
    }

    $runId = ''
    $runLine = Select-String -Path $log -Pattern 'structured storage run_id:' -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($runLine) {
        $runId = ($runLine.Line -split ':', 2)[1].Trim()
    }

    $metrics = ''
    $final = $null
    $mean = $null
    $max = $null

    if ($runId) {
        $metrics = Join-Path $projectRoot ("data/runs/" + $runId + "/metrics.csv")
        if (Test-Path $metrics) {
            $rows = Import-Csv $metrics
            $scores = @()
            foreach ($r in $rows) {
                $d = 0.0
                if ([double]::TryParse([string]$r.score, [ref]$d)) {
                    $scores += $d
                }
            }
            if ($scores.Count -gt 0) {
                $final = [Math]::Round($scores[-1], 2)
                $mean = [Math]::Round((($scores | Measure-Object -Average).Average), 2)
                $max = [Math]::Round((($scores | Measure-Object -Maximum).Maximum), 2)
            }
        }
    }

    [PSCustomObject]@{
        video = $video.Name
        technique = $tech
        exit_code = $rc
        run_id = $runId
        final_score = $final
        mean_score = $mean
        max_score = $max
        metrics_csv = $metrics
        log_file = $log
        output_overlay = $overlay
    }
}

$summary = @()

if ($Jobs -le 1 -or $videos.Count -le 1) {
    # Original sequential path.
    foreach ($video in $videos) {
        $summary += Invoke-OneVideo -video $video -projectRoot $projectRoot -pythonExe $pythonExe -logDir $logDir -outDir $outDir
    }
} else {
    # PowerShell 5.1 has no ForEach-Object -Parallel (that's PS7+), so use
    # Start-Job: each job is its own process with its own YOLO model on the
    # GPU, same VRAM caveat as action_recognition.py's --detect-stride /
    # run_reference_collection_batch.py's --jobs — keep -Jobs modest.
    Write-Host "running up to $Jobs video(s) concurrently"
    $funcDef = "function Invoke-OneVideo { ${function:Invoke-OneVideo} }"
    $running = @()
    $queue = [System.Collections.Generic.Queue[object]]::new($videos)

    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        while ($running.Count -lt $Jobs -and $queue.Count -gt 0) {
            $video = $queue.Dequeue()
            $job = Start-Job -ScriptBlock {
                param($funcDef, $video, $projectRoot, $pythonExe, $logDir, $outDir)
                Invoke-Expression $funcDef
                Invoke-OneVideo -video $video -projectRoot $projectRoot -pythonExe $pythonExe -logDir $logDir -outDir $outDir
            } -ArgumentList $funcDef, $video, $projectRoot, $pythonExe, $logDir, $outDir
            $running += $job
        }

        $doneJobs = @(Wait-Job -Job $running -Any)
        $doneIds = $doneJobs | ForEach-Object { $_.Id }
        $summary += Receive-Job -Job $doneJobs
        Remove-Job -Job $doneJobs
        $running = @($running | Where-Object { $doneIds -notcontains $_.Id })
    }
}

$summaryPath = Join-Path $logDir 'summary.csv'
$summary | Export-Csv -NoTypeInformation -Path $summaryPath
$summary | Format-Table -AutoSize
Write-Host ('summary_csv: ' + $summaryPath)
