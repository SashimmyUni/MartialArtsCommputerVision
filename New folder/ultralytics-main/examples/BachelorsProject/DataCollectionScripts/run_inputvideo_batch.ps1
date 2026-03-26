$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }
$batchTag = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $PSScriptRoot ("data/runs/inputvideo_final_" + $batchTag)
$outDir = Join-Path $PSScriptRoot "output_input_batch"
$inputDir = Join-Path $PSScriptRoot "InputVideo"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$videos = Get-ChildItem $inputDir -File | Where-Object { $_.Extension.ToLower() -in @('.mp4','.mov','.avi','.mkv','.webm','.m4v') } | Sort-Object Name
$summary = @()

foreach ($video in $videos) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($video.Name)
    $tech = $name -replace '_Input$',''
    $log = Join-Path $logDir ($name + '.log')
    $overlay = Join-Path $outDir ($name + '_overlay.mp4')

    Write-Host "Running $($video.Name) -> technique=$tech"
    & $pythonExe action_recognition.py --source $video.FullName --target-technique $tech --reference-dir reference_poses --output-path $overlay --disable-video-classifier --skip-frame 2 --no-display *> $log
    $rc = $LASTEXITCODE

    $runId = ''
    $runLine = Select-String -Path $log -Pattern 'structured storage run_id:' -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($runLine) {
        $runId = ($runLine.Line -split ':',2)[1].Trim()
    }

    $metrics = ''
    $final = $null
    $mean = $null
    $max = $null

    if ($runId) {
        $metrics = Join-Path $PSScriptRoot ("data/runs/" + $runId + "/metrics.csv")
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

    $summary += [PSCustomObject]@{
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

$summaryPath = Join-Path $logDir 'summary.csv'
$summary | Export-Csv -NoTypeInformation -Path $summaryPath
$summary | Format-Table -AutoSize
Write-Host ('summary_csv: ' + $summaryPath)
