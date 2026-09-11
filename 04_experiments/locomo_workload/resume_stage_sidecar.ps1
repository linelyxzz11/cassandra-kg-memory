$ErrorActionPreference = 'Stop'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$pythonExe = if ($env:CASSMEM_PYTHON_EXE) { $env:CASSMEM_PYTHON_EXE } else { 'D:\ANACONDA\python.exe' }
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}
$cells = @('cassandra-base','cassandra-materialized','neo4j-native','neo4j-materialized')
foreach ($rep in @(0,1,2)) {
    foreach ($concurrency in @(1,8,16,32,64)) {
        foreach ($cell in $cells) {
            $summary = "05_reports\locomo_workload_100k\stage_sidecar\${cell}_c${concurrency}_r${rep}_summary.json"
            if (-not (Test-Path -LiteralPath $summary)) {
                & $pythonExe '04_experiments\locomo_workload\run_95_5_stage_sidecar.py' --cell $cell --concurrency $concurrency --rep $rep
                if ($LASTEXITCODE -ne 0) {
                    throw "sidecar failed: $cell c$concurrency r$rep"
                }
            }
        }
    }
}
