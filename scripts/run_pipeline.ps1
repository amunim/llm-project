# Run the data pipeline with default settings
$repoRoot = Split-Path -Parent $PSScriptRoot
$input = Join-Path $repoRoot "data\raw\NUST Bank-Product-Knowledge.xlsx"
$output = Join-Path $repoRoot "data\processed\cleaned_chunks.json"

python (Join-Path $repoRoot "src\data_pipeline.py") --input $input --output $output --lowercase --tokenize
