$ErrorActionPreference = "Stop"

$Root = "E:\project\Qwen3-TTS-main\finetuning"
$GenerationPid = 21680

Set-Location $Root

"[$(Get-Date -Format s)] Waiting for generation PID $GenerationPid" | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append

try {
    $proc = Get-Process -Id $GenerationPid -ErrorAction SilentlyContinue
    if ($proc) {
        Wait-Process -Id $GenerationPid
    }
} catch {
    "[$(Get-Date -Format s)] Wait-Process warning: $($_.Exception.Message)" | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append
}

$wavCount = 0
$jsonlCount = 0
if (Test-Path ".\data_1000") {
    $wavCount = (Get-ChildItem ".\data_1000" -Filter "*.wav" -File | Measure-Object).Count
}
if (Test-Path ".\train_raw_1000.jsonl") {
    $jsonlCount = (Get-Content ".\train_raw_1000.jsonl" | Measure-Object -Line).Lines
}

"[$(Get-Date -Format s)] Generation finished. wav=$wavCount jsonl=$jsonlCount" | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append

if ($wavCount -lt 1000 -or $jsonlCount -lt 1000) {
    "[$(Get-Date -Format s)] Not enough generated samples; skipping tokenizer/training." | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append
    exit 1
}

"[$(Get-Date -Format s)] Starting tokenizer encode -> train_with_codes_1000.jsonl" | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append
python ".\prepare_data_vd.py" `
  --device cuda:0 `
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz `
  --input_jsonl train_raw_1000.jsonl `
  --output_jsonl train_with_codes_1000.jsonl `
  *>&1 | Tee-Object -FilePath ".\vd_1000_prepare_codes.log" -Append

"[$(Get-Date -Format s)] Starting VoiceDesign training -> output_vd_1000" | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append
python ".\sft_voice_design.py" `
  --train_jsonl train_with_codes_1000.jsonl `
  --init_model_path e:/project/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --output_model_path output_vd_1000 `
  --batch_size 1 `
  --num_epochs 5 `
  --lr 1e-5 `
  --model_dtype bfloat16 `
  --mixed_precision bf16 `
  *>&1 | Tee-Object -FilePath ".\vd_1000_train.log" -Append

"[$(Get-Date -Format s)] Pipeline completed." | Tee-Object -FilePath ".\vd_1000_pipeline.log" -Append
