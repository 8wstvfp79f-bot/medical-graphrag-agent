$ErrorActionPreference = "Stop"

$pythonPath = Join-Path $env:USERPROFILE "anaconda3\envs\medical-graphrag\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "未找到 medical-graphrag Conda 环境：$pythonPath"
}

Set-Location -LiteralPath $PSScriptRoot
Write-Host "正在启动 Medical GraphRAG API 与前端..."
Write-Host "前端：http://127.0.0.1:8000/"
Write-Host "接口文档：http://127.0.0.1:8000/docs"
& $pythonPath -m src.api.app
