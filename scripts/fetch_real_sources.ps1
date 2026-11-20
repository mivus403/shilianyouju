param([string]$TargetDirectory = ".build\real_sources")

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$targetRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot $TargetDirectory))
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$sources = @(
    @{
        File = "pexels-3196290.mp4"
        Url = "https://videos.pexels.com/video-files/3196290/3196290-hd_1920_1080_25fps.mp4"
        Sha256 = "984F076CFB3662FE79576BC097BD8EB0F679C8877CC3E0CF0F645606C255F7A8"
    },
    @{
        File = "pexels-3196293.mp4"
        Url = "https://videos.pexels.com/video-files/3196293/3196293-hd_1920_1080_25fps.mp4"
        Sha256 = "30960BE21CAEF9194388E7664FF3562272CA646F82FA6FC17DAF14A716254381"
    }
)

foreach ($source in $sources) {
    $target = Join-Path $targetRoot $source.File
    if (-not (Test-Path -LiteralPath $target)) {
        & curl.exe --fail --location $source.Url --output $target
        if ($LASTEXITCODE -ne 0) {
            throw "下载失败：$($source.Url)"
        }
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash
    if ($actual -ne $source.Sha256) {
        throw "文件校验失败：$target"
    }
    Write-Output "已校验 $($source.File)"
}

