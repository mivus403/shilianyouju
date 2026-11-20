param(
    [string]$SourceDirectory = ".build\real_sources",
    [string]$OutputDirectory = "samples\real_world",
    [string]$FfmpegPath = $env:FFMPEG_PATH
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $FfmpegPath) {
    $command = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($command) {
        $FfmpegPath = $command.Source
    }
}
if (-not $FfmpegPath -or -not (Test-Path -LiteralPath $FfmpegPath)) {
    throw "未找到 FFmpeg，请设置 FFMPEG_PATH 或传入 -FfmpegPath。"
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$sourceRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot $SourceDirectory))
$outputRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
$fontPath = "C\:/Windows/Fonts/msyh.ttc"
$overlayText = "真人素材改编 · 中文旁白为项目后加 · 不代表画面人物观点 · 内容提取测试 · 不供跟练"

function Invoke-Ffmpeg {
    param([string[]]$Arguments)
    & $FfmpegPath -hide_banner -loglevel error @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "FFmpeg 执行失败，退出码 $LASTEXITCODE"
    }
}

function New-NarratedSample {
    param(
        [string]$SourceName,
        [string]$CaseName,
        [string]$NarrationPath,
        [double[]]$FrameTimes
    )
    $source = Join-Path $sourceRoot $SourceName
    if (-not (Test-Path -LiteralPath $source)) {
        throw "缺少源视频：$source"
    }
    $caseDirectory = Join-Path $outputRoot $CaseName
    New-Item -ItemType Directory -Force -Path $caseDirectory | Out-Null
    $video = Join-Path $caseDirectory "video.mp4"

    $filter = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,drawbox=x=0:y=0:w=iw:h=66:color=black@0.72:t=fill,drawtext=fontfile='$fontPath':text='$overlayText':fontcolor=white:fontsize=25:x=(w-text_w)/2:y=18"
    Invoke-Ffmpeg -Arguments @(
        "-y", "-stream_loop", "-1", "-i", $source, "-i", $NarrationPath,
        "-map", "0:v:0", "-map", "1:a:0", "-vf", $filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-shortest", "-movflags", "+faststart", $video
    )

    $index = 1
    foreach ($second in $FrameTimes) {
        $framePath = Join-Path $caseDirectory ("scene_frame_{0:00}.jpg" -f $index)
        Invoke-Ffmpeg -Arguments @(
            "-y", "-ss", $second.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture),
            "-i", $video, "-frames:v", "1", "-update", "1", "-q:v", "2", $framePath
        )
        $index += 1
    }
}

$caseBVideo = Join-Path $projectRoot "samples\case_b_preparation\video.mp4"
$caseCVideo = Join-Path $projectRoot "samples\case_c_unknown\video.mp4"
$audioD = Join-Path $sourceRoot "narration-d.wav"
$audioE = Join-Path $sourceRoot "narration-e.wav"

$audioDFilter = "[0:a]atrim=start=4.723:end=7.837,asetpts=PTS-STARTPTS[a1];anullsrc=r=22050:cl=mono,atrim=duration=0.55[s1];[1:a]atrim=start=18.538:end=23.815,asetpts=PTS-STARTPTS[a2];anullsrc=r=22050:cl=mono,atrim=duration=0.55[s2];[0:a]atrim=start=20.881:end=25.699,asetpts=PTS-STARTPTS[a3];[a1][s1][a2][s2][a3]concat=n=5:v=0:a=1[out]"
Invoke-Ffmpeg -Arguments @(
    "-y", "-i", $caseBVideo, "-i", $caseCVideo,
    "-filter_complex", $audioDFilter, "-map", "[out]", "-c:a", "pcm_s16le", $audioD
)
Invoke-Ffmpeg -Arguments @(
    "-y", "-i", $caseBVideo, "-vn", "-c:a", "pcm_s16le", $audioE
)

New-NarratedSample `
    -SourceName "pexels-3196290.mp4" `
    -CaseName "case_d_real_seated" `
    -NarrationPath $audioD `
    -FrameTimes @(2.0, 6.0, 10.0)

New-NarratedSample `
    -SourceName "pexels-3196293.mp4" `
    -CaseName "case_e_real_mixed" `
    -NarrationPath $audioE `
    -FrameTimes @(2.0, 8.0, 14.0)

Remove-Item -LiteralPath $audioD, $audioE -Force

Write-Output "真人画面测试样例已生成到 $outputRoot"
