# InfoPilot Windows 실행 파일 빌드 스크립트
# 사용법 예시:
#   powershell -ExecutionPolicy Bypass -File scripts/build_windows_exe.ps1

param(
    [string]$Python = "python",
    [string]$OutputName = "InfoPilotLauncher"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

if (-not (Test-Path "$repoRoot\windows_launcher.py")) {
    Write-Error "windows_launcher.py 파일을 찾을 수 없습니다. 저장소 루트에서 실행해주세요."
}

Write-Host "🚧 PyInstaller로 Windows 실행 파일을 빌드합니다..." -ForegroundColor Cyan
Write-Host "   Python  : $Python"
Write-Host "   Name    : $OutputName"

$cmd = @(
    $Python, "-m", "PyInstaller",
    "--clean",
    "--noconfirm",
    "--onefile",
    "--name", $OutputName,
    "windows_launcher.py"
)

Write-Host "   Command : $($cmd -join ' ')"
& $cmd[0] $cmd[1..($cmd.Length-1)]

Write-Host "✅ 빌드가 완료되었습니다. dist\\$OutputName.exe 파일을 확인하세요." -ForegroundColor Green
