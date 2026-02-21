# PowerShell script to build multi-arch images and save them locally

$ImageName = "evm-phase-app"
$OutputDir = "docker_images"

Write-Host "Creating output directory: $OutputDir"
if (!(Test-Path -Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

Write-Host "============================"
Write-Host " 1. Building AMD64 Image"
Write-Host "============================"
# Clean build for amd64
docker buildx build --platform linux/amd64 -t "${ImageName}:local-amd64" --load .
if ($LASTEXITCODE -eq 0) {
    Write-Host "Saving AMD64 Image to tar..."
    docker save -o "${OutputDir}/${ImageName}_amd64.tar" "${ImageName}:local-amd64"
    Write-Host "-> Saved to ${OutputDir}/${ImageName}_amd64.tar"
} else {
    Write-Host "Failed to build AMD64 image" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================"
Write-Host " 2. Building ARM64 Image"
Write-Host "============================"
# Clean build for arm64 (Raspberry Pi, Mac M1/M2)
docker buildx build --platform linux/arm64 -t "${ImageName}:local-arm64" --load .
if ($LASTEXITCODE -eq 0) {
    Write-Host "Saving ARM64 Image to tar..."
    docker save -o "${OutputDir}/${ImageName}_arm64.tar" "${ImageName}:local-arm64"
    Write-Host "-> Saved to ${OutputDir}/${ImageName}_arm64.tar"
} else {
    Write-Host "Failed to build ARM64 image" -ForegroundColor Red
}

Write-Host ""
Write-Host "Build complete. Check the ./$OutputDir folder."
