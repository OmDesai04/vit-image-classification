# Git Push Script for Windows PowerShell
# Run this script to push your changes to GitHub

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  GitHub Push Helper Script" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit
}

Write-Host "✅ Git is installed`n" -ForegroundColor Green

# Navigate to project directory
$projectDir = "c:\Users\desai\Desktop\PRL\IMAGE_CLASSIFICATION"
Set-Location $projectDir
Write-Host "📁 Current directory: $projectDir`n" -ForegroundColor Cyan

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "🔧 Initializing Git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
    Write-Host "✅ Git initialized`n" -ForegroundColor Green
}

# Check git status
Write-Host "📊 Git Status:" -ForegroundColor Cyan
git status

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "1. Create a GitHub repository:" -ForegroundColor White
Write-Host "   - Go to https://github.com/new" -ForegroundColor Gray
Write-Host "   - Create a new repository (e.g., 'image-classification')`n" -ForegroundColor Gray

Write-Host "2. Run these commands (replace YOUR_USERNAME and YOUR_REPO):`n" -ForegroundColor White

Write-Host "   # Stage all changes" -ForegroundColor Green
Write-Host "   git add ." -ForegroundColor Gray

Write-Host "`n   # Commit changes" -ForegroundColor Green
Write-Host '   git commit -m "Optimized training pipeline with AMP, mixup, and Jupyter support"' -ForegroundColor Gray

Write-Host "`n   # Add remote (first time only)" -ForegroundColor Green
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git" -ForegroundColor Gray

Write-Host "`n   # Push to GitHub" -ForegroundColor Green
Write-Host "   git push -u origin main`n" -ForegroundColor Gray

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Quick Command (Copy & Edit):" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

$quickCommands = @"
git add .
git commit -m "Optimized training pipeline"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
"@

Write-Host $quickCommands -ForegroundColor White

Write-Host "`n⚠️  Remember to:" -ForegroundColor Yellow
Write-Host "   - Replace YOUR_USERNAME with your GitHub username" -ForegroundColor Gray
Write-Host "   - Replace YOUR_REPO with your repository name" -ForegroundColor Gray
Write-Host "   - Ensure .gitignore excludes large files (already configured)`n" -ForegroundColor Gray

Write-Host "========================================`n" -ForegroundColor Cyan

# Offer to run git add automatically
$response = Read-Host "Do you want to stage all files now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "`n📦 Staging files..." -ForegroundColor Cyan
    git add .
    Write-Host "✅ Files staged!`n" -ForegroundColor Green
    
    git status
    
    Write-Host "`nNext: Run the commit and push commands above." -ForegroundColor Yellow
}

Write-Host "`n✅ Script complete!`n" -ForegroundColor Green
