$ErrorActionPreference = 'Stop'
Write-Host 'Gold Price Prediction V9 — AWS preflight' -ForegroundColor Cyan
Write-Host 'This script checks tools and identity only. It does NOT deploy or create resources.'
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) { throw 'AWS CLI v2 is not installed or not on PATH. Install it, restart PowerShell, and retry.' }
aws --version
if (Get-Command sam -ErrorAction SilentlyContinue) { sam --version } else { Write-Warning 'AWS SAM CLI not found. Install it before deploying template.yaml.' }
if (-not $env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION = 'eu-central-1'; Write-Host 'Using default AWS_DEFAULT_REGION=eu-central-1' }
Write-Host "AWS_PROFILE: $($env:AWS_PROFILE)"
Write-Host "AWS_DEFAULT_REGION: $($env:AWS_DEFAULT_REGION)"
Write-Host 'Calling STS to verify the currently configured identity...'
aws sts get-caller-identity
if ($LASTEXITCODE -ne 0) { throw 'AWS identity check failed. Configure an AWS CLI profile or approved environment credentials.' }
Write-Host 'Preflight passed. Review docs/AWS_IRONHACK_RUNBOOK.md before deploying.' -ForegroundColor Green
