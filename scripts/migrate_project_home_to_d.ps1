param(
  [string]$SourceRoot = "",
  [string]$TargetRoot = "D:\ProjectTheseus\repo",
  [string]$RuntimeRoot = "D:\ProjectTheseus\runtime",
  [switch]$Execute,
  [switch]$CreateCompatibilityJunction,
  [switch]$RemoveBackupAfterJunction,
  [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"

function FullPath([string]$Path) {
  return [System.IO.Path]::GetFullPath($Path)
}

function IsReparsePoint([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    return $false
  }
  $item = Get-Item -LiteralPath $Path -Force
  return (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Get-TreeBytesNoJunctions([string]$Path) {
  $total = [int64]0
  $stack = New-Object System.Collections.Generic.Stack[string]
  $stack.Push($Path)
  while ($stack.Count -gt 0) {
    $current = $stack.Pop()
    foreach ($child in Get-ChildItem -LiteralPath $current -Force -ErrorAction SilentlyContinue) {
      if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        continue
      }
      if ($child.PSIsContainer) {
        $stack.Push($child.FullName)
      } else {
        $total += [int64]$child.Length
      }
    }
  }
  return $total
}

function Write-JsonReport($Report, [string]$Path) {
  $parent = Split-Path -Parent $Path
  if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
  $Report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function New-Junction([string]$Link, [string]$Target) {
  if (Test-Path -LiteralPath $Link) {
    if (IsReparsePoint $Link) {
      return @{ ok = $true; status = "already_reparse_point"; link = $Link; target = $Target }
    }
    $children = @(Get-ChildItem -LiteralPath $Link -Force -ErrorAction SilentlyContinue)
    if ($children.Count -gt 0) {
      return @{ ok = $false; error = "link_path_exists_nonempty"; link = $Link; target = $Target }
    }
    Remove-Item -LiteralPath $Link -Force
  }
  New-Item -ItemType Directory -Force -Path $Target | Out-Null
  $result = & cmd /c mklink /J "$Link" "$Target" 2>&1
  if ($LASTEXITCODE -ne 0) {
    return @{ ok = $false; error = "mklink_failed"; link = $Link; target = $Target; output = $result }
  }
  return @{ ok = $true; status = "created"; link = $Link; target = $Target; output = $result }
}

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
  $SourceRoot = Join-Path $PSScriptRoot ".."
}

$source = (Resolve-Path -LiteralPath $SourceRoot).Path
$source = FullPath $source
$target = FullPath $TargetRoot
$runtime = FullPath $RuntimeRoot
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$reportPath = Join-Path $source "reports\project_home_migration_plan.json"

$report = [ordered]@{
  ok = $false
  policy = "project_theseus_home_migration_to_d_v1"
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  execute_requested = [bool]$Execute
  create_compatibility_junction = [bool]$CreateCompatibilityJunction
  remove_backup_after_junction = [bool]$RemoveBackupAfterJunction
  source = $source
  target = $target
  runtime_root = $runtime
  blockers = @()
  warnings = @()
  actions = @()
}

try {
  $allowedRoot = FullPath "D:\ProjectTheseus"
  if (-not $target.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    $report.blockers += "target_must_be_under_D:\ProjectTheseus"
  }
  if (-not $runtime.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    $report.blockers += "runtime_root_must_be_under_D:\ProjectTheseus"
  }
  if ($target.Equals($source, [System.StringComparison]::OrdinalIgnoreCase)) {
    $report.blockers += "target_equals_source"
  }
  if ($target.StartsWith($source + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    $report.blockers += "target_inside_source"
  }
  if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
    $report.blockers += "source_missing_git_directory"
  }

  $gitDirty = & git -C $source status --porcelain 2>$null
  $report.git_dirty_entries = @($gitDirty)
  if ($gitDirty -and -not $AllowDirty) {
    $report.blockers += "git_workspace_dirty_use_AllowDirty_if_intentional"
  }

  $sourceBytes = Get-TreeBytesNoJunctions $source
  $report.source_size_gib_excluding_junctions = [math]::Round($sourceBytes / 1GB, 3)
  $targetDriveName = ([System.IO.Path]::GetPathRoot($target)).TrimEnd("\").TrimEnd(":")
  $targetDrive = Get-PSDrive -Name $targetDriveName -PSProvider FileSystem
  $report.target_free_gib = [math]::Round($targetDrive.Free / 1GB, 3)
  if ($targetDrive.Free -lt ($sourceBytes * 1.2)) {
    $report.blockers += "target_drive_free_space_below_120_percent_of_source"
  }

  if (Test-Path -LiteralPath $target) {
    $targetChildren = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue)
    $report.target_exists = $true
    $report.target_child_count = $targetChildren.Count
    if ($targetChildren.Count -gt 0 -and -not (Test-Path -LiteralPath (Join-Path $target ".git"))) {
      $report.blockers += "target_exists_nonempty_without_git"
    }
  } else {
    $report.target_exists = $false
  }

  $excludeDirs = @("reports", "checkpoints", "target")
  $robocopyArgs = @(
    $source,
    $target,
    "/MIR",
    "/XJ",
    "/XD"
  ) + $excludeDirs + @(
    "/R:2",
    "/W:2",
    "/COPY:DAT",
    "/DCOPY:DAT",
    "/FFT",
    "/NP"
  )
  $report.actions += [ordered]@{
    action = "copy_source_to_d"
    command = "robocopy " + ($robocopyArgs | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }) -join " "
    excludes = $excludeDirs
  }
  $report.actions += [ordered]@{
    action = "create_target_runtime_junctions"
    junctions = @(
      @{ link = (Join-Path $target "reports"); target = (Join-Path $runtime "reports") },
      @{ link = (Join-Path $target "checkpoints"); target = (Join-Path $runtime "checkpoints") },
      @{ link = (Join-Path $target "target"); target = (Join-Path $runtime "cargo-target") }
    )
  }
  if ($CreateCompatibilityJunction) {
    $backup = "$source.pre-d-migration-$timestamp"
    $report.actions += [ordered]@{
      action = "replace_old_c_path_with_junction"
      backup = $backup
      link = $source
      target = $target
      removes_backup_after_junction = [bool]$RemoveBackupAfterJunction
    }
  } else {
    $report.warnings += "compatibility_junction_not_requested_old_c_path_will_remain_full_copy"
  }

  if ($report.blockers.Count -gt 0) {
    $report.ok = $false
    Write-JsonReport $report $reportPath
    $report | ConvertTo-Json -Depth 12
    exit 2
  }

  if (-not $Execute) {
    $report.ok = $true
    $report.status = "dry_run_only_add_Execute_to_copy"
    Write-JsonReport $report $reportPath
    $report | ConvertTo-Json -Depth 12
    exit 0
  }

  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
  $copyOutput = & robocopy @robocopyArgs
  $robocopyExit = $LASTEXITCODE
  $report.robocopy_exit_code = $robocopyExit
  $report.robocopy_ok = ($robocopyExit -lt 8)
  if ($robocopyExit -ge 8) {
    $report.ok = $false
    $report.blockers += "robocopy_failed_exit_$robocopyExit"
    $report.robocopy_tail = @($copyOutput | Select-Object -Last 40)
    Write-JsonReport $report (Join-Path $target "reports\project_home_migration_plan.json")
    $report | ConvertTo-Json -Depth 12
    exit 3
  }

  $junctionResults = @()
  $junctionResults += New-Junction (Join-Path $target "reports") (Join-Path $runtime "reports")
  $junctionResults += New-Junction (Join-Path $target "checkpoints") (Join-Path $runtime "checkpoints")
  $junctionResults += New-Junction (Join-Path $target "target") (Join-Path $runtime "cargo-target")
  $report.junction_results = $junctionResults
  if ($junctionResults | Where-Object { -not $_.ok }) {
    $report.ok = $false
    $report.blockers += "target_runtime_junction_creation_failed"
    Write-JsonReport $report (Join-Path $target "reports\project_home_migration_plan.json")
    $report | ConvertTo-Json -Depth 12
    exit 4
  }

  $targetGitStatus = & git -C $target status --short --branch 2>&1
  $report.target_git_status = @($targetGitStatus)

  if ($CreateCompatibilityJunction) {
    $backup = "$source.pre-d-migration-$timestamp"
    try {
      Rename-Item -LiteralPath $source -NewName (Split-Path -Leaf $backup)
      $linkResult = New-Junction $source $target
      $report.compatibility_link = $linkResult
      if (-not $linkResult.ok) {
        Rename-Item -LiteralPath $backup -NewName (Split-Path -Leaf $source)
        $report.ok = $false
        $report.blockers += "compatibility_junction_failed_source_restored"
        Write-JsonReport $report (Join-Path $target "reports\project_home_migration_plan.json")
        $report | ConvertTo-Json -Depth 12
        exit 5
      }
      if ($RemoveBackupAfterJunction) {
        if (-not (Test-Path -LiteralPath (Join-Path $backup ".git"))) {
          throw "backup_safety_check_failed_missing_git"
        }
        Remove-Item -LiteralPath $backup -Recurse -Force
        $report.backup_removed = $true
      } else {
        $report.backup_path = $backup
        $report.warnings += "backup_retained_remove_it_after_you_confirm_the_d_home"
      }
    } catch {
      $report.ok = $false
      $report.blockers += "compatibility_junction_phase_failed: $($_.Exception.Message)"
      Write-JsonReport $report (Join-Path $target "reports\project_home_migration_plan.json")
      $report | ConvertTo-Json -Depth 12
      exit 5
    }
  }

  $report.ok = $true
  $report.status = "migration_complete"
  $finalReport = Join-Path $target "reports\project_home_migration_plan.json"
  Write-JsonReport $report $finalReport
  $report | ConvertTo-Json -Depth 12
  exit 0
} catch {
  $report.ok = $false
  $report.blockers += $_.Exception.Message
  Write-JsonReport $report $reportPath
  $report | ConvertTo-Json -Depth 12
  exit 1
}
