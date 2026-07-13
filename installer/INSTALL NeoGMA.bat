@echo off
REM ===========================================================================
REM  NeoGMA — double-click this file. That is the whole installation.
REM
REM  It installs Python (if you don't have it), PyTorch matched to your GPU,
REM  the ViTPose-H pose model, and puts a NeoGMA icon on your Desktop.
REM  No Docker. No compiler. No terminal.
REM
REM  Windows will ask for permission because the script installs software.
REM ===========================================================================
title Installing NeoGMA
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-NeoGMA.ps1"
