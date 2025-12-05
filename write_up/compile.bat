@echo off
REM ============================================================================
REM QC-Amp Write-up Compilation Script
REM ============================================================================
REM
REM This script compiles the LaTeX document to PDF.
REM 
REM Requirements:
REM   - A LaTeX distribution (MiKTeX, TeX Live, or similar)
REM   - pdflatex in PATH
REM
REM Usage:
REM   compile.bat
REM
REM ============================================================================

REM Change to the directory where this script is located
cd /d "%~dp0"

echo ============================================
echo QC-Amp Thesis/Report Compilation
echo ============================================
echo Working directory: %cd%

REM Run pdflatex three times for cross-references
echo.
echo [1/3] First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex

echo.
echo [2/3] Second pdflatex pass (cross-references)...
pdflatex -interaction=nonstopmode main.tex

echo.
echo [3/3] Third pdflatex pass (final)...
pdflatex -interaction=nonstopmode main.tex

echo.
echo ============================================
echo Compilation complete!
echo Output: main.pdf
echo ============================================

REM Clean up auxiliary files (optional - uncomment if desired)
REM del *.aux *.log *.out *.toc *.lof *.lot

pause
