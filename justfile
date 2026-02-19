# Set PowerShell as the default shell on Windows for better compatibility.
set windows-powershell := true
set dotenv-load

# List all available commands. This is the default recipe.
default:
    @just --list

# ==============================================================================
# 🚀 Application
# ==============================================================================
# Commands for running the main application.

# ==============================================================================
# 📦 Dependency Management
# ==============================================================================
# Commands for managing project dependencies with uv.

# Add a new package to the project's virtual environment.
# Usage: just add <package-name>
add package:
    uv add {{package}}

# ==============================================================================
# 📖 Documentation
# ==============================================================================
# Commands for building and serving the project documentation.

# Start a live-reloading local server for documentation.
serve:
    mkdocs serve

# ==============================================================================
# ✅ Git Hooks Management
# ==============================================================================
# Commands for setting up and tearing down Git pre-commit hooks.

# Install pre-commit hooks into the .git/ directory.
install-hooks:
    pre-commit install

# Uninstall pre-commit hooks from the .git/ directory.
uninstall-hooks:
    pre-commit uninstall

# Run all pre-commit hooks against all files.
lint:
    pre-commit run --all-files
