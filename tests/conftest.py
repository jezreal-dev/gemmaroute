"""
tests/conftest.py — Shared pytest fixtures for GemmaRoute tests.

Adds the backend directory to sys.path so all test files can import
backend modules directly without installing the package.
"""
import sys
import os

# Make backend importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
