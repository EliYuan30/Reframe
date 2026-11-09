import os

# LLM judge settings; override with the run.py --judge-* flags or these environment variables.
# The API key is read from LLM_JUDGE_API_KEY.
# "openai" calls an OpenAI-compatible endpoint (Gemini's by default); "gemini" calls the native Gemini API.
JUDGE_BACKEND = os.environ.get("LLM_JUDGE_BACKEND") or "openai"
JUDGE_BASE_URL = os.environ.get("LLM_JUDGE_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai"
JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL") or "gemini-3.1-flash-lite"
# Used by the "gemini" backend only.
JUDGE_THINKING_LEVEL = os.environ.get("LLM_JUDGE_THINKING_LEVEL") or "minimal"
