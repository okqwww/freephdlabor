#!/usr/bin/env python3
"""
API Key/Token Test Script for freephdlabor

This script tests all external API connections to ensure they work before
starting a long-running experiment. Run this before launching experiments.

Usage:
    python test_api_keys.py
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BLUE}{'='*60}")
    print(f" {title}")
    print(f"{'='*60}{RESET}\n")

def print_success(msg):
    print(f"{GREEN}[SUCCESS]{RESET} {msg}")

def print_error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[WARNING]{RESET} {msg}")

def print_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")


def test_newapi_llm():
    """Test NewAPI LLM connectivity"""
    print_header("1. Testing NewAPI LLM (Large Language Model)")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://newapi.tsingyuai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY not set in environment")
        return False

    print_info(f"API Base: {api_base}")
    print_info(f"API Key: {api_key[:20]}...")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=api_base)

        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'API test successful' in exactly 3 words."}],
            max_tokens=20
        )

        result = response.choices[0].message.content
        print_success(f"LLM Response: {result}")
        print_success("NewAPI LLM connection successful!")
        return True

    except Exception as e:
        print_error(f"NewAPI LLM test failed: {str(e)}")
        return False


def test_semantic_scholar():
    """Test Semantic Scholar API connectivity"""
    print_header("2. Testing Semantic Scholar API")

    api_key = os.environ.get("S2_API_KEY")

    if not api_key:
        print_warning("S2_API_KEY not set - using anonymous rate limits (slower)")
    else:
        print_info(f"API Key: {api_key[:20]}...")

    try:
        import requests

        headers = {}
        if api_key:
            headers["X-API-KEY"] = api_key

        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": "transformer attention mechanism",
                "limit": 1,
                "fields": "title,authors,year"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("total", 0) > 0:
                paper = data["data"][0]
                print_success(f"Found paper: {paper.get('title', 'Unknown')}")
                print_success("Semantic Scholar API connection successful!")
                return True
            else:
                print_warning("No papers found, but API is accessible")
                return True
        else:
            print_error(f"API returned status code: {response.status_code}")
            print_error(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print_error(f"Semantic Scholar API test failed: {str(e)}")
        return False


def test_serper_api():
    """Test Serper API connectivity"""
    print_header("3. Testing Serper API (Web Search)")

    api_key = os.environ.get("SERPER_API_KEY")

    if not api_key:
        print_error("SERPER_API_KEY not set in environment")
        return False

    print_info(f"API Key: {api_key[:20]}...")

    try:
        import requests

        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            },
            json={"q": "test query", "num": 1},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if "organic" in data and len(data["organic"]) > 0:
                print_success(f"Found result: {data['organic'][0].get('title', 'Unknown')}")
            print_success("Serper API connection successful!")
            return True
        else:
            print_error(f"API returned status code: {response.status_code}")
            print_error(f"Response: {response.text[:200]}")
            return False

    except Exception as e:
        print_error(f"Serper API test failed: {str(e)}")
        return False


def test_huggingface():
    """Test HuggingFace Hub connectivity"""
    print_header("4. Testing HuggingFace Hub")

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print_warning("HF_TOKEN not set - public datasets only")
    else:
        print_info(f"Token: {hf_token[:20]}...")

    try:
        from huggingface_hub import HfApi, login

        # Login if token provided
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        api = HfApi()

        # Test by listing a small public dataset
        dataset_info = api.dataset_info("imdb")
        print_success(f"Found dataset: {dataset_info.id}")
        print_success(f"Downloads: {dataset_info.downloads}")
        print_success("HuggingFace Hub connection successful!")
        return True

    except Exception as e:
        print_error(f"HuggingFace Hub test failed: {str(e)}")
        return False


def test_huggingface_datasets():
    """Test HuggingFace Datasets download capability"""
    print_header("5. Testing HuggingFace Datasets Download")

    try:
        from datasets import load_dataset

        print_info("Attempting to load a small dataset (first 10 samples)...")

        # Load a small portion of IMDB dataset
        dataset = load_dataset("imdb", split="train[:10]")

        print_success(f"Loaded {len(dataset)} samples")
        print_success(f"Columns: {dataset.column_names}")
        print_success("HuggingFace Datasets download successful!")
        return True

    except Exception as e:
        print_error(f"HuggingFace Datasets test failed: {str(e)}")
        return False


def test_newapi_embeddings():
    """Test NewAPI Embeddings capability"""
    print_header("6. Testing NewAPI Embeddings")

    api_key = os.environ.get("OPENAI_API_KEY_EMBEDDINGS") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://newapi.tsingyuai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY not set for embeddings")
        return False

    print_info(f"API Base: {api_base}")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=api_base)

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="Test embedding text"
        )

        embedding = response.data[0].embedding
        print_success(f"Embedding dimension: {len(embedding)}")
        print_success("NewAPI Embeddings successful!")
        return True

    except Exception as e:
        print_error(f"NewAPI Embeddings test failed: {str(e)}")
        print_warning("Note: Embeddings are used for knowledge base indexing. If not needed, this is OK.")
        return False


def test_latex_compilation():
    """Test LaTeX compilation capability"""
    print_header("7. Testing LaTeX Compilation")

    try:
        import subprocess
        import tempfile
        import shutil

        # Check if pdflatex is available
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print_success(f"pdflatex found: {version_line}")
        else:
            print_error("pdflatex not found or not working")
            return False

        # Check if bibtex is available
        result = subprocess.run(
            ["bibtex", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print_success("bibtex found")
        else:
            print_warning("bibtex not found - citations may not work")

        # Try a simple compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = os.path.join(tmpdir, "test.tex")
            with open(tex_file, "w") as f:
                f.write(r"""\documentclass{article}
\begin{document}
Hello, World!
\end{document}
""")

            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "test.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )

            pdf_file = os.path.join(tmpdir, "test.pdf")
            if os.path.exists(pdf_file):
                print_success("LaTeX compilation test successful!")
                return True
            else:
                print_error("LaTeX compilation failed to produce PDF")
                print_error(result.stderr[:500] if result.stderr else "No error output")
                return False

    except FileNotFoundError:
        print_error("pdflatex not found in PATH")
        print_info("Please install a TeX distribution (e.g., TeX Live, MiKTeX)")
        return False
    except Exception as e:
        print_error(f"LaTeX test failed: {str(e)}")
        return False


def main():
    print(f"""
{BLUE}================================================================
       freephdlabor API Key/Token Test Suite
================================================================{RESET}

This script tests all external API connections required for
running experiments. Please ensure all tests pass before
starting long-running experiments.
""")

    results = {}

    # Run all tests
    results["NewAPI LLM"] = test_newapi_llm()
    results["Semantic Scholar"] = test_semantic_scholar()
    results["Serper API"] = test_serper_api()
    results["HuggingFace Hub"] = test_huggingface()
    results["HuggingFace Datasets"] = test_huggingface_datasets()
    results["NewAPI Embeddings"] = test_newapi_embeddings()
    results["LaTeX Compilation"] = test_latex_compilation()

    # Print summary
    print_header("TEST SUMMARY")

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n{BLUE}Total: {passed}/{passed+failed} tests passed{RESET}")

    # Critical tests check
    critical_tests = ["NewAPI LLM", "HuggingFace Datasets", "LaTeX Compilation"]
    critical_failed = [t for t in critical_tests if not results.get(t, False)]

    if critical_failed:
        print(f"\n{RED}CRITICAL:{RESET} The following essential tests failed:")
        for t in critical_failed:
            print(f"  - {t}")
        print(f"\n{RED}Please fix these issues before running experiments!{RESET}")
        return 1

    if failed > 0:
        print(f"\n{YELLOW}WARNING:{RESET} Some non-critical tests failed.")
        print("The experiment may still work, but some features might be unavailable.")
    else:
        print(f"\n{GREEN}All tests passed! Ready to run experiments.{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
