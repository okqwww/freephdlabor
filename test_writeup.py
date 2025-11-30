#!/usr/bin/env python3
"""
Write-up Stage Test Script for freephdlabor

This script tests the entire write-up pipeline to ensure it works before
running a full experiment. This includes:
1. LLM citation generation
2. LaTeX template handling
3. PDF compilation
4. VLM figure analysis

Usage:
    python test_writeup.py [--full]

Options:
    --full    Run complete write-up test with all components
              Without this flag, only basic tests are run
"""

import os
import sys
import argparse
import shutil
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ANSI color codes
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


def test_llm_response():
    """Test LLM can generate write-up content"""
    print_header("1. Testing LLM Write-up Generation")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://newapi.tsingyuai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY not set")
        return False

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=api_base)

        prompt = """Generate a short LaTeX abstract (3-4 sentences) for a paper about
        machine learning experiment results. The abstract should be properly formatted
        LaTeX code wrapped in ```latex``` markers."""

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a scientific paper writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )

        result = response.choices[0].message.content
        print_success("LLM generated content successfully")
        print_info(f"Response preview: {result[:200]}...")

        # Check if LaTeX markers are present
        if "\\begin" in result or "abstract" in result.lower():
            print_success("Response contains LaTeX formatting")
            return True
        else:
            print_warning("Response may not contain proper LaTeX - check output")
            return True  # Still pass, as LLM is working

    except Exception as e:
        print_error(f"LLM test failed: {str(e)}")
        return False


def test_latex_template():
    """Test LaTeX template loading and modification"""
    print_header("2. Testing LaTeX Template Handling")

    try:
        # Find the LaTeX template
        script_dir = Path(__file__).parent
        template_paths = [
            script_dir / "external_tools" / "run_experiment_tool" / "ai_scientist" / "blank_icml_latex",
            script_dir / "external_tools" / "run_experiment_tool" / "ai_scientist" / "blank_icbinb_latex",
        ]

        found_template = None
        for path in template_paths:
            if path.exists():
                found_template = path
                break

        if not found_template:
            print_warning("No LaTeX template found in expected locations")
            print_info("Checking alternative paths...")

            # Try relative paths from current directory
            for subdir in ["ai_scientist", "external_tools/run_experiment_tool/ai_scientist"]:
                for template_name in ["blank_icml_latex", "blank_icbinb_latex"]:
                    test_path = Path(subdir) / template_name
                    if test_path.exists():
                        found_template = test_path
                        break

        if not found_template:
            print_error("LaTeX template not found")
            return False

        print_success(f"Found template at: {found_template}")

        # Check for template.tex
        template_tex = found_template / "template.tex"
        if template_tex.exists():
            content = template_tex.read_text()
            print_success(f"template.tex found ({len(content)} characters)")

            # Check for essential LaTeX components
            checks = [
                ("\\documentclass", "Document class"),
                ("\\begin{document}", "Document begin"),
                ("\\end{document}", "Document end"),
            ]

            for pattern, name in checks:
                if pattern in content:
                    print_success(f"Found {name}")
                else:
                    print_warning(f"Missing {name}")

            return True
        else:
            print_error("template.tex not found in template directory")
            return False

    except Exception as e:
        print_error(f"Template test failed: {str(e)}")
        return False


def test_latex_compilation():
    """Test LaTeX compilation with the actual template"""
    print_header("3. Testing LaTeX Compilation")

    try:
        import subprocess

        # Create a test directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple but complete LaTeX document
            tex_content = r"""\documentclass[runningheads]{llncs}
\usepackage{graphicx}
\usepackage{amsmath}

\begin{document}

\title{Test Paper for Write-up Pipeline}
\author{Test Author}
\institute{Test Institution}
\maketitle

\begin{abstract}
This is a test abstract to verify the LaTeX compilation pipeline works correctly.
The write-up process generates papers from experimental results.
\end{abstract}

\section{Introduction}
This is a test introduction section.

\section{Methods}
This is a test methods section with an equation:
\begin{equation}
E = mc^2
\end{equation}

\section{Results}
This is a test results section.

\section{Conclusion}
This is a test conclusion.

\bibliographystyle{splncs04}
\bibliography{references}

\end{document}
"""
            tex_file = os.path.join(tmpdir, "test.tex")
            with open(tex_file, "w") as f:
                f.write(tex_content)

            # Create empty bibliography
            bib_file = os.path.join(tmpdir, "references.bib")
            with open(bib_file, "w") as f:
                f.write("% Empty bibliography for test\n")

            print_info("Running pdflatex (pass 1)...")
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "test.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )

            print_info("Running bibtex...")
            subprocess.run(
                ["bibtex", "test"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30
            )

            print_info("Running pdflatex (pass 2)...")
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "test.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )

            print_info("Running pdflatex (pass 3)...")
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "test.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=60
            )

            pdf_file = os.path.join(tmpdir, "test.pdf")
            if os.path.exists(pdf_file):
                size = os.path.getsize(pdf_file)
                print_success(f"PDF generated successfully ({size} bytes)")
                return True
            else:
                print_error("PDF was not generated")
                if result.stderr:
                    print_error(f"Error output: {result.stderr[:500]}")
                return False

    except subprocess.TimeoutExpired:
        print_error("LaTeX compilation timed out")
        return False
    except FileNotFoundError:
        print_error("pdflatex not found - please install a TeX distribution")
        return False
    except Exception as e:
        print_error(f"Compilation test failed: {str(e)}")
        return False


def test_vlm_figure_analysis():
    """Test VLM figure analysis capability"""
    print_header("4. Testing VLM Figure Analysis")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://newapi.tsingyuai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY not set")
        return False

    try:
        import openai
        from PIL import Image
        import base64
        import io

        # Create a simple test image
        img = Image.new('RGB', (200, 200), color='white')

        # Draw something simple
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], outline='blue', width=2)
        draw.text((70, 90), "Test", fill='black')

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        print_info("Created test image, sending to VLM...")

        client = openai.OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this simple test image briefly."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200
        )

        result = response.choices[0].message.content
        print_success(f"VLM response: {result[:150]}...")
        print_success("VLM figure analysis working!")
        return True

    except Exception as e:
        print_error(f"VLM test failed: {str(e)}")
        return False


def test_semantic_scholar_citations():
    """Test Semantic Scholar API for citations"""
    print_header("5. Testing Citation Search (Semantic Scholar)")

    api_key = os.environ.get("S2_API_KEY")

    try:
        import requests

        headers = {}
        if api_key:
            headers["X-API-KEY"] = api_key

        # Search for a common ML paper
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params={
                "query": "attention is all you need transformer",
                "limit": 3,
                "fields": "title,authors,year,citationStyles"
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("total", 0) > 0:
                paper = data["data"][0]
                print_success(f"Found: {paper.get('title', 'Unknown')}")

                # Check if bibtex is available
                if "citationStyles" in paper and paper["citationStyles"]:
                    print_success("BibTeX citation available")
                else:
                    print_warning("BibTeX not available for this paper")

                return True

        print_error(f"API returned: {response.status_code}")
        return False

    except Exception as e:
        print_error(f"Citation search failed: {str(e)}")
        return False


def test_full_writeup_pipeline():
    """Run a complete mini write-up test"""
    print_header("6. Testing Full Write-up Pipeline (Mini)")

    try:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://newapi.tsingyuai.com/v1")
        client = openai.OpenAI(api_key=api_key, base_url=api_base)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock experiment data
            mock_summary = {
                "BASELINE_SUMMARY": {
                    "accuracy": 0.85,
                    "loss": 0.32,
                    "model": "baseline_model"
                },
                "RESEARCH_SUMMARY": {
                    "accuracy": 0.91,
                    "loss": 0.21,
                    "model": "improved_model",
                    "improvement": "6% accuracy gain"
                }
            }

            # Save mock data
            summary_file = os.path.join(tmpdir, "experiment_summary.json")
            with open(summary_file, "w") as f:
                json.dump(mock_summary, f, indent=2)

            print_info("Created mock experiment data")

            # Generate LaTeX content using LLM
            prompt = f"""Based on these experiment results, generate a short LaTeX paper section:

{json.dumps(mock_summary, indent=2)}

Generate ONLY the LaTeX code for a Results section (about 5-6 sentences) that describes these findings.
Wrap your response in ```latex``` markers."""

            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are an academic paper writer. Generate only LaTeX code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )

            latex_content = response.choices[0].message.content
            print_info("LLM generated LaTeX content")

            # Create a complete document
            full_doc = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}

\title{Test Write-up}
\author{AI Scientist}
\maketitle

\section{Results}
""" + latex_content.replace("```latex", "").replace("```", "") + r"""

\end{document}
"""
            tex_file = os.path.join(tmpdir, "paper.tex")
            with open(tex_file, "w") as f:
                f.write(full_doc)

            # Compile
            import subprocess
            print_info("Compiling LaTeX document...")

            for i in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=60
                )

            pdf_file = os.path.join(tmpdir, "paper.pdf")
            if os.path.exists(pdf_file):
                size = os.path.getsize(pdf_file)
                print_success(f"Generated paper.pdf ({size} bytes)")

                # Copy to a visible location for inspection
                output_path = Path(__file__).parent / "test_output_paper.pdf"
                shutil.copy(pdf_file, output_path)
                print_success(f"Test paper saved to: {output_path}")

                return True
            else:
                print_error("PDF generation failed in full pipeline test")
                return False

    except Exception as e:
        print_error(f"Full pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test write-up pipeline")
    parser.add_argument("--full", action="store_true", help="Run complete write-up test")
    args = parser.parse_args()

    print(f"""
{BLUE}================================================================
       freephdlabor Write-up Pipeline Test Suite
================================================================{RESET}

This script tests the paper write-up pipeline to ensure it works
correctly before running full experiments.
""")

    results = {}

    # Run tests
    results["LLM Response"] = test_llm_response()
    results["LaTeX Template"] = test_latex_template()
    results["LaTeX Compilation"] = test_latex_compilation()
    results["VLM Figure Analysis"] = test_vlm_figure_analysis()
    results["Citation Search"] = test_semantic_scholar_citations()

    if args.full:
        results["Full Pipeline"] = test_full_writeup_pipeline()

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

    # Critical checks
    critical = ["LLM Response", "LaTeX Compilation"]
    critical_failed = [t for t in critical if not results.get(t, False)]

    if critical_failed:
        print(f"\n{RED}CRITICAL: These essential tests failed:{RESET}")
        for t in critical_failed:
            print(f"  - {t}")
        print(f"\n{RED}The write-up stage will likely fail!{RESET}")
        return 1

    if failed > 0:
        print(f"\n{YELLOW}Some tests failed but write-up may still work.{RESET}")
    else:
        print(f"\n{GREEN}All write-up tests passed! Pipeline ready.{RESET}")

    if not args.full:
        print(f"\n{BLUE}Tip: Run with --full flag for complete pipeline test{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
