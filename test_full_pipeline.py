#!/usr/bin/env python3
"""
Full Pipeline Smoke Test for freephdlabor

This script runs a comprehensive pre-flight check of the entire freephdlabor system.
It tests all components in sequence to identify potential failures before
starting a long-running experiment.

Usage:
    python test_full_pipeline.py [--skip-experiment] [--verbose]

Options:
    --skip-experiment   Skip the actual experiment execution test
    --verbose          Show detailed output for all tests

Recommended workflow:
    1. First run: python test_full_pipeline.py --skip-experiment
       (Quick check of all components, ~2-5 minutes)

    2. If all pass: python test_full_pipeline.py
       (Full test including mini experiment, ~15-30 minutes)

    3. If all pass: python launch_multiagent.py --task "your research task"
"""

import os
import sys
import json
import argparse
import tempfile
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BLUE}{BOLD}{'='*70}")
    print(f" {title}")
    print(f"{'='*70}{RESET}\n")

def print_section(title):
    print(f"\n{CYAN}--- {title} ---{RESET}\n")

def print_success(msg):
    print(f"{GREEN}[PASS]{RESET} {msg}")

def print_error(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")

def print_info(msg):
    print(f"{BLUE}[INFO]{RESET} {msg}")

def print_skip(msg):
    print(f"{YELLOW}[SKIP]{RESET} {msg}")


class PipelineTestSuite:
    """Comprehensive test suite for the freephdlabor pipeline"""

    def __init__(self, verbose=False, skip_experiment=False):
        self.verbose = verbose
        self.skip_experiment = skip_experiment
        self.results = {}
        self.warnings = []
        self.start_time = None

    def run_all(self):
        """Run all pipeline tests"""
        self.start_time = time.time()

        print(f"""
{BLUE}{BOLD}================================================================
       freephdlabor Full Pipeline Smoke Test
================================================================{RESET}

This comprehensive test validates all system components before
running long experiments. Estimated time: {'5-10' if self.skip_experiment else '15-30'} minutes.

Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

        # Phase 1: Environment
        print_header("PHASE 1: Environment & Configuration")
        self.results["1.1 Environment Variables"] = self.test_env_vars()
        self.results["1.2 Python Dependencies"] = self.test_python_deps()
        self.results["1.3 System Tools"] = self.test_system_tools()
        self.results["1.4 GPU Availability"] = self.test_gpu()

        # Phase 2: API Connectivity
        print_header("PHASE 2: API Connectivity")
        self.results["2.1 OpenAI/NewAPI LLM"] = self.test_llm_api()
        self.results["2.2 Semantic Scholar"] = self.test_semantic_scholar()
        self.results["2.3 Serper Search"] = self.test_serper()
        self.results["2.4 HuggingFace Hub"] = self.test_huggingface()

        # Phase 3: Core Framework
        print_header("PHASE 3: Core Framework")
        self.results["3.1 Smolagents Framework"] = self.test_smolagents()
        self.results["3.2 Agent Initialization"] = self.test_agent_init()
        self.results["3.3 Tool System"] = self.test_tools()
        self.results["3.4 Workspace Management"] = self.test_workspace()

        # Phase 4: Experiment Pipeline
        print_header("PHASE 4: Experiment Pipeline")
        self.results["4.1 AI-Scientist Structure"] = self.test_ai_scientist()
        self.results["4.2 Idea Standardization"] = self.test_idea_standardization()
        self.results["4.3 RunExperimentTool"] = self.test_run_experiment_tool()

        if not self.skip_experiment:
            self.results["4.4 Mini Experiment"] = self.test_mini_experiment()
        else:
            print_skip("4.4 Mini Experiment (use without --skip-experiment to test)")

        # Phase 5: Write-up Pipeline
        print_header("PHASE 5: Write-up Pipeline")
        self.results["5.1 LaTeX Installation"] = self.test_latex()
        self.results["5.2 VLM Figure Analysis"] = self.test_vlm()
        self.results["5.3 Citation Search"] = self.test_citations()

        # Generate report
        self.print_report()

        return all(v for v in self.results.values() if v is not None)

    def test_env_vars(self):
        """Test required environment variables"""
        print_section("Checking Environment Variables")

        required = {
            "OPENAI_API_KEY": "LLM API access",
        }

        optional = {
            "OPENAI_BASE_URL": "Custom API endpoint",
            "SERPER_API_KEY": "Web search",
            "S2_API_KEY": "Semantic Scholar",
            "HF_TOKEN": "HuggingFace private models",
            "GOOGLE_API_KEY": "Google/Gemini models",
            "ANTHROPIC_API_KEY": "Claude models",
        }

        all_required = True
        for var, desc in required.items():
            value = os.environ.get(var)
            if value:
                print_success(f"{var}: Set ({desc})")
            else:
                print_error(f"{var}: NOT SET - Required for {desc}")
                all_required = False

        for var, desc in optional.items():
            value = os.environ.get(var)
            if value:
                print_success(f"{var}: Set ({desc})")
            else:
                print_warning(f"{var}: Not set ({desc})")
                self.warnings.append(f"{var} not set - {desc} may not work")

        return all_required

    def test_python_deps(self):
        """Test Python dependencies"""
        print_section("Checking Python Dependencies")

        deps = [
            ("torch", "PyTorch"),
            ("transformers", "HuggingFace Transformers"),
            ("datasets", "HuggingFace Datasets"),
            ("smolagents", "Agent Framework"),
            ("litellm", "LLM Wrapper"),
            ("openai", "OpenAI Client"),
            ("yaml", "YAML Parser"),
            ("PIL", "Image Processing"),
            ("requests", "HTTP Client"),
        ]

        all_passed = True
        for module, name in deps:
            try:
                __import__(module)
                print_success(f"{name} ({module})")
            except ImportError as e:
                print_error(f"{name} ({module}): {e}")
                all_passed = False

        return all_passed

    def test_system_tools(self):
        """Test system tools availability"""
        print_section("Checking System Tools")

        import subprocess

        tools = [
            (["pdflatex", "--version"], "pdflatex"),
            (["bibtex", "--version"], "bibtex"),
        ]

        all_passed = True
        for cmd, name in tools:
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.returncode == 0:
                    print_success(f"{name}")
                else:
                    print_error(f"{name}: returned non-zero")
                    all_passed = False
            except FileNotFoundError:
                print_error(f"{name}: NOT FOUND")
                all_passed = False
            except Exception as e:
                print_error(f"{name}: {e}")
                all_passed = False

        return all_passed

    def test_gpu(self):
        """Test GPU availability"""
        print_section("Checking GPU")

        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print_success(f"CUDA available: {count} GPU(s)")
                print_info(f"  GPU 0: {name} ({memory:.1f} GB)")
                return True
            else:
                print_warning("CUDA not available - will use CPU (slower)")
                self.warnings.append("No GPU - experiments will be slow")
                return True  # Not a failure, just slower
        except Exception as e:
            print_warning(f"GPU check failed: {e}")
            return True

    def test_llm_api(self):
        """Test LLM API connectivity"""
        print_section("Testing LLM API")

        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print_error("OPENAI_API_KEY not set")
            return False

        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=api_base)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say 'OK' only."}],
                max_tokens=10
            )

            result = response.choices[0].message.content
            print_success(f"LLM responded: {result}")
            return True

        except Exception as e:
            print_error(f"LLM API test failed: {e}")
            return False

    def test_semantic_scholar(self):
        """Test Semantic Scholar API"""
        print_section("Testing Semantic Scholar")

        try:
            import requests

            api_key = os.environ.get("S2_API_KEY")
            headers = {"X-API-KEY": api_key} if api_key else {}

            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params={"query": "transformer", "limit": 1},
                timeout=30
            )

            if response.status_code == 200:
                print_success("Semantic Scholar API accessible")
                return True
            else:
                print_warning(f"API returned {response.status_code}")
                return True  # Non-critical

        except Exception as e:
            print_warning(f"Semantic Scholar test failed: {e}")
            self.warnings.append("Semantic Scholar may not work")
            return True

    def test_serper(self):
        """Test Serper API"""
        print_section("Testing Serper API")

        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            print_warning("SERPER_API_KEY not set - web search disabled")
            self.warnings.append("Web search will not work without SERPER_API_KEY")
            return True

        try:
            import requests

            response = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": "test", "num": 1},
                timeout=30
            )

            if response.status_code == 200:
                print_success("Serper API working")
                return True
            else:
                print_error(f"Serper returned {response.status_code}")
                return False

        except Exception as e:
            print_error(f"Serper test failed: {e}")
            return False

    def test_huggingface(self):
        """Test HuggingFace Hub"""
        print_section("Testing HuggingFace Hub")

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.dataset_info("imdb")
            print_success(f"HuggingFace Hub accessible (found: {info.id})")
            return True
        except Exception as e:
            print_warning(f"HuggingFace test: {e}")
            self.warnings.append("HuggingFace access may be limited")
            return True

    def test_smolagents(self):
        """Test smolagents framework"""
        print_section("Testing Smolagents Framework")

        try:
            from smolagents import LiteLLMModel, CodeAgent, Tool
            from smolagents.models import ChatMessage
            print_success("Smolagents imports successful")
            return True
        except Exception as e:
            print_error(f"Smolagents import failed: {e}")
            return False

    def test_agent_init(self):
        """Test agent initialization"""
        print_section("Testing Agent Initialization")

        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print_error("OPENAI_API_KEY required")
            return False

        try:
            from freephdlabor.utils import create_model, initialize_agent_system
            from freephdlabor.interpreters import WorkspacePythonExecutor

            model = create_model("gpt-4o", "medium", "low", None)
            print_success("Model created")

            with tempfile.TemporaryDirectory() as tmpdir:
                executor = WorkspacePythonExecutor(
                    workspace_dir=tmpdir,
                    additional_authorized_imports=["json", "os"]
                )

                manager = initialize_agent_system(
                    model=model,
                    workspace_dir=tmpdir,
                    workspace_interpreter=executor,
                    essential_imports=["json", "os"]
                )
                print_success("ManagerAgent initialized with sub-agents")

            return True

        except Exception as e:
            print_error(f"Agent init failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_tools(self):
        """Test tool system"""
        print_section("Testing Tool System")

        try:
            from freephdlabor.toolkits.general_tools.file_editing.file_editing_tools import (
                SeeFile, CreateFileWithContent, ListDir
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                # Test file operations
                create_tool = CreateFileWithContent(working_dir=tmpdir)
                create_tool.forward("test.txt", "Hello World")

                see_tool = SeeFile(working_dir=tmpdir)
                content = see_tool.forward("test.txt")

                if "Hello" in content:
                    print_success("File tools working")
                    return True
                else:
                    print_error("File tools returned unexpected content")
                    return False

        except Exception as e:
            print_error(f"Tool test failed: {e}")
            return False

    def test_workspace(self):
        """Test workspace management"""
        print_section("Testing Workspace Management")

        try:
            from freephdlabor.interpreters import WorkspacePythonExecutor

            with tempfile.TemporaryDirectory() as tmpdir:
                executor = WorkspacePythonExecutor(
                    workspace_dir=tmpdir,
                    additional_authorized_imports=["json"]
                )
                print_success("WorkspacePythonExecutor created")

                # Verify workspace dir
                if os.path.exists(tmpdir):
                    print_success("Workspace directory accessible")
                    return True

        except Exception as e:
            print_error(f"Workspace test failed: {e}")
            return False

    def test_ai_scientist(self):
        """Test AI-Scientist-v2 structure"""
        print_section("Testing AI-Scientist-v2")

        script_dir = Path(__file__).parent
        ai_path = script_dir / "external_tools" / "run_experiment_tool"

        if not ai_path.exists():
            print_error(f"AI-Scientist not found at {ai_path}")
            return False

        required = [
            "launch_scientist_bfts.py",
            "bfts_config.yaml",
            "ai_scientist/llm.py",
        ]

        all_found = True
        for f in required:
            if (ai_path / f).exists():
                print_success(f"Found: {f}")
            else:
                print_error(f"Missing: {f}")
                all_found = False

        return all_found

    def test_idea_standardization(self):
        """Test idea standardization"""
        print_section("Testing Idea Standardization")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print_error("OPENAI_API_KEY required")
            return False

        try:
            from freephdlabor.toolkits.idea_standardization_tool import IdeaStandardizationTool
            from smolagents import LiteLLMModel

            api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = LiteLLMModel(model_id="gpt-4o", api_key=api_key, api_base=api_base)

            tool = IdeaStandardizationTool(model=model)

            test_idea = {"title": "Test", "research_question": "Test question"}
            result = tool.forward(json.dumps(test_idea))

            data = json.loads(result)
            if isinstance(data, list) and len(data) > 0:
                print_success("Idea standardization working")
                return True

            print_error("Standardization returned invalid format")
            return False

        except Exception as e:
            print_error(f"Idea standardization failed: {e}")
            return False

    def test_run_experiment_tool(self):
        """Test RunExperimentTool initialization"""
        print_section("Testing RunExperimentTool")

        try:
            from freephdlabor.toolkits.run_experiment_tool import RunExperimentTool

            with tempfile.TemporaryDirectory() as tmpdir:
                tool = RunExperimentTool(workspace_dir=tmpdir)

                if os.path.exists(tool.ai_scientist_path):
                    print_success("RunExperimentTool initialized")
                    print_info(f"  AI-Scientist path: {tool.ai_scientist_path}")
                    return True
                else:
                    print_error("AI-Scientist path not found")
                    return False

        except Exception as e:
            print_error(f"RunExperimentTool test failed: {e}")
            return False

    def test_mini_experiment(self):
        """Run a minimal experiment"""
        print_section("Testing Mini Experiment (Stage 1 Only)")

        print_warning("This test runs a real experiment and may take 10-20 minutes.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print_error("OPENAI_API_KEY required")
            return False

        try:
            from freephdlabor.toolkits.run_experiment_tool import RunExperimentTool
            from freephdlabor.toolkits.idea_standardization_tool import IdeaStandardizationTool
            from smolagents import LiteLLMModel

            api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = LiteLLMModel(model_id="gpt-4o", api_key=api_key, api_base=api_base)

            # Minimal test idea
            test_idea = {
                "title": "Pipeline Validation Test",
                "research_question": "Validate experiment pipeline",
                "methodology": {"approach": "Train minimal model", "procedure": ["Quick test"]}
            }

            print_info("Standardizing idea...")
            standardizer = IdeaStandardizationTool(model=model)
            standardized = standardizer.forward(json.dumps(test_idea))

            with tempfile.TemporaryDirectory() as tmpdir:
                print_info(f"Running experiment in: {tmpdir}")
                os.environ['POC_MODE'] = 'true'

                tool = RunExperimentTool(workspace_dir=tmpdir)
                result = tool.forward(
                    idea_json=standardized,
                    code_model="gpt-4o",
                    feedback_model="gpt-4o",
                    vlm_model="gpt-4o",
                    report_model="gpt-4o",
                    end_stage=1  # Only stage 1 for speed
                )

                data = json.loads(result)
                if data.get("status") == "success":
                    print_success("Mini experiment completed!")
                    return True
                else:
                    print_error(f"Experiment failed: {data.get('summary', 'Unknown')}")
                    return False

        except Exception as e:
            print_error(f"Mini experiment failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def test_latex(self):
        """Test LaTeX compilation"""
        print_section("Testing LaTeX")

        try:
            import subprocess
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, "test.tex")
                with open(tex_file, "w") as f:
                    f.write(r"""\documentclass{article}
\begin{document}
Test
\end{document}
""")

                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "test.tex"],
                    cwd=tmpdir, capture_output=True, timeout=60
                )

                if os.path.exists(os.path.join(tmpdir, "test.pdf")):
                    print_success("LaTeX compilation working")
                    return True
                else:
                    print_error("LaTeX failed to produce PDF")
                    return False

        except Exception as e:
            print_error(f"LaTeX test failed: {e}")
            return False

    def test_vlm(self):
        """Test VLM for figure analysis"""
        print_section("Testing VLM Figure Analysis")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print_error("OPENAI_API_KEY required")
            return False

        try:
            import openai
            from PIL import Image
            import base64
            import io

            api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            client = openai.OpenAI(api_key=api_key, base_url=api_base)

            # Create test image
            img = Image.new('RGB', (100, 100), color='white')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }],
                max_tokens=50
            )

            print_success("VLM analysis working")
            return True

        except Exception as e:
            print_error(f"VLM test failed: {e}")
            return False

    def test_citations(self):
        """Test citation search"""
        print_section("Testing Citation Search")

        try:
            import requests

            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": "attention transformer", "limit": 1, "fields": "title"},
                timeout=30
            )

            if response.status_code == 200 and response.json().get("data"):
                print_success("Citation search working")
                return True
            else:
                print_warning("Citation search returned no results")
                return True

        except Exception as e:
            print_warning(f"Citation test failed: {e}")
            return True

    def print_report(self):
        """Print final test report"""
        elapsed = time.time() - self.start_time

        print_header("FINAL REPORT")

        # Count results
        passed = sum(1 for v in self.results.values() if v is True)
        failed = sum(1 for v in self.results.values() if v is False)
        skipped = sum(1 for v in self.results.values() if v is None)
        total = len(self.results)

        # Print each result
        for test, result in self.results.items():
            if result is True:
                print(f"  {GREEN}PASS{RESET} {test}")
            elif result is False:
                print(f"  {RED}FAIL{RESET} {test}")
            else:
                print(f"  {YELLOW}SKIP{RESET} {test}")

        print(f"\n{BOLD}Summary:{RESET}")
        print(f"  Passed: {GREEN}{passed}{RESET}")
        print(f"  Failed: {RED}{failed}{RESET}")
        print(f"  Skipped: {YELLOW}{skipped}{RESET}")
        print(f"  Time: {elapsed:.1f} seconds")

        # Print warnings
        if self.warnings:
            print(f"\n{YELLOW}{BOLD}Warnings:{RESET}")
            for w in self.warnings:
                print(f"  - {w}")

        # Final verdict
        print(f"\n{BOLD}{'='*70}{RESET}")

        if failed == 0:
            print(f"{GREEN}{BOLD}ALL CRITICAL TESTS PASSED!{RESET}")
            print(f"{GREEN}The freephdlabor pipeline is ready to run.{RESET}")

            if self.skip_experiment:
                print(f"\n{YELLOW}Recommendation: Run again without --skip-experiment")
                print(f"to fully validate the experiment pipeline.{RESET}")
            else:
                print(f"\n{GREEN}You can now run:{RESET}")
                print(f"  python launch_multiagent.py --task \"your research task\"")
        else:
            print(f"{RED}{BOLD}SOME TESTS FAILED!{RESET}")
            print(f"{RED}Please fix the failed tests before running experiments.{RESET}")

            # Identify critical failures
            critical = ["1.1 Environment Variables", "2.1 OpenAI/NewAPI LLM",
                       "3.1 Smolagents Framework", "3.2 Agent Initialization"]
            critical_failures = [t for t in critical if self.results.get(t) is False]

            if critical_failures:
                print(f"\n{RED}Critical failures that must be fixed:{RESET}")
                for t in critical_failures:
                    print(f"  - {t}")

        print(f"{BOLD}{'='*70}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline smoke test for freephdlabor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick validation (5-10 min):
    python test_full_pipeline.py --skip-experiment

  Full validation (15-30 min):
    python test_full_pipeline.py

  Verbose output:
    python test_full_pipeline.py --verbose
"""
    )
    parser.add_argument("--skip-experiment", action="store_true",
                       help="Skip mini experiment test (faster)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed error output")
    args = parser.parse_args()

    suite = PipelineTestSuite(
        verbose=args.verbose,
        skip_experiment=args.skip_experiment
    )

    success = suite.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
