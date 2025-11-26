#!/usr/bin/env python3
"""
Experiment Pipeline Test Script for freephdlabor

This script tests the experiment execution pipeline (RunExperimentTool and AI-Scientist-v2)
to ensure it works before running full experiments. This is the most time-consuming part
of the pipeline, so catching errors early is critical.

Usage:
    python test_experiment.py [--quick] [--stage1-only]

Options:
    --quick         Run minimal validation only (no actual experiment)
    --stage1-only   Run only stage 1 (initial implementation) - fastest real test
"""

import os
import sys
import json
import argparse
import tempfile
import shutil
import subprocess
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


def test_ai_scientist_structure():
    """Test AI-Scientist-v2 directory structure and dependencies"""
    print_header("1. Testing AI-Scientist-v2 Structure")

    script_dir = Path(__file__).parent
    ai_scientist_path = script_dir / "external_tools" / "run_experiment_tool"

    if not ai_scientist_path.exists():
        print_error(f"AI-Scientist-v2 directory not found: {ai_scientist_path}")
        return False

    print_success(f"Found AI-Scientist-v2 at: {ai_scientist_path}")

    # Check required files
    required_files = [
        "launch_scientist_bfts.py",
        "bfts_config.yaml",
        "ai_scientist/__init__.py",
        "ai_scientist/llm.py",
        "ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py",
    ]

    missing_files = []
    for file in required_files:
        file_path = ai_scientist_path / file
        if file_path.exists():
            print_success(f"Found: {file}")
        else:
            print_error(f"Missing: {file}")
            missing_files.append(file)

    if missing_files:
        print_error(f"Missing {len(missing_files)} required files")
        return False

    print_success("All required AI-Scientist-v2 files present")
    return True


def test_experiment_dependencies():
    """Test that experiment dependencies are importable"""
    print_header("2. Testing Experiment Dependencies")

    dependencies = [
        ("torch", "PyTorch for ML experiments"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("yaml", "YAML config parsing"),
        ("litellm", "LLM API wrapper"),
        ("smolagents", "Agent framework"),
    ]

    all_passed = True
    for module, description in dependencies:
        try:
            __import__(module)
            print_success(f"{module}: {description}")
        except ImportError as e:
            print_error(f"{module}: {str(e)}")
            all_passed = False

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print_success(f"CUDA available: {gpu_count} GPU(s) - {gpu_name}")
        else:
            print_warning("CUDA not available - experiments will run on CPU (slower)")
    except Exception as e:
        print_warning(f"Could not check CUDA: {e}")

    return all_passed


def test_idea_standardization():
    """Test IdeaStandardizationTool functionality"""
    print_header("3. Testing Idea Standardization Tool")

    try:
        from freephdlabor.toolkits.idea_standardization_tool import IdeaStandardizationTool
        from smolagents import LiteLLMModel

        # Create a minimal model for testing
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print_warning("OPENAI_API_KEY not set, skipping LLM-based test")
            return True

        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )

        tool = IdeaStandardizationTool(model=model)
        print_success("IdeaStandardizationTool initialized")

        # Test with a simple idea
        test_idea = {
            "title": "Test Research Idea",
            "research_question": "Can we improve model performance?",
            "methodology": {
                "approach": "Fine-tuning with custom loss",
                "baselines": ["Standard fine-tuning", "LoRA"]
            }
        }

        print_info("Converting test idea to AI-Scientist-v2 format...")
        result = tool.forward(json.dumps(test_idea))

        # Validate result
        result_data = json.loads(result)
        if isinstance(result_data, list) and len(result_data) > 0:
            idea = result_data[0]
            required_fields = ["Name", "Title", "Short Hypothesis", "Abstract", "Experiments"]
            missing = [f for f in required_fields if f not in idea]

            if missing:
                print_error(f"Converted idea missing fields: {missing}")
                return False

            print_success("Idea converted successfully")
            print_info(f"  Name: {idea['Name']}")
            print_info(f"  Title: {idea['Title'][:50]}...")
            return True
        else:
            print_error("Conversion returned invalid format")
            return False

    except Exception as e:
        print_error(f"IdeaStandardizationTool test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_run_experiment_tool_init():
    """Test RunExperimentTool initialization"""
    print_header("4. Testing RunExperimentTool Initialization")

    try:
        from freephdlabor.toolkits.run_experiment_tool import RunExperimentTool

        with tempfile.TemporaryDirectory() as tmpdir:
            tool = RunExperimentTool(workspace_dir=tmpdir)
            print_success("RunExperimentTool initialized")
            print_info(f"  AI-Scientist path: {tool.ai_scientist_path}")
            print_info(f"  Workspace: {tool.workspace_dir}")

            # Check AI-Scientist path exists
            if os.path.exists(tool.ai_scientist_path):
                print_success("AI-Scientist path exists")
            else:
                print_error(f"AI-Scientist path not found: {tool.ai_scientist_path}")
                return False

            # Check Python executable detection
            python_exec = tool._get_python_executable()
            print_info(f"  Python executable: {python_exec}")

            if os.path.exists(python_exec) or python_exec == "python":
                print_success("Python executable found")
            else:
                print_warning(f"Python executable may not exist: {python_exec}")

            return True

    except Exception as e:
        print_error(f"RunExperimentTool init failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_bfts_config():
    """Test BFTS config loading and validation"""
    print_header("5. Testing BFTS Configuration")

    try:
        import yaml

        script_dir = Path(__file__).parent
        config_path = script_dir / "external_tools" / "run_experiment_tool" / "bfts_config.yaml"

        if not config_path.exists():
            print_error(f"Config not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print_success("Config loaded successfully")

        # Check important settings
        checks = [
            ("agent.code.model", config.get('agent', {}).get('code', {}).get('model')),
            ("agent.feedback.model", config.get('agent', {}).get('feedback', {}).get('model')),
            ("agent.num_workers", config.get('agent', {}).get('num_workers')),
            ("exec.timeout", config.get('exec', {}).get('timeout')),
        ]

        for name, value in checks:
            if value is not None:
                print_info(f"  {name}: {value}")
            else:
                print_warning(f"  {name}: not set")

        return True

    except Exception as e:
        print_error(f"Config test failed: {str(e)}")
        return False


def test_llm_for_experiments():
    """Test LLM connectivity for experiment code generation"""
    print_header("6. Testing LLM for Code Generation")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY not set")
        return False

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=api_base)

        # Test code generation capability
        prompt = """Write a simple Python function that:
1. Takes a list of numbers
2. Returns the mean and standard deviation
Keep it short (under 10 lines)."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python code assistant. Output only code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        result = response.choices[0].message.content

        if "def " in result and "return" in result:
            print_success("LLM can generate Python code")
            print_info(f"Response preview: {result[:100]}...")
            return True
        else:
            print_warning("LLM response may not contain valid code")
            return True  # Still pass, LLM is working

    except Exception as e:
        print_error(f"LLM code generation test failed: {str(e)}")
        return False


def test_mini_experiment(stage1_only=False):
    """Run a minimal experiment to test the full pipeline"""
    print_header("7. Testing Mini Experiment Execution")

    print_warning("This test runs a real (but minimal) experiment.")
    print_warning("It may take 5-15 minutes depending on your hardware.")

    try:
        from freephdlabor.toolkits.run_experiment_tool import RunExperimentTool
        from freephdlabor.toolkits.idea_standardization_tool import IdeaStandardizationTool
        from smolagents import LiteLLMModel

        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            print_error("OPENAI_API_KEY required for mini experiment")
            return False

        # Create model for standardization
        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )

        # Create a minimal test idea
        test_idea = {
            "title": "Pipeline Test: Simple Classification",
            "research_question": "Verify the experiment pipeline works correctly",
            "methodology": {
                "approach": "Train a tiny model on minimal data",
                "procedure": [
                    "Load a small subset of data (100 samples)",
                    "Train for 1 epoch",
                    "Report accuracy"
                ]
            }
        }

        print_info("Step 1: Standardizing idea...")
        standardizer = IdeaStandardizationTool(model=model)
        standardized_idea = standardizer.forward(json.dumps(test_idea))
        print_success("Idea standardized")

        # Create workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            print_info(f"Step 2: Creating experiment in {tmpdir}")

            tool = RunExperimentTool(workspace_dir=tmpdir)

            # Set environment for minimal run
            os.environ['POC_MODE'] = 'true'

            end_stage = 1 if stage1_only else 2
            print_info(f"Step 3: Running experiment (stages 1-{end_stage})...")
            print_info("This may take several minutes...")

            result = tool.forward(
                idea_json=standardized_idea,
                code_model="gpt-4o-mini",
                feedback_model="gpt-4o-mini",
                vlm_model="gpt-4o-mini",
                report_model="gpt-4o-mini",
                end_stage=end_stage
            )

            result_data = json.loads(result)

            if result_data.get("status") == "success":
                print_success("Mini experiment completed successfully!")
                print_info(f"Results directory: {result_data.get('results_directory')}")
                return True
            else:
                print_error(f"Experiment failed: {result_data.get('summary', 'Unknown error')}")
                return False

    except Exception as e:
        print_error(f"Mini experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test experiment pipeline")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation only (no actual experiment)")
    parser.add_argument("--stage1-only", action="store_true",
                       help="Run only stage 1 in mini experiment (faster)")
    args = parser.parse_args()

    print(f"""
{BLUE}================================================================
       freephdlabor Experiment Pipeline Test Suite
================================================================{RESET}

This script tests the experiment execution pipeline to ensure
it works before running long experiments.
""")

    results = {}

    # Core tests (always run)
    results["AI-Scientist Structure"] = test_ai_scientist_structure()
    results["Experiment Dependencies"] = test_experiment_dependencies()
    results["BFTS Config"] = test_bfts_config()
    results["RunExperimentTool Init"] = test_run_experiment_tool_init()
    results["LLM Code Generation"] = test_llm_for_experiments()
    results["Idea Standardization"] = test_idea_standardization()

    # Optional: Run actual experiment
    if not args.quick:
        print(f"\n{YELLOW}Running mini experiment test...{RESET}")
        print(f"{YELLOW}Use --quick to skip this if you just want validation.{RESET}\n")
        results["Mini Experiment"] = test_mini_experiment(stage1_only=args.stage1_only)
    else:
        print_info("Skipping mini experiment (--quick mode)")

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

    # Critical tests
    critical = ["AI-Scientist Structure", "Experiment Dependencies", "RunExperimentTool Init"]
    critical_failed = [t for t in critical if not results.get(t, False)]

    if critical_failed:
        print(f"\n{RED}CRITICAL: These essential tests failed:{RESET}")
        for t in critical_failed:
            print(f"  - {t}")
        print(f"\n{RED}Experiments will likely fail! Fix these issues first.{RESET}")
        return 1

    if failed > 0:
        print(f"\n{YELLOW}Some tests failed but experiments may still work.{RESET}")
    else:
        print(f"\n{GREEN}All tests passed! Experiment pipeline ready.{RESET}")

    if args.quick:
        print(f"\n{BLUE}Tip: Run without --quick to test actual experiment execution{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
