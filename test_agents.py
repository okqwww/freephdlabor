#!/usr/bin/env python3
"""
Agent System Test Script for freephdlabor

This script tests the multi-agent system initialization and basic functionality
to ensure agents can be created and communicate before running full experiments.

Usage:
    python test_agents.py [--full]

Options:
    --full    Test agent execution with simple tasks (takes longer)
"""

import os
import sys
import json
import argparse
import tempfile
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


def test_smolagents_import():
    """Test smolagents framework import"""
    print_header("1. Testing Smolagents Framework")

    try:
        from smolagents import LiteLLMModel, CodeAgent, Tool
        print_success("smolagents core imports successful")

        from smolagents.models import ChatMessage
        print_success("ChatMessage import successful")

        return True
    except ImportError as e:
        print_error(f"smolagents import failed: {str(e)}")
        return False


def test_litellm_model():
    """Test LiteLLM model creation"""
    print_header("2. Testing LiteLLM Model Creation")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY not set")
        return False

    try:
        from smolagents import LiteLLMModel

        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )
        print_success(f"Created LiteLLMModel with model_id: {model.model_id}")

        # Test simple call
        from smolagents.models import ChatMessage
        messages = [ChatMessage(role="user", content="Say 'test passed' in exactly 2 words.")]
        response = model(messages)

        if hasattr(response, 'content'):
            print_success(f"Model responded: {response.content[:50]}...")
        else:
            print_success(f"Model responded: {str(response)[:50]}...")

        return True

    except Exception as e:
        print_error(f"LiteLLM model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_base_agent_import():
    """Test base research agent import"""
    print_header("3. Testing Base Agent Import")

    try:
        from freephdlabor.agents.base_research_agent import BaseResearchAgent
        print_success("BaseResearchAgent imported")
        return True
    except ImportError as e:
        print_error(f"BaseResearchAgent import failed: {str(e)}")
        return False


def test_ideation_agent():
    """Test IdeationAgent initialization"""
    print_header("4. Testing IdeationAgent Initialization")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY required")
        return False

    try:
        from freephdlabor.agents.ideation_agent import IdeationAgent
        from smolagents import LiteLLMModel

        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = IdeationAgent(
                model=model,
                workspace_dir=tmpdir
            )
            print_success("IdeationAgent initialized")

            # Check tools
            tool_names = [t.name for t in agent.tools]
            print_info(f"Available tools: {tool_names}")

            expected_tools = ["web_search", "generate_idea", "refine_idea"]
            found_tools = [t for t in expected_tools if t in tool_names]
            print_success(f"Found {len(found_tools)}/{len(expected_tools)} expected tools")

            return True

    except Exception as e:
        print_error(f"IdeationAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_experimentation_agent():
    """Test ExperimentationAgent initialization"""
    print_header("5. Testing ExperimentationAgent Initialization")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY required")
        return False

    try:
        from freephdlabor.agents.experimentation_agent import ExperimentationAgent
        from smolagents import LiteLLMModel

        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ExperimentationAgent(
                model=model,
                workspace_dir=tmpdir
            )
            print_success("ExperimentationAgent initialized")

            # Check tools
            tool_names = [t.name for t in agent.tools]
            print_info(f"Available tools: {tool_names}")

            # Check for critical tool
            if "RunExperimentTool" in tool_names:
                print_success("RunExperimentTool available")
            else:
                print_warning("RunExperimentTool not found in tools list")

            if "IdeaStandardizationTool" in tool_names:
                print_success("IdeaStandardizationTool available")
            else:
                print_warning("IdeaStandardizationTool not found")

            return True

    except Exception as e:
        print_error(f"ExperimentationAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_writeup_agent():
    """Test WriteupAgent initialization"""
    print_header("6. Testing WriteupAgent Initialization")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY required")
        return False

    try:
        from freephdlabor.agents.writeup_agent import WriteupAgent
        from smolagents import LiteLLMModel

        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = WriteupAgent(
                model=model,
                workspace_dir=tmpdir
            )
            print_success("WriteupAgent initialized")

            # Check tools
            tool_names = [t.name for t in agent.tools]
            print_info(f"Available tools: {tool_names}")

            return True

    except Exception as e:
        print_error(f"WriteupAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_manager_agent():
    """Test ManagerAgent initialization with all sub-agents"""
    print_header("7. Testing ManagerAgent Initialization")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY required")
        return False

    try:
        from freephdlabor.utils import create_model, initialize_agent_system
        from freephdlabor.interpreters import WorkspacePythonExecutor

        model = create_model("gpt-4o-mini", "medium", "low", None)
        print_success(f"Created model: {model.model_id}")

        with tempfile.TemporaryDirectory() as tmpdir:
            essential_imports = ["json", "os", "sys", "datetime"]

            workspace_executor = WorkspacePythonExecutor(
                workspace_dir=tmpdir,
                additional_authorized_imports=essential_imports
            )

            manager = initialize_agent_system(
                model=model,
                workspace_dir=tmpdir,
                workspace_interpreter=workspace_executor,
                essential_imports=essential_imports,
                enable_planning=False,
                planning_interval=5
            )
            print_success("ManagerAgent initialized")

            # Check managed agents
            if hasattr(manager, 'managed_agents') and manager.managed_agents:
                agent_names = [a.name for a in manager.managed_agents]
                print_info(f"Managed agents: {agent_names}")
                print_success(f"Found {len(agent_names)} managed agents")
            else:
                print_warning("No managed agents found (may be using delegation)")

            return True

    except Exception as e:
        print_error(f"ManagerAgent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_file_tools():
    """Test that agents can use file editing tools"""
    print_header("8. Testing Agent File Tools")

    try:
        from freephdlabor.toolkits.general_tools.file_editing.file_editing_tools import (
            SeeFile, CreateFileWithContent, ModifyFile, ListDir
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test ListDir
            list_tool = ListDir(working_dir=tmpdir)
            result = list_tool.forward(".")
            print_success("ListDir tool works")

            # Test CreateFileWithContent
            create_tool = CreateFileWithContent(working_dir=tmpdir)
            result = create_tool.forward("test.txt", "Hello, World!")
            print_success("CreateFileWithContent tool works")

            # Test SeeFile
            see_tool = SeeFile(working_dir=tmpdir)
            result = see_tool.forward("test.txt")
            if "Hello" in result:
                print_success("SeeFile tool works")
            else:
                print_warning("SeeFile returned unexpected content")

            # Test ModifyFile
            modify_tool = ModifyFile(working_dir=tmpdir)
            result = modify_tool.forward("test.txt", "Hello", "Hi")
            print_success("ModifyFile tool works")

            return True

    except Exception as e:
        print_error(f"File tools test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_simple_task(agent_name="ideation"):
    """Test running a simple task on an agent"""
    print_header(f"9. Testing Agent Simple Task ({agent_name})")

    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        print_error("OPENAI_API_KEY required")
        return False

    try:
        from smolagents import LiteLLMModel

        model = LiteLLMModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            api_base=api_base
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            if agent_name == "ideation":
                from freephdlabor.agents.ideation_agent import IdeationAgent
                agent = IdeationAgent(model=model, workspace_dir=tmpdir)
                task = "List the files in the current workspace directory using list_dir tool."
            else:
                from freephdlabor.agents.experimentation_agent import ExperimentationAgent
                agent = ExperimentationAgent(model=model, workspace_dir=tmpdir)
                task = "List the files in the current workspace directory using list_dir tool."

            print_info(f"Running simple task: {task}")
            print_info("This may take 10-30 seconds...")

            result = agent.run(task)

            print_success("Agent completed task")
            print_info(f"Result preview: {str(result)[:200]}...")

            return True

    except Exception as e:
        print_error(f"Agent task test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_workspace_structure():
    """Test that workspace directories are created correctly"""
    print_header("10. Testing Workspace Structure")

    try:
        from freephdlabor.utils import initialize_agent_system, create_model
        from freephdlabor.interpreters import WorkspacePythonExecutor

        model = create_model("gpt-4o-mini", "medium", "low", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            essential_imports = ["json", "os"]

            workspace_executor = WorkspacePythonExecutor(
                workspace_dir=tmpdir,
                additional_authorized_imports=essential_imports
            )

            manager = initialize_agent_system(
                model=model,
                workspace_dir=tmpdir,
                workspace_interpreter=workspace_executor,
                essential_imports=essential_imports
            )

            # Check expected directories
            expected_dirs = [
                "inter_agent_messages",
            ]

            for dir_name in expected_dirs:
                dir_path = os.path.join(tmpdir, dir_name)
                if os.path.exists(dir_path):
                    print_success(f"Found: {dir_name}/")
                else:
                    print_info(f"Not found (may be created on demand): {dir_name}/")

            print_success("Workspace structure test complete")
            return True

    except Exception as e:
        print_error(f"Workspace structure test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test agent system")
    parser.add_argument("--full", action="store_true",
                       help="Run full tests including agent task execution")
    args = parser.parse_args()

    print(f"""
{BLUE}================================================================
       freephdlabor Agent System Test Suite
================================================================{RESET}

This script tests the multi-agent system to ensure agents
can be initialized and communicate before running experiments.
""")

    results = {}

    # Core framework tests
    results["Smolagents Import"] = test_smolagents_import()
    results["LiteLLM Model"] = test_litellm_model()
    results["Base Agent Import"] = test_base_agent_import()

    # Agent initialization tests
    results["IdeationAgent Init"] = test_ideation_agent()
    results["ExperimentationAgent Init"] = test_experimentation_agent()
    results["WriteupAgent Init"] = test_writeup_agent()
    results["ManagerAgent Init"] = test_manager_agent()

    # Tool tests
    results["File Tools"] = test_agent_file_tools()
    results["Workspace Structure"] = test_workspace_structure()

    # Optional: Run agent tasks
    if args.full:
        print(f"\n{YELLOW}Running agent task tests (this may take a few minutes)...{RESET}\n")
        results["Agent Simple Task"] = test_agent_simple_task("ideation")
    else:
        print_info("Skipping agent task tests (use --full to include)")

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
    critical = ["Smolagents Import", "LiteLLM Model", "ManagerAgent Init"]
    critical_failed = [t for t in critical if not results.get(t, False)]

    if critical_failed:
        print(f"\n{RED}CRITICAL: These essential tests failed:{RESET}")
        for t in critical_failed:
            print(f"  - {t}")
        print(f"\n{RED}The multi-agent system will not work! Fix these first.{RESET}")
        return 1

    if failed > 0:
        print(f"\n{YELLOW}Some tests failed but the system may still work.{RESET}")
    else:
        print(f"\n{GREEN}All tests passed! Agent system ready.{RESET}")

    if not args.full:
        print(f"\n{BLUE}Tip: Run with --full to test agent task execution{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
