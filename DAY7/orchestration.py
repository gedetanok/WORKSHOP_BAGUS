from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from .agents import (
    PlannerAgent, CoderAgent, TesterAgent, DebuggerAgent, DocumentationAgent, CodeReviewerAgent, DependencyManagerAgent
)

from DAY7.tools import ask_approval, ask_review

class CodingState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    plan: str
    code_generated: bool
    tests_passed: bool
    needs_debugging: bool
    current_step: str
    files_created: List[str]
    next_action: str
    debug_attempts: int  # Track number of debug attempts

class CodingOrchestrator:
    def __init__(self):
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.tester = TesterAgent()
        self.debugger = DebuggerAgent()
        self.doc_agent = DocumentationAgent()
        self.reviewer = CodeReviewerAgent()
        self.dep_manager = DependencyManagerAgent()
        self.workflow = self._build_workflow()

    
    def _planner_node(self, state: CodingState) -> CodingState:
        """Node untuk planning"""

        print("PLANNER AGENT: Creating execution plan...")
        plan = self.planner.run(state['task'])

        print("\nPLAN GENERATED:")
        print(plan)

        approved = ask_approval(
            "Proceed with this plan?",
            "The planner has created an execution plan above."
        )

        next_action = "approved" if approved else "rejected"
        
        if not approved:
            print("\n❌ Plan rejected by user. Stopping workflow.")
        
        return {
            "messages": [AIMessage(content=f"Plan: {plan}")],
            "plan": plan,
            "next_action": next_action,
            "current_step": "planning_complete"
        }
    
    def _dependency_manager_node(self, state: CodingState) -> CodingState:
        """Node untuk install dependencies."""
        print("DEPENDENCY MANAGER: Installing dependencies...")
        
        instruction = f"""Based on this plan:
        {state['plan']}

        Task: {state['task']}

        Install all necessary dependencies. Ask for approval before installing."""
        
        # Run dependency manager
        result = self.dep_manager.run(instruction)
        
        print(f"\n{result}")
        
        return {
            "messages": [AIMessage(content=f"Dependencies: {result}")],
            "current_step": "dependencies_installed"
        }
    
    
    def _coder_node(self, state: CodingState) -> CodingState:
        """Node untuk generate code"""
        print("\n" + "="*60)
        print("CODER AGENT: Generating code...")
        print("="*60)

        instruction = f"""Task: {state['task']}

Plan to follow:
{state['plan']}

Generate the code files as specified in the plan.
Use the write_file tool to create each file.
Write clean, well-documented code.

IMPORTANT: After creating each file, list all the files you created with their full paths.
At the end of your response, include a section like:

FILES CREATED:
- path/to/file1.py
- path/to/file2.py
- path/to/file3.py"""

        # Get code from coder agent
        result = self.coder.run(instruction)

        print(f"\n{result}")

        # Extract created files from result
        files_created = []
        if "FILES CREATED:" in result or "File '" in result:
            # Parse files from response
            import re
            # Match patterns like "File 'filename.py'" or "- filename.py"
            patterns = [
                r"File ['\"]([^'\"]+)['\"]",  # Matches: File 'filename.py'
                r"- ([^\n]+\.py)",  # Matches: - filename.py
                r"created ([^\n]+\.py)",  # Matches: created filename.py
                r"wrote to ([^\n]+\.py)",  # Matches: wrote to filename.py
            ]

            for pattern in patterns:
                matches = re.findall(pattern, result, re.IGNORECASE)
                files_created.extend(matches)

            # Remove duplicates and strip whitespace
            files_created = list(set([f.strip() for f in files_created]))

        if not files_created:
            print("\n⚠️  Warning: Could not detect created files from response. Using fallback.")
            files_created = ["code_file.py"]

        print(f"\n📁 Files tracked: {files_created}")

        return {
            "messages": [AIMessage(content=f"Code: {result}")],
            "code_generated": True,
            "files_created": state.get("files_created", []) + files_created,
            "current_step": "code_generated"
        }
    
    def _reviewer_node(self, state: CodingState) -> CodingState:
        """Node untuk code review"""
        print("\n" + "="*60)
        print("REVIEWER AGENT: Reviewing code...")
        print("="*60)

        files_list = state.get('files_created', [])

        if not files_list:
            print("⚠️  No files tracked. Attempting to find created files...")
            # Fallback: ask reviewer to list files first
            files_list = ["(files not tracked - reviewer will search)"]

        instruction = f"""Review the code that was just generated for:
Task: {state['task']}

Files that should have been created: {files_list}

Your tasks:
1. Use the read_file tool to read each file that was created
2. If you can't find the files from the list, use list_files tool to see what files exist in the current directory
3. Provide feedback on:
   - Code quality
   - Best practices
   - Potential issues
   - Suggestions for improvement

Make sure to actually READ the files before reviewing!"""

        review = self.reviewer.run(instruction)

        print("\nCODE REVIEW:")
        print("-"*60)
        print(review)
        print("-"*60)

        # Human-in-the-Loop: Ask if user wants to proceed or revise
        print("\nOptions:")
        print("1. Approve and continue to testing")
        print("2. Request revision")
        print("3. Skip tests and go to documentation")

        choice = input("\nYour choice (1/2/3): ").strip()

        next_action = "approved"
        if choice == "2":
            next_action = "needs_revision"
            print("\nRequesting code revision...")
        elif choice == "3":
            next_action = "skip_tests"
            print("\nSkipping tests...")
        else:
            print("\nProceeding to tests...")

        return {
            "messages": [AIMessage(content=f"Review: {review}")],
            "next_action": next_action,
            "current_step": "code_reviewed"
        }


    
    def _tester_node(self, state: CodingState) -> CodingState:
        """Node untuk testing code"""
        print("\n" + "="*60)
        print("🧪 TESTER AGENT: Running tests...")
        print("="*60)

        files_list = state.get('files_created', [])

        instruction = f"""Create and run tests for:
Task: {state['task']}
Files to test: {files_list}

IMPORTANT:
1. First, use read_file tool to read the actual code files: {files_list}
2. Understand what the code does
3. Create appropriate test files (name them test_*.py)
4. Write comprehensive tests for the actual functionality
5. Run the tests using execute_command tool with 'pytest -v'
6. Report the results clearly

If files list is empty or unclear, use list_files tool first to see what files exist."""

        test_result = self.tester.run(instruction)

        print(f"\n{test_result}")

        # Simple heuristic untuk determine if tests passed
        # In production, parse actual test output
        tests_passed = "passed" in test_result.lower() and "failed" not in test_result.lower()

        # Also check for "0 failed" or similar success indicators
        if "0 failed" in test_result.lower() or "success" in test_result.lower():
            tests_passed = True

        next_action = "passed" if tests_passed else "failed"

        if tests_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed. Will proceed to debugging...")

        # Reset debug attempts if tests passed (for fresh start next time)
        debug_attempts_reset = 0 if tests_passed else state.get('debug_attempts', 0)

        return {
            "messages": [AIMessage(content=f"Tests: {test_result}")],
            "tests_passed": tests_passed,
            "next_action": next_action,
            "current_step": "tests_completed",
            "debug_attempts": debug_attempts_reset
        }
    
    def _debugger_node(self, state: CodingState) -> CodingState:
        """Node untuk debugging jika tests fail"""
        print("\n" + "="*60)
        print("DEBUGGER AGENT: Fixing bugs...")
        print("="*60)

        # Track debug attempts
        debug_attempts = state.get('debug_attempts', 0) + 1
        max_attempts = 3

        print(f"🔄 Debug attempt {debug_attempts} of {max_attempts}")

        files_list = state.get('files_created', [])
        previous_messages = state.get('messages', [])

        # Get test results from previous messages
        test_output = ""
        for msg in reversed(previous_messages):
            if "Tests:" in msg.content:
                test_output = msg.content
                break

        instruction = f"""Debug and fix the failing tests for:
Task: {state['task']}
Files involved: {files_list}

Previous test output:
{test_output}

Your tasks:
1. Use read_file to read the source files: {files_list}
2. Use read_file to read the test files (usually test_*.py)
3. Analyze the error messages from the test output above
4. Identify the root cause
5. Use write_file to fix the bugs in the source files
6. Explain what was fixed and why

Be specific about which files you're reading and modifying."""

        debug_result = self.debugger.run(instruction)

        print(f"\n{debug_result}")

        # Human-in-the-Loop: Ask what to do next
        print("\n" + "="*60)
        print("DEBUG COMPLETED")
        print("="*60)
        print(f"\n⚠️  Debug attempts so far: {debug_attempts}/{max_attempts}")
        print("\nWhat would you like to do?")
        print("1. Re-run tests to verify fixes")
        print("2. Skip to documentation (accept current state)")
        print("3. Stop workflow (end here)")

        next_action = "retry"  # default

        if debug_attempts >= max_attempts:
            print(f"\n⚠️  Maximum debug attempts ({max_attempts}) reached!")
            print("You can either:")
            print("- Continue to documentation (recommended)")
            print("- Force one more retry")

            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == "2":
                next_action = "skip_to_docs"
                print("\n➡️  Skipping to documentation...")
            elif choice == "3":
                next_action = "stop"
                print("\n🛑 Stopping workflow...")
            else:
                print("\n🔄 Forcing one more retry...")
                next_action = "retry"
        else:
            choice = input("\nYour choice (1/2/3): ").strip()

            if choice == "2":
                next_action = "skip_to_docs"
                print("\n➡️  Skipping to documentation...")
            elif choice == "3":
                next_action = "stop"
                print("\n🛑 Stopping workflow...")
            else:
                print("\n🔄 Re-running tests...")
                next_action = "retry"

        return {
            "messages": [AIMessage(content=f"Debug: {debug_result}")],
            "current_step": "bugs_fixed",
            "debug_attempts": debug_attempts,
            "next_action": next_action
        }
    
    def _documentation_node(self, state: CodingState) -> CodingState:
        """Node untuk create documentation"""
        print("\n" + "="*60)
        print("📚 DOCUMENTATION AGENT: Creating documentation...")
        print("="*60)

        files_list = state.get('files_created', [])

        instruction = f"""Create comprehensive documentation for:
Task: {state['task']}
Files created: {files_list}

Your tasks:
1. Use read_file to read each source file: {files_list}
2. Understand what each file does
3. Use write_file to create a README.md with:
   - Project overview
   - Installation instructions (including dependencies)
   - Usage examples with code
   - File structure explanation
   - API documentation (if applicable)

Make sure to include actual code examples from the files you read!"""

        doc_result = self.doc_agent.run(instruction)

        print(f"\n{doc_result}")
        print("\n✅ Documentation created!")

        return {
            "messages": [AIMessage(content=f"Docs: {doc_result}")],
            "current_step": "documentation_complete"
        }


    # conditinoal functions
    def _should_proceed_from_plan(self, state: CodingState) -> Literal['approved', 'rejected']:
        return state.get('next_action', 'rejected')

    def _should_proceed_from_review(self, state: CodingState) -> Literal["approved", "needs_revision", "skip_tests"]:
        """Decide next step after code review"""
        return state.get("next_action", "approved")

    def _should_proceed_from_tests(self, state: CodingState) -> Literal["passed", "failed"]:
        """Decide if proceed to docs or debugging based on test results"""
        return state.get("next_action", "failed")

    def _should_proceed_from_debugger(self, state: CodingState) -> Literal["retry", "skip_to_docs", "stop"]:
        """Decide next step after debugging"""
        return state.get("next_action", "retry")
    

    # construct workflow
    def _build_workflow(self) -> StateGraph:
        wf = StateGraph(CodingState)

        wf.add_node("planner", self._planner_node)
        wf.add_node("coder", self._coder_node)
        wf.add_node("reviewer", self._reviewer_node)
        wf.add_node("tester", self._tester_node)
        wf.add_node("debugger", self._debugger_node)
        wf.add_node("dependency_manager", self._dependency_manager_node)
        wf.add_node("documentation", self._documentation_node)

        wf.add_edge(START, 'planner')

        wf.add_conditional_edges(
            "planner",
            self._should_proceed_from_plan,
            {
                "approved": "dependency_manager",
                "rejected": END
            }
        )

        wf.add_edge("dependency_manager", "coder")
        wf.add_edge("coder", "reviewer")

        wf.add_conditional_edges(
            'reviewer',
            self._should_proceed_from_review,
            {
                "approved": "tester",
                "needs_revision": "coder",
                "skip_tests": "documentation"
            }
        )

        wf.add_conditional_edges(
            "tester",
            self._should_proceed_from_tests,
            {
                "passed": "documentation",
                "failed": "debugger"
            }
        )

        # Change debugger to use conditional edges instead of unconditional
        wf.add_conditional_edges(
            "debugger",
            self._should_proceed_from_debugger,
            {
                "retry": "tester",  # Try again
                "skip_to_docs": "documentation",  # Skip to docs
                "stop": END  # Stop workflow
            }
        )

        wf.add_edge("documentation", END)

        memory = MemorySaver()
        return wf.compile(checkpointer=memory)
    

    def run(self, task: str, thread_id: str = 'default'):
        print("\n" + "AI CODING AGENT".center(60, "="))
        print(f"\n📝 TASK: {task}\n")
        
        initial_state = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "plan": "",
            "code_generated": False,
            "tests_passed": False,
            "needs_debugging": False,
            "current_step": "start",
            "files_created": [],
            "next_action": "",
            "debug_attempts": 0
        }


        # run workflow
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = self.workflow.invoke(initial_state, config)
            
            print("\n" + "="*60)
            print("✅WORKFLOW COMPLETED!")
            print("="*60)
            
            print(f"\nSummary:")
            print(f"  • Files created: {final_state.get('files_created', [])}")
            print(f"  • Tests passed: {final_state.get('tests_passed', False)}")
            print(f"  • Final step: {final_state.get('current_step', 'unknown')}")
            
            return final_state
            
        except Exception as e:
            print(f"\n❌ ERROR in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        

def main():
    print("Welcome to AI Coding Agent")

    orchestrator = CodingOrchestrator()

    task = input("\nWhat would you like me to build? (or 'exit' to quit)\n\n> ").strip()

    if not task or task.lower() == 'exit':
        print("\n👋 Goodbye!\n")
        return
    
    orchestrator.run(task, thread_id="session-1")

if __name__ == "__main__":
    main()