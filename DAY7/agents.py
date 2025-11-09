
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from .tools import read_file, write_file, list_files, execute_command


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class PlannerAgent:
    """
    Agent untuk membuat rencana/planning dari user request.
    Memecah task besar menjadi langkah-langkah kecil.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [list_files]

        system_prompt = """You are an expert planning agent for coding tasks.

Your job is to:
1. Analyze the user's request
2. Break it down into clear, actionable steps
3. Identify what files need to be created/modified
4. Identify what dependencies/packages need to be installed
5. Output a structured plan

Output format MUST be:
PLAN:
1. [First step]
2. [Second step]
3. [Third step]
...

FILES TO CREATE:
- filename1.py: [purpose]
- filename2.py: [purpose]

DEPENDENCIES:
- package1
- package2

Be specific and detailed. Think step-by-step."""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, task: str) -> str:
        """Execute planning"""
        result = self.agent.invoke({"messages": [("user", task)]})
        return result["messages"][-1].content


class CoderAgent:
    """
    Agent untuk menulis code.
    Generate clean, well-documented code berdasarkan specifications.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.tools = [write_file, read_file, list_files]

        system_prompt = """You are an expert Python developer.

Your responsibilities:
1. Write clean, readable, and well-documented code
2. Follow Python best practices (PEP 8)
3. Add helpful comments and docstrings
4. Handle errors appropriately
5. Use the write_file tool to create files

Guidelines:
- Always add docstrings to functions and classes
- Use type hints where appropriate
- Add error handling with try-except
- Keep functions focused and small
- Use meaningful variable names

When you create a file, use the write_file tool with:
- filepath: the exact path/filename
- content: the complete code

Think step-by-step and write production-ready code."""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, instruction: str) -> str:
        """Execute coding task"""
        result = self.agent.invoke({"messages": [("user", instruction)]})
        return result["messages"][-1].content


class TesterAgent:
    """
    Agent untuk testing code.
    Membuat dan menjalankan tests, melaporkan hasil.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [read_file, write_file, execute_command]

        system_prompt = """You are an expert software tester.

Your responsibilities:
1. Create comprehensive unit tests
2. Test edge cases and error conditions
3. Use pytest framework
4. Run tests and report results
5. Identify bugs and issues

When creating tests:
- Use pytest conventions
- Test normal cases AND edge cases
- Add clear test function names
- Use assertions effectively
- Mock external dependencies if needed

When running tests:
- Use execute_command tool to run pytest
- Analyze the output
- Report pass/fail clearly
- Suggest fixes if tests fail"""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, instruction: str) -> str:
        """Execute testing task"""
        result = self.agent.invoke({"messages": [("user", instruction)]})
        return result["messages"][-1].content


class DebuggerAgent:
    """
    Agent untuk debugging code.
    Menganalisis error, menemukan bugs, dan memperbaiki code.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        self.tools = [read_file, write_file, execute_command]

        system_prompt = """You are an expert debugger and problem solver.

Your responsibilities:
1. Analyze error messages and stack traces
2. Identify the root cause of bugs
3. Propose and implement fixes
4. Test the fixes
5. Explain what was wrong and how you fixed it

Debugging process:
1. Read the error message carefully
2. Examine the relevant code
3. Identify the root cause
4. Implement the fix
5. Verify the fix works

Be systematic and thorough. Explain your reasoning."""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, instruction: str) -> str:
        """Execute debugging task"""
        result = self.agent.invoke({"messages": [("user", instruction)]})
        return result["messages"][-1].content


class DocumentationAgent:
    """
    Agent untuk membuat dokumentasi.
    Membuat README, docstrings, dan documentation files.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)
        self.tools = [read_file, write_file, list_files]

        system_prompt = """You are a technical documentation expert.

Your responsibilities:
1. Create clear, comprehensive documentation
2. Write README.md files
3. Document APIs and functions
4. Add usage examples
5. Keep documentation up-to-date

Documentation should include:
- Project overview and purpose
- Installation instructions
- Usage examples
- API reference (if applicable)
- Contributing guidelines (if applicable)
- License information

Write in clear, concise language. Use markdown formatting.
Add code examples where helpful."""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, instruction: str) -> str:
        """Execute documentation task"""
        result = self.agent.invoke({"messages": [("user", instruction)]})
        return result["messages"][-1].content


class CodeReviewerAgent:
    """
    Agent untuk review code.
    Memberikan feedback, suggestions, dan improvements.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
        self.tools = [read_file]

        system_prompt = """You are an expert code reviewer.

Your responsibilities:
1. Review code for quality and best practices
2. Check for potential bugs or issues
3. Suggest improvements
4. Verify code follows standards
5. Provide constructive feedback

Review criteria:
- Code clarity and readability
- Proper error handling
- Performance considerations
- Security issues
- Test coverage
- Documentation quality

Be constructive and specific in your feedback.
Highlight both good practices and areas for improvement."""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, instruction: str) -> str:
        """Execute code review"""
        result = self.agent.invoke({"messages": [("user", instruction)]})
        return result["messages"][-1].content


class DependencyManagerAgent:
    """
    Agent untuk manage dependencies.
    Install packages, create requirements.txt, dll.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [execute_command, write_file, read_file]

        system_prompt = """You are a dependency management expert.

Your responsibilities:
1. Identify required packages/dependencies
2. Install packages using pip
3. Create requirements.txt
4. Manage virtual environments
5. Handle version conflicts

When installing packages:
- Use execute_command with 'pip install package_name'
- Check if package is already installed first
- Create requirements.txt with all dependencies
- Specify versions when necessary

Be careful and verify installations succeed."""

        self.agent = create_agent(self.llm, self.tools, system_prompt=system_prompt)

    def run(self, instruction: str) -> str:
        """Execute dependency management"""
        result = self.agent.invoke({"messages": [("user", instruction)]})
        return result["messages"][-1].content


# Dictionary untuk easy access ke semua agents
AGENTS = {
    "planner": PlannerAgent,
    "coder": CoderAgent,
    "tester": TesterAgent,
    "debugger": DebuggerAgent,
    "documentation": DocumentationAgent,
    "reviewer": CodeReviewerAgent,
    "dependency_manager": DependencyManagerAgent,
}


def get_agent(agent_type: str):
    """
    Helper function untuk mendapatkan agent instance.
    
    Args:
        agent_type: Tipe agent yang diinginkan
        
    Returns:
        Agent instance
    """
    if agent_type not in AGENTS:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(AGENTS.keys())}")
    
    return AGENTS[agent_type]()