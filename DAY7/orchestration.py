from dotenv import load_dotenv
load_dotenv()

from .agents import (
    PlannerAgent, CodeReviewerAgent, CoderAgent, TesterAgent, DebuggerAgent, DocumentationAgent, DependencyManagerAgent, get_agent
)

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from .tools import ask_approval


class CodingState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: str
    plan: str
    code_generated: bool
    code_result: str
    tests_passed: bool
    needs_debugging: bool
    current_step: str
    file_created: List[str]
    next_action: str


class CodingOrchestration:
    def __init__(self):
        self.planner = PlannerAgent()
        self.reviewer = CodeReviewerAgent()
        self.coder = CoderAgent()
        self.tester = TesterAgent()
        self.debugger = DebuggerAgent()
        self.doc_agent = DocumentationAgent()
        self.dep_manager = DependencyManagerAgent()


    def _planner_node(self, state: CodingState) -> CodingState:
        """Node untuk planning"""

        print('PLANNER AGENT: Beraksi...')
        plan = self.planner.run(state['task'])

        print("PLAN:")
        print(plan)

        approved = ask_approval(
            'Proses ga nih?',
            'Planner agent udah bikin plan untuk dieksekusi'
        )

        if not approved:
            print('User tidak menyetujui hal tersebut')

        next_action = 'approved' if approved else 'rejected'

        return {
            'messages': [AIMessage(f'Plan: {plan}')],
            'plan': plan,
            'next_action':next_action,
            'current_step':'planning_complete'
        }
    
    def _dependency_manager_node(self, state: CodingState) -> CodingState:
        """Node untuk inatall dependencies."""

        print('DEPENDENCY MANAGER AGENT: Beraksi...')
        intruksi = f"""Berdasarkan plan ini:
        {state['plan']}

        task: {state['task']}
        Install semua dependency yang diperlukan, dan penting.
        """

        result = self.dep_manager.run(intruksi)

        print(result)

        return {
            'messages': [AIMessage(f'{result}')],
            'current_step':'dependencies_installed'
        }
    
    def _coder_node(self, state: CodingState) -> CodingState:
        """Node untuk membuat code dalam file."""

        print('CODER AGENT: Beraksi...')
        intruksi = f"""Task: {state['task']}

        Plan yang perlu diikuti: 
        {state['plan']}
        Buatkan sebuah file yang berisi code sesuai dengan yang dispesifikkan pada plan.
        Gunakan write_file tool untuk membuat setiap file.
        Buat code yang bersih, dan terdokumentasi dengan baik.
        """

        result = self.coder.run(intruksi)

        print(result)

        file_created = []
        if 'created' in result.lower() or 'wrote' in result.lower():
            file_created.append('copy_code.py')

        return {
            'messages': [AIMessage(f'Code: {result}')],
            'current_step':'code_generated',
            'code_generated':True,
            'file_created': state.get('file_created', []),
            'code_result':result
        }
    

    def _reviewer_node(self, state: CodingState) -> CodingState:
        """Node review hasil code yang sudah digenerate."""

        print('REVIEWER AGENT: Beraksi...')
        intruksi = f"""Review code yang sudah dibuat pada node sebelumnya ini:
        {state['code_result']}

        Dengan tugas: {state['task']}

        Berikan feedback dengan memperhatikan beberapa hal ini:
        1. Code quality
        2. Best practice
        3. Potential issues
        4. Suggestion for improvement
        """

        result = self.reviewer.run(intruksi)

        print('\nPilihan:')
        print('1. Approve and contionue to testing')
        print('2. Request revision')
        print('3. Skip tests')

        option = input('Pilihanmu(1/2/3): ').strip()
        
        next_action = 'approved'
        if option == '2':
            next_action = 'need_revision'
        elif option == '3':
            next_action = 'skip_tests'
        else:
            print('Proceding to test...')

        return {
            'messages': [AIMessage(f'Review result: {result}')],
            'current_step':'code_reviewed',
            'next_action': next_action
        }
    

    def _tester_node(self, state: CodingState) -> CodingState:
        """Node untuk test hasil code yang sudah direview."""

        print('TESTER AGENT: Beraksi...')

        intruksi = f"""Buat dan jalankan sebuah test untuk:
        Task: {state['task']}
        Code Result: {state['code_result']}
        """

        result = self.tester.run(intruksi)

        tests_passed = 'passed' in result.lower() or 'success' in result.lower()

        next_action = 'passed' if tests_passed else 'failed'

        return {
            'messages': [AIMessage(f'Test result: {result}')],
            'current_step':'test_completed',
            'test_passed': tests_passed,
            'next_action': next_action
        }
    

    def _debugger_node(self, state: CodingState) -> CodingState:
        """Node debug error pada code."""

        print('DEBUGGER AGENT: Beraksi...')

        intruksi = f"""Debug and fix code dari hasil testing:
        Task: {state['task']}
        Code: {state['code_result']}
        
        Test sebelumnya menunjukkan error, bantu untuk:
        1. Analisa bagian mana yang kurang tepat
        2. Idetifikasi akar penyebab masalah
        3. Perbaiki errornya
        4. Jelaskan apa yang sudah di fix
        """

        result = self.debugger.run(intruksi)
        
        print('Re-running the tests after fixed')

        return {
            'messages': [AIMessage(f'Debug Result: {result}')],
            'current_step':'bug_fixed',
            'code_result': result
        }
        
        
    def _documentation_node(self, state: CodingState) -> CodingState:
        """Node debug error pada code."""

        print('DOCUMENTATION AGENT: Beraksi...')

        intruksi = f"""Buatkan dokumentasi dari apa yang sudah dibuat sebelumya:
        Task: {state['task']}
        Code: {state['code_result']}
        
        Test sebelumnya menunjukkan error, bantu untuk:
        1. Analisa bagian mana yang kurang tepat
        2. Idetifikasi akar penyebab masalah
        3. Perbaiki errornya
        4. Jelaskan apa yang sudah di fix
        """

        result = self.doc_agent.run(intruksi)
        
        print('Re-running the tests after fixed')

        return {
            'messages': [AIMessage(f'Review_result: {result}')],
            'current_step':'documentation_completed',
        }
    
    
    
    
    

        





        

        

