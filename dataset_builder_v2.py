"""
Dataset Builder v2: Hard-Category Algorithmic & Recursive Expansion
Targeting failure modes identified in HumanEval:
- Recursive state machines & nested parenthesis tracking (e.g. parse_nested_parens, separate_paren_groups)
- Dynamic programming, modular arithmetic & number theory (e.g. modp, prime_fib)
- List partitioning, selective sorting & median calculations (e.g. intersperse, sort_third, median)
- String parsing, dictionary mapping & stateful token dispatch
"""

import json
import random
from pathlib import Path
from config import DATA_DIR

PROMPTS_V2_FILE = DATA_DIR / "curated_prompts_v2.jsonl"

HARD_ALGORITHMIC_PATTERNS = [
    # 1. Parenthesis Depth & State Machines
    ("Python", "implement separate_paren_groups(paren_string: str) -> List[str] that splits nested paren strings into separate groups balancing '(' and ')' with an explicit stack state machine"),
    ("Python", "implement parse_nested_parens(paren_string: str) -> List[int] that calculates the maximum nesting depth for each group separated by whitespace"),
    ("Python", "write an iterative and recursive bracket validator supporting '()', '[]', '{}' with exact position error reporting"),
    ("Python", "build a deterministic finite automaton (DFA) state machine to tokenize and parse arithmetic expressions with operator precedence"),
    
    # 2. Selective Sorting, Partitioning & List Slicing
    ("Python", "implement sort_third(l: list) -> list that returns a list identical to l except elements at indices divisible by 3 are sorted in ascending order"),
    ("Python", "implement intersperse(numbers: List[int], delimeter: int) -> List[int] that inserts delimeter between consecutive elements"),
    ("Python", "implement median(l: list) -> float that correctly handles both even and odd length lists using quickselect partitioning in O(n) expected time"),
    ("Python", "implement sort_numbers(numbers: str) -> str that takes a space-delimited string of number words ('zero' through 'nine') and returns them sorted"),
    ("Python", "write a function to partition an array into three parts around two pivots with zero memory allocation"),

    # 3. Modular Arithmetic, Exponentiation & Number Theory
    ("Python", "implement modp(n: int, p: int) -> int that computes (2^n) % p efficiently in O(log n) time using binary modular exponentiation without integer overflow"),
    ("Python", "implement prime_fib(n: int) -> int that returns the n-th Fibonacci number that is also prime, with Miller-Rabin primality testing"),
    ("Python", "write an extended Euclidean algorithm that computes gcd(a, b) and Bézout coefficients (x, y) satisfying a*x + b*y = gcd(a, b)"),
    ("Python", "implement Pollard's rho integer factorization algorithm with cycle detection"),
    
    # 4. Dynamic Programming & Recursion with Memoization
    ("Python", "implement min_coins_change(coins: List[int], amount: int) -> int using bottom-up DP and return the optimal coin multiset"),
    ("Python", "implement longest_palindromic_subsequence(s: str) -> str with 2D DP matrix reconstruction"),
    ("Python", "write an optimal binary search tree builder using dynamic programming in O(n^3) time"),
    ("Python", "implement wildcard string matching ('?' and '*') using memoized recursive descent"),
    
    # 5. Tree, Graph & Disjoint-Set Algorithms
    ("Python", "implement a Union-Find (Disjoint Set Union) data structure with path compression and union-by-rank"),
    ("Python", "implement Tarjan's algorithm for strongly connected components in a directed graph with recursive DFS and low-link values"),
    ("Python", "implement an interval tree supporting point query and range stabbing queries in O(log n + k)"),
    ("Python", "implement A* pathfinding on a 2D grid with tie-breaking octile distance heuristic"),

    # 6. Systems, Concurrency & Low-Level Memory
    ("C++", "write a lock-free multi-producer single-consumer (MPSC) bounded queue using atomic acquire/release semantics"),
    ("Rust", "implement a custom intrusive doubly-linked list with Pin and non-null pointers"),
    ("Rust", "write a zero-copy SIMD parser for JSON strings utilizing AVX-512 bit manipulation instructions"),
    ("Python", "implement a custom memory-mapped cache with LRU eviction and segment hashing")
]

TOOL_LOGIC_PATTERNS = [
    "You are an autonomous coding agent. Parse the following stack trace: 'RecursionError: maximum recursion depth exceeded in comparison' in modp(n, p). Diagnose and refactor to iterative binary exponentiation.",
    "Given a JSON schema with nested oneOf and allOf conditional dependencies, synthesize a recursive Pydantic v2 validation model.",
    "Construct a topological sort algorithm for asynchronous DAG workflow execution handling dynamic branch failures.",
    "Evaluate AST node transformations for dead-code elimination in Python bytecode using the ast and dis modules.",
    "Design an atomic two-phase commit protocol coordinator with persistent write-ahead logging and crash-recovery simulation."
]

GENERAL_REASONING_PATTERNS = [
    "Derive the time and memory bandwidth complexity of FlashAttention-2 vs Standard Multi-Head Self-Attention during prefill.",
    "Explain the mathematical relationship between router top-k entropy in MoE models and downstream representation collapse.",
    "Analyze how speculative decoding with draft acceptance rate alpha affects effective decoding latency as a function of verification overhead gamma.",
    "Explain the exact numerical stability properties of LayerNorm vs RMSNorm under float16 and bfloat16 quantization."
]

def build_v2_dataset(num_samples=2500):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(1337)
    
    samples = []
    
    # Category 1: 50% Hard Algorithmic & Recursive (1250 samples)
    target_hard = int(num_samples * 0.50)
    for i in range(target_hard):
        lang, task = random.choice(HARD_ALGORITHMIC_PATTERNS)
        prompt = (
            f"Prompt #{i+1:04d} [Algorithmic]: In {lang}, {task}. "
            f"Provide an optimal implementation with detailed edge case handling (empty inputs, boundaries, large values), "
            f"exact type hints, and complete test assertions."
        )
        samples.append({
            "id": f"hard_algo_{i:04d}",
            "category": "hard_algorithmic",
            "lang": lang,
            "text": prompt
        })
        
    # Category 2: 30% Logic & Autonomous Tool Execution (750 samples)
    target_tool = int(num_samples * 0.30)
    for i in range(target_tool):
        topic = random.choice(TOOL_LOGIC_PATTERNS)
        prompt = f"Prompt #{i+1:04d} [Logic/Tools]: {topic} Provide formal rationale and verify step-by-step."
        samples.append({
            "id": f"tool_logic_{i:04d}",
            "category": "tool_logic",
            "lang": "Python",
            "text": prompt
        })
        
    # Category 3: 20% Architecture & Theoretical Foundations (500 samples)
    target_theory = num_samples - len(samples)
    for i in range(target_theory):
        topic = random.choice(GENERAL_REASONING_PATTERNS)
        prompt = f"Prompt #{i+1:04d} [Theory]: {topic} Provide rigorous mathematical formulas and derivation."
        samples.append({
            "id": f"theory_{i:04d}",
            "category": "theory",
            "lang": "Text",
            "text": prompt
        })
        
    random.shuffle(samples)
    
    with open(PROMPTS_V2_FILE, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"[OK] Successfully built {len(samples)} curated prompts in: {PROMPTS_V2_FILE}")
    print(f"     Hard Algorithmic (50%): {target_hard}")
    print(f"     Tool/Logic (30%):        {target_tool}")
    print(f"     Theory/Reasoning (20%):  {target_theory}")

if __name__ == "__main__":
    build_v2_dataset()
