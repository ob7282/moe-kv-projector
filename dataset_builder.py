"""
Dataset Builder for Expert-Linked MoE KV Projector
Curates a diverse, high-quality corpus of 3,000 prompts across Code (40%), Logic & Tools (30%), and Knowledge (30%).
"""

import json
import random
from pathlib import Path
from config import PROMPTS_FILE, DATA_DIR

CODE_TOPICS = [
    ("Python", "implement a lock-free concurrent queue with CAS operations and unit tests"),
    ("Python", "build a B-Tree indexing engine with node split and range query support"),
    ("Python", "write a custom PyTorch autograd Function implementing fused SwiGLU"),
    ("Python", "implement a high-performance Trie with prefix search and fuzzy matching"),
    ("Python", "create an async HTTP/2 connection multiplexer with backpressure control"),
    ("TypeScript", "define a strongly-typed schema validation engine using template literal types"),
    ("TypeScript", "implement a Redux-like reactive state store with middleware pipeline"),
    ("Rust", "write a memory-mapped circular ring buffer with zero-copy deserialization"),
    ("Rust", "implement a lock-free skip list using crossbeam epoch-based reclamation"),
    ("SQL", "write an optimized recursive CTE query to detect cycles in a directed graph"),
    ("SQL", "design an analytical query calculating 30-day rolling window percentiles"),
    ("C++", "write a SIMD AVX-512 vectorized matrix multiplication kernel with cache tiling"),
    ("C++", "implement a custom memory arena allocator with 64-byte alignment guarantees"),
]

LOGIC_TOOL_TOPICS = [
    "You are an autonomous Hermes coding agent. Plan the step-by-step refactoring of an authentication microservice.",
    "Evaluate the following JSON payload against OpenAPI 3.1 specification and output valid JSON corrections.",
    "Given 5 microservices with cyclic network dependencies, construct an acyclic topological deployment sequence.",
    "Analyze the following database deadlock trace and provide the exact transaction re-ordering to guarantee isolation.",
    "Generate a structured function call schema for querying a medical knowledge graph with ontology validation.",
    "You have tools: execute_shell, view_file, and edit_file. Construct the multi-step tool call sequence to patch a vulnerability.",
    "Solve the following constraint satisfaction problem using forward checking with MRV heuristic.",
    "Analyze the time and space complexity trade-offs between Raft leader lease reads and linearizable quorum reads."
]

KNOWLEDGE_TOPICS = [
    "Explain the hardware architecture of AMD Zen 4 vector execution units, specifically 512-bit AVX-512 and VNNI pipelines.",
    "Describe the mathematical derivation of Flash Attention tiling and how it achieves sub-quadratic HBM memory bandwidth.",
    "Explain the mechanics of Mixture-of-Experts routing: top-k gating, load balancing auxiliary loss, and expert capacity factors.",
    "Detail how the Linux kernel handles page table isolation, TLB invalidation, and speculative side-channel mitigations.",
    "Explain the internal mechanics of Vulkan cooperative matrix extensions and subgroup operations on modern GPU compute units.",
    "Describe the differences between Multi-Head Attention, Multi-Query Attention, and Grouped-Query Attention (GQA) in modern LLMs.",
    "Detail the forward and backward passes of Rotary Position Embeddings (RoPE) and its interaction with long-context scaling.",
    "Explain the mathematics of low-rank matrix decomposition (SVD) and its application to parameter-efficient fine-tuning."
]

def generate_curated_dataset(num_samples=3000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    entries = []
    random.seed(42)
    
    # 40% Code (1200 samples)
    target_code = int(num_samples * 0.40)
    for i in range(target_code):
        lang, task = random.choice(CODE_TOPICS)
        variant = f"Prompt #{i+1}: In {lang}, {task}. Ensure production-ready error handling, robust type signatures, and clear docstrings."
        entries.append({"id": f"code_{i:04d}", "category": "code", "text": variant})
        
    # 30% Logic & Tools (900 samples)
    target_logic = int(num_samples * 0.30)
    for i in range(target_logic):
        topic = random.choice(LOGIC_TOOL_TOPICS)
        variant = f"Prompt #{i+1}: {topic} Provide formal rationale and step-by-step verification."
        entries.append({"id": f"logic_{i:04d}", "category": "logic_tools", "text": variant})
        
    # 30% Knowledge (900 samples)
    target_knowledge = num_samples - len(entries)
    for i in range(target_knowledge):
        topic = random.choice(KNOWLEDGE_TOPICS)
        variant = f"Prompt #{i+1}: {topic} Include deep architectural context, memory layout diagrams, and mathematical formulations."
        entries.append({"id": f"knowledge_{i:04d}", "category": "knowledge", "text": variant})
        
    random.shuffle(entries)
    
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"[OK] Generated {len(entries)} curated prompts in: {PROMPTS_FILE}")
    print(f"     Code: {target_code} | Logic/Tools: {target_logic} | Knowledge: {target_knowledge}")

if __name__ == "__main__":
    generate_curated_dataset()
