#!/usr/bin/env python3
"""
Test script to verify multi-GPU memory management in ContextLM2.
Run this to check device allocation and memory usage.
"""

import torch
from scholarlm.contextlm2 import ContextLM2

def print_gpu_memory():
    """Print current GPU memory usage for all available GPUs."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print("  No GPUs available")

def main():
    print("=" * 70)
    print("Multi-GPU Memory Management Test")
    print("=" * 70)
    
    # Check available GPUs
    print(f"\nAvailable GPUs: {torch.cuda.device_count()}")
    print_gpu_memory()
    
    # Initialize model with verbose output
    print("\n" + "=" * 70)
    print("Initializing ContextLM2...")
    print("=" * 70)
    
    # Use a small model for testing - replace with your model
    model = ContextLM2(
        model_name="your-model-name-here",  # Replace with actual model
        verbose=True,
        sampling_params={'max_new_tokens': 10}
    )
    
    print(f"\nLLM Device: {model.llm_device}")
    print(f"Tensor Device: {model.tensor_device}")
    
    print("\nMemory after initialization:")
    print_gpu_memory()
    
    # Test generation
    print("\n" + "=" * 70)
    print("Testing generation...")
    print("=" * 70)
    
    result = model.generate(
        instructions="Answer the following question based on the context.",
        context="The capital of France is Paris. It is a beautiful city.",
        query="What is the capital of France?"
    )
    
    print(f"\nGenerated response: {result['response']}")
    
    print("\nMemory after generation:")
    print_gpu_memory()
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
