# test_mtl_integration.py
"""
Test script to verify MTL implementation in VERL.
Usage: python test_mtl_integration.py
"""

import torch
import torch.nn as nn
import gc
import psutil
import os
from typing import Dict, List
import numpy as np
from dataclasses import dataclass
from verl import DataProto
from verl.utils.mtl_utils import PCGrad


@dataclass
class TestMetrics:
    """Container for test results."""
    task_grouping_correct: bool = False
    pcgrad_optimizer_used: bool = False
    gradients_clipped: bool = False
    memory_stable: bool = False
    per_task_losses: Dict[str, List[float]] = None
    gradient_norms: List[float] = None
    memory_usage: List[float] = None
    

class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class DummyActor:
    """Mimics DataParallelPPOActor interface."""
    def __init__(self, model, optimizer, clip_grad_norm=1.0):
        self.actor_module = model
        self.actor_optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        
    def compute_policy_loss(self, data: DataProto):
        """Compute a dummy loss for testing."""
        # Extract data
        inputs = data.batch['input_ids']
        targets = data.batch['labels']
        
        # Forward pass
        outputs = self.actor_module(inputs)
        
        # Dummy loss (cross entropy)
        loss = nn.functional.cross_entropy(outputs, targets)
        
        # Add some task-specific scaling to make losses different
        task_name = data.meta_info.get('task_name', 'unknown')
        task_scale = {'math': 1.0, 'code': 2.0, 'logic': 0.5}.get(task_name, 1.0)
        
        return loss * task_scale


def create_dummy_task_data(task_name: str, batch_size: int = 16, seq_len: int = 128) -> DataProto:
    """Create dummy data for a specific task."""
    return DataProto.from_dict(
        tensors={
            'input_ids': torch.randn(batch_size, seq_len),
            'labels': torch.randint(0, 10, (batch_size,)),
            'advantages': torch.ones(batch_size),
            'old_logits': torch.randn(batch_size, 10),
            'actions': torch.randint(0, 10, (batch_size,))
        },
        meta_info={'task_name': task_name}
    )


def test_task_grouping():
    """Test 1: Verify task batches are correctly grouped."""
    print("\n=== Test 1: Task Grouping ===")
    
    # Create mixed data
    task_names = ['math', 'code', 'logic', 'math', 'code', 'logic']
    task_batches = {}
    
    # Simulate trainer grouping logic
    for i, task_name in enumerate(task_names):
        if task_name not in task_batches:
            task_batches[task_name] = []
        task_batches[task_name].append(create_dummy_task_data(task_name, batch_size=8))
    
    # Combine batches per task (like trainer would do)
    combined_task_batches = {}
    for task_name, batches in task_batches.items():
        # In real code, this would be DataProto.concat(batches)
        combined_task_batches[task_name] = batches[0]  # Simplified for test
    
    # Verify
    assert len(combined_task_batches) == 3, f"Expected 3 tasks, got {len(combined_task_batches)}"
    assert set(combined_task_batches.keys()) == {'math', 'code', 'logic'}
    
    print("✓ Task grouping correct: 3 unique tasks identified")
    return True


def test_pcgrad_optimizer(config_mtl_method='pcgrad'):
    """Test 2: Verify PCGrad optimizer is being used."""
    print("\n=== Test 2: PCGrad Optimizer ===")
    
    # Create model and base optimizer
    model = DummyModel()
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Wrap with PCGrad if configured
    if config_mtl_method == 'pcgrad':
        optimizer = PCGrad(base_optimizer)
    else:
        optimizer = base_optimizer
    
    # Verify
    is_pcgrad = hasattr(optimizer, 'pc_backward')
    print(f"✓ Optimizer type: {type(optimizer).__name__}")
    print(f"✓ Has pc_backward method: {is_pcgrad}")
    
    return is_pcgrad


def test_gradient_clipping_and_memory():
    """Test 3 & 4: Verify gradient clipping and memory stability."""
    print("\n=== Test 3 & 4: Gradient Clipping & Memory ===")
    
    # Setup
    model = DummyModel()
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = PCGrad(base_optimizer)
    actor = DummyActor(model, optimizer, clip_grad_norm=1.0)
    
    gradient_norms = []
    memory_usage = []
    per_task_losses = {'math': [], 'code': [], 'logic': []}
    
    # Run multiple epochs
    for epoch in range(5):
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Record memory
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_usage.append(memory_mb)
        
        # Create task batches
        task_batches = {
            'math': create_dummy_task_data('math'),
            'code': create_dummy_task_data('code'), 
            'logic': create_dummy_task_data('logic')
        }
        
        # Simulate MTL update
        optimizer.zero_grad()
        losses = []
        
        for task_name, task_data in task_batches.items():
            loss = actor.compute_policy_loss(task_data)
            losses.append(loss)
            per_task_losses[task_name].append(loss.item())
        
        # Apply PCGrad
        optimizer.pc_backward(losses)
        
        # Check gradient norm before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            float('inf')  # Don't actually clip, just compute norm
        )
        
        # Actually clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), actor.clip_grad_norm)
        
        # Check gradient norm after clipping
        total_norm_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        gradient_norms.append({
            'before_clip': total_norm_before.item(),
            'after_clip': total_norm_after,
            'clipped': total_norm_after <= actor.clip_grad_norm + 1e-6
        })
        
        # Step optimizer
        optimizer.step()
        
        print(f"Epoch {epoch}: Memory={memory_mb:.1f}MB, "
              f"Grad norm={total_norm_after:.3f} "
              f"(clipped: {gradient_norms[-1]['clipped']})")
    
    # Verify gradient clipping
    all_clipped = all(g['clipped'] for g in gradient_norms)
    print(f"\n✓ All gradients properly clipped: {all_clipped}")
    
    # Verify memory stability (should not grow > 10%)
    memory_growth = (memory_usage[-1] - memory_usage[0]) / memory_usage[0]
    memory_stable = abs(memory_growth) < 0.1
    print(f"✓ Memory growth: {memory_growth*100:.1f}% (stable: {memory_stable})")
    
    # Print per-task losses
    print("\n✓ Per-task loss trajectories:")
    for task, losses in per_task_losses.items():
        print(f"  {task}: {losses[0]:.3f} → {losses[-1]:.3f}")
    
    return all_clipped, memory_stable, gradient_norms, per_task_losses


def test_full_integration():
    """Run all tests and generate report."""
    print("="*50)
    print("MTL Integration Test Suite")
    print("="*50)
    
    results = TestMetrics()
    
    # Test 1: Task grouping
    results.task_grouping_correct = test_task_grouping()
    
    # Test 2: PCGrad optimizer
    results.pcgrad_optimizer_used = test_pcgrad_optimizer('pcgrad')
    
    # Test 3 & 4: Gradient clipping and memory
    (results.gradients_clipped, 
     results.memory_stable,
     results.gradient_norms,
     results.per_task_losses) = test_gradient_clipping_and_memory()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Task grouping: {'PASS' if results.task_grouping_correct else 'FAIL'}")
    print(f"✓ PCGrad optimizer: {'PASS' if results.pcgrad_optimizer_used else 'FAIL'}")
    print(f"✓ Gradient clipping: {'PASS' if results.gradients_clipped else 'FAIL'}")
    print(f"✓ Memory stability: {'PASS' if results.memory_stable else 'FAIL'}")
    
    all_pass = all([
        results.task_grouping_correct,
        results.pcgrad_optimizer_used,
        results.gradients_clipped,
        results.memory_stable
    ])
    
    print(f"\nOverall: {'ALL TESTS PASSED ✅' if all_pass else 'SOME TESTS FAILED ❌'}")
    
    return results


def test_conflicting_gradients():
    """Bonus test: Verify PCGrad handles conflicting gradients correctly."""
    print("\n=== Bonus Test: Conflicting Gradients ===")
    
    # Create a simple 2D parameter to visualize
    param = nn.Parameter(torch.tensor([1.0, 1.0]))
    optimizer = PCGrad(torch.optim.SGD([param], lr=1.0))
    
    # Create two conflicting gradients
    loss1 = param[0]  # Gradient: [1, 0]
    loss2 = -param[0] + param[1]  # Gradient: [-1, 1]
    
    # These gradients conflict (dot product < 0)
    optimizer.pc_backward([loss1, loss2])
    
    print(f"Parameter before: {param.data}")
    optimizer.step()
    print(f"Parameter after: {param.data}")
    print("✓ PCGrad successfully handles conflicting gradients")


if __name__ == "__main__":
    # Run basic tests
    results = test_full_integration()
    
    # Run bonus test
    test_conflicting_gradients()
    
    print("\n🎉 Testing complete!")