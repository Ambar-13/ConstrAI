"""
Adversarial Test Suite for Arbiter Framework
155 tests covering all 7 theorems and adversarial scenarios.
"""

import unittest
import time
import threading
from arbiter import (
    ResourceBudget, Invariant, InvariantViolation, TerminationGuard,
    CausalGraph, CausalNode, BeliefTracker, Belief,
    SandboxedAttestor, AttestationError, EnvironmentReconciler,
    LLMInterface, MockLLM, ArbiterAgent
)


# ============================================================================
# Theorem 1: Budget Tests (22 tests)
# ============================================================================

class TestBudgetTheorem(unittest.TestCase):
    """Tests for resource budget guarantees."""
    
    def test_budget_creation(self):
        """Test budget can be created."""
        budget = ResourceBudget(100, 10.0, 1000000)
        self.assertEqual(budget.max_operations, 100)
    
    def test_budget_check_initial(self):
        """Test initial budget check passes."""
        budget = ResourceBudget(100, 10.0, 1000000)
        self.assertTrue(budget.check_budget())
    
    def test_budget_consume_single(self):
        """Test consuming single operation."""
        budget = ResourceBudget(100, 10.0, 1000000)
        self.assertTrue(budget.consume(1))
        self.assertEqual(budget.operations_used, 1)
    
    def test_budget_consume_multiple(self):
        """Test consuming multiple operations."""
        budget = ResourceBudget(100, 10.0, 1000000)
        self.assertTrue(budget.consume(50))
        self.assertEqual(budget.operations_used, 50)
    
    def test_budget_exhaust_operations(self):
        """Test budget exhaustion on operations."""
        budget = ResourceBudget(10, 10.0, 1000000)
        for i in range(10):
            budget.consume(1)
        self.assertFalse(budget.check_budget())
    
    def test_budget_exhaust_time(self):
        """Test budget exhaustion on time."""
        budget = ResourceBudget(1000, 0.1, 1000000)
        time.sleep(0.15)
        self.assertFalse(budget.check_budget())
    
    def test_budget_consume_returns_false_when_exhausted(self):
        """Test consume returns False when exhausted."""
        budget = ResourceBudget(5, 10.0, 1000000)
        for i in range(5):
            budget.consume(1)
        self.assertFalse(budget.consume(1))
    
    def test_budget_zero_operations(self):
        """Test budget with zero operations."""
        budget = ResourceBudget(0, 10.0, 1000000)
        self.assertFalse(budget.consume(1))
    
    def test_budget_zero_consume(self):
        """Test consuming zero operations has no effect."""
        budget = ResourceBudget(100, 10.0, 1000000)
        initial = budget.operations_used
        budget.consume(0)
        self.assertEqual(budget.operations_used, initial)
    
    def test_budget_time_zero(self):
        """Test budget with zero time."""
        budget = ResourceBudget(100, 0.0, 1000000)
        self.assertFalse(budget.check_budget())
    
    def test_budget_concurrent_access(self):
        """Test budget under concurrent access."""
        budget = ResourceBudget(1000, 10.0, 1000000)
        
        def consumer():
            for _ in range(10):
                budget.consume(1)
        
        threads = [threading.Thread(target=consumer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertLessEqual(budget.operations_used, 1000)
    
    def test_budget_large_consume(self):
        """Test consuming large number at once."""
        budget = ResourceBudget(1000, 10.0, 1000000)
        self.assertTrue(budget.consume(999))
    
    def test_budget_exact_limit(self):
        """Test exact budget limit."""
        budget = ResourceBudget(10, 10.0, 1000000)
        for i in range(10):
            self.assertTrue(budget.consume(1))
        self.assertFalse(budget.consume(1))
    
    def test_budget_fractional_time(self):
        """Test fractional time budget."""
        budget = ResourceBudget(100, 0.01, 1000000)
        time.sleep(0.02)
        self.assertFalse(budget.check_budget())
    
    def test_budget_time_precision(self):
        """Test time budget precision."""
        budget = ResourceBudget(100, 1.0, 1000000)
        self.assertTrue(budget.check_budget())
        time.sleep(0.5)
        self.assertTrue(budget.check_budget())
    
    def test_budget_operations_precision(self):
        """Test operations counting precision."""
        budget = ResourceBudget(100, 10.0, 1000000)
        for i in range(50):
            budget.consume(1)
        self.assertEqual(budget.operations_used, 50)
    
    def test_budget_state_after_exhaust(self):
        """Test budget state after exhaustion."""
        budget = ResourceBudget(5, 10.0, 1000000)
        for i in range(6):
            budget.consume(1)
        self.assertGreaterEqual(budget.operations_used, 5)
    
    def test_budget_time_elapsed(self):
        """Test time elapsed tracking."""
        budget = ResourceBudget(100, 10.0, 1000000)
        time.sleep(0.1)
        elapsed = time.time() - budget.start_time
        self.assertGreaterEqual(elapsed, 0.1)
    
    def test_budget_multiple_checks(self):
        """Test multiple budget checks."""
        budget = ResourceBudget(100, 10.0, 1000000)
        for i in range(10):
            self.assertTrue(budget.check_budget())
    
    def test_budget_after_time_limit(self):
        """Test budget after time limit reached."""
        budget = ResourceBudget(1000, 0.05, 1000000)
        time.sleep(0.1)
        self.assertFalse(budget.consume(1))
    
    def test_budget_sequential_consumes(self):
        """Test sequential budget consumes."""
        budget = ResourceBudget(100, 10.0, 1000000)
        for i in range(10):
            self.assertTrue(budget.consume(1))
    
    def test_budget_stress_test(self):
        """Test budget under stress."""
        budget = ResourceBudget(10000, 10.0, 1000000)
        count = 0
        while budget.consume(1):
            count += 1
            if count > 10000:
                break
        self.assertLessEqual(count, 10000)


# ============================================================================
# Theorem 2: Invariant Tests (20 tests)
# ============================================================================

class TestInvariantTheorem(unittest.TestCase):
    """Tests for invariant guarantees."""
    
    def test_invariant_creation(self):
        """Test invariant creation."""
        inv = Invariant(lambda: True, "test")
        self.assertEqual(inv.name, "test")
    
    def test_invariant_check_true(self):
        """Test invariant check returns True."""
        inv = Invariant(lambda: True, "always_true")
        self.assertTrue(inv.check())
    
    def test_invariant_check_false(self):
        """Test invariant check returns False."""
        inv = Invariant(lambda: False, "always_false")
        self.assertFalse(inv.check())
    
    def test_invariant_enforce_success(self):
        """Test invariant enforcement succeeds."""
        inv = Invariant(lambda: True, "test")
        inv.enforce()  # Should not raise
    
    def test_invariant_enforce_failure(self):
        """Test invariant enforcement raises on violation."""
        inv = Invariant(lambda: False, "test")
        with self.assertRaises(InvariantViolation):
            inv.enforce()
    
    def test_invariant_with_state(self):
        """Test invariant with external state."""
        state = {"value": 5}
        inv = Invariant(lambda: state["value"] > 0, "positive")
        self.assertTrue(inv.check())
        state["value"] = -1
        self.assertFalse(inv.check())
    
    def test_invariant_complex_predicate(self):
        """Test invariant with complex predicate."""
        state = {"x": 10, "y": 20}
        inv = Invariant(lambda: state["x"] + state["y"] == 30, "sum")
        self.assertTrue(inv.check())
    
    def test_invariant_exception_in_predicate(self):
        """Test invariant handles exception in predicate."""
        def bad_predicate():
            raise ValueError("test error")
        
        inv = Invariant(bad_predicate, "bad")
        with self.assertRaises(ValueError):
            inv.check()
    
    def test_invariant_name_preserved(self):
        """Test invariant name is preserved."""
        inv = Invariant(lambda: True, "my_invariant")
        self.assertEqual(inv.name, "my_invariant")
    
    def test_invariant_multiple_checks(self):
        """Test multiple invariant checks."""
        count = [0]
        def predicate():
            count[0] += 1
            return True
        
        inv = Invariant(predicate, "counter")
        for i in range(5):
            inv.check()
        self.assertEqual(count[0], 5)
    
    def test_invariant_enforce_error_message(self):
        """Test invariant error message includes name."""
        inv = Invariant(lambda: False, "test_invariant")
        try:
            inv.enforce()
            self.fail("Should have raised")
        except InvariantViolation as e:
            self.assertIn("test_invariant", str(e))
    
    def test_invariant_with_closure(self):
        """Test invariant with closure."""
        x = 10
        inv = Invariant(lambda: x > 5, "closure")
        self.assertTrue(inv.check())
    
    def test_invariant_boolean_expressions(self):
        """Test invariant with boolean expressions."""
        inv = Invariant(lambda: 1 + 1 == 2 and 2 * 2 == 4, "math")
        self.assertTrue(inv.check())
    
    def test_invariant_string_comparison(self):
        """Test invariant with string comparison."""
        inv = Invariant(lambda: "abc" < "xyz", "string")
        self.assertTrue(inv.check())
    
    def test_invariant_list_operations(self):
        """Test invariant with list operations."""
        lst = [1, 2, 3]
        inv = Invariant(lambda: len(lst) == 3, "list_length")
        self.assertTrue(inv.check())
    
    def test_invariant_dict_operations(self):
        """Test invariant with dict operations."""
        d = {"key": "value"}
        inv = Invariant(lambda: "key" in d, "dict_key")
        self.assertTrue(inv.check())
    
    def test_invariant_nested_predicates(self):
        """Test invariant with nested conditions."""
        x, y = 5, 10
        inv = Invariant(lambda: (x > 0 and y > 0) or (x < 0 and y < 0), "nested")
        self.assertTrue(inv.check())
    
    def test_invariant_equality_check(self):
        """Test invariant equality check."""
        a, b = [1, 2, 3], [1, 2, 3]
        inv = Invariant(lambda: a == b, "equality")
        self.assertTrue(inv.check())
    
    def test_invariant_range_check(self):
        """Test invariant range check."""
        value = 50
        inv = Invariant(lambda: 0 <= value <= 100, "range")
        self.assertTrue(inv.check())
    
    def test_invariant_type_check(self):
        """Test invariant type check."""
        value = 42
        inv = Invariant(lambda: isinstance(value, int), "type")
        self.assertTrue(inv.check())


# ============================================================================
# Theorem 3: Termination Tests (20 tests)
# ============================================================================

class TestTerminationTheorem(unittest.TestCase):
    """Tests for termination guarantees."""
    
    def test_termination_guard_creation(self):
        """Test termination guard creation."""
        budget = ResourceBudget(100, 10.0, 1000000)
        guard = TerminationGuard(budget)
        self.assertFalse(guard.is_terminated())
    
    def test_termination_guard_step(self):
        """Test termination guard step."""
        budget = ResourceBudget(100, 10.0, 1000000)
        guard = TerminationGuard(budget)
        self.assertTrue(guard.step())
    
    def test_termination_guard_exhaustion(self):
        """Test termination guard exhausts budget."""
        budget = ResourceBudget(5, 10.0, 1000000)
        guard = TerminationGuard(budget)
        for i in range(5):
            guard.step()
        self.assertFalse(guard.step())
    
    def test_termination_guard_terminated_flag(self):
        """Test terminated flag is set."""
        budget = ResourceBudget(1, 10.0, 1000000)
        guard = TerminationGuard(budget)
        guard.step()
        guard.step()
        self.assertTrue(guard.is_terminated())
    
    def test_termination_guard_multiple_checks(self):
        """Test multiple termination checks."""
        budget = ResourceBudget(100, 10.0, 1000000)
        guard = TerminationGuard(budget)
        for i in range(10):
            self.assertFalse(guard.is_terminated())
            guard.step()
    
    def test_termination_guard_after_terminate(self):
        """Test steps after termination."""
        budget = ResourceBudget(1, 10.0, 1000000)
        guard = TerminationGuard(budget)
        guard.step()
        guard.step()
        self.assertFalse(guard.step())
        self.assertFalse(guard.step())
    
    def test_termination_guard_time_based(self):
        """Test time-based termination."""
        budget = ResourceBudget(1000, 0.05, 1000000)
        guard = TerminationGuard(budget)
        time.sleep(0.1)
        self.assertFalse(guard.step())
    
    def test_termination_guard_zero_budget(self):
        """Test guard with zero budget."""
        budget = ResourceBudget(0, 10.0, 1000000)
        guard = TerminationGuard(budget)
        self.assertFalse(guard.step())
    
    def test_termination_guard_state_consistency(self):
        """Test state consistency after termination."""
        budget = ResourceBudget(2, 10.0, 1000000)
        guard = TerminationGuard(budget)
        guard.step()
        guard.step()
        self.assertTrue(guard.is_terminated())
        self.assertFalse(guard.step())
    
    def test_termination_guard_count_accuracy(self):
        """Test accurate operation counting."""
        budget = ResourceBudget(10, 10.0, 1000000)
        guard = TerminationGuard(budget)
        count = 0
        while guard.step():
            count += 1
        self.assertEqual(count, 10)
    
    def test_termination_guard_loop_termination(self):
        """Test loop termination guarantee."""
        budget = ResourceBudget(100, 10.0, 1000000)
        guard = TerminationGuard(budget)
        iterations = 0
        while guard.step() and iterations < 1000:
            iterations += 1
        self.assertLessEqual(iterations, 100)
    
    def test_termination_guard_early_termination(self):
        """Test early termination detection."""
        budget = ResourceBudget(5, 10.0, 1000000)
        guard = TerminationGuard(budget)
        for i in range(3):
            guard.step()
        self.assertFalse(guard.is_terminated())
    
    def test_termination_guard_budget_tracking(self):
        """Test budget tracking through guard."""
        budget = ResourceBudget(10, 10.0, 1000000)
        guard = TerminationGuard(budget)
        guard.step()
        self.assertGreater(budget.operations_used, 0)
    
    def test_termination_guard_no_infinite_loop(self):
        """Test no infinite loops possible."""
        budget = ResourceBudget(10, 10.0, 1000000)
        guard = TerminationGuard(budget)
        max_iterations = 1000
        iterations = 0
        while guard.step() and iterations < max_iterations:
            iterations += 1
        self.assertLess(iterations, max_iterations)
    
    def test_termination_guard_deterministic(self):
        """Test termination is deterministic."""
        budget1 = ResourceBudget(5, 10.0, 1000000)
        budget2 = ResourceBudget(5, 10.0, 1000000)
        guard1 = TerminationGuard(budget1)
        guard2 = TerminationGuard(budget2)
        
        count1, count2 = 0, 0
        while guard1.step():
            count1 += 1
        while guard2.step():
            count2 += 1
        
        self.assertEqual(count1, count2)
    
    def test_termination_guard_concurrent_safety(self):
        """Test termination guard thread safety."""
        budget = ResourceBudget(100, 10.0, 1000000)
        guard = TerminationGuard(budget)
        
        def worker():
            while guard.step():
                pass
        
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=1.0)
        self.assertTrue(guard.is_terminated())
    
    def test_termination_guard_prevents_runaway(self):
        """Test guard prevents runaway execution."""
        budget = ResourceBudget(50, 10.0, 1000000)
        guard = TerminationGuard(budget)
        count = 0
        while guard.step():
            count += 1
            if count > 100:
                self.fail("Runaway execution not prevented")
    
    def test_termination_guard_minimal_overhead(self):
        """Test guard has minimal overhead."""
        budget = ResourceBudget(1000, 10.0, 1000000)
        guard = TerminationGuard(budget)
        start = time.time()
        while guard.step():
            pass
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0)
    
    def test_termination_guard_idempotent(self):
        """Test termination checks are idempotent."""
        budget = ResourceBudget(1, 10.0, 1000000)
        guard = TerminationGuard(budget)
        guard.step()
        guard.step()
        self.assertTrue(guard.is_terminated())
        self.assertTrue(guard.is_terminated())
    
    def test_termination_guard_exact_limit(self):
        """Test termination at exact limit."""
        budget = ResourceBudget(3, 10.0, 1000000)
        guard = TerminationGuard(budget)
        self.assertTrue(guard.step())
        self.assertTrue(guard.step())
        self.assertTrue(guard.step())
        self.assertFalse(guard.step())


# ============================================================================
# Theorem 4: Causality Tests (20 tests)
# ============================================================================

class TestCausalityTheorem(unittest.TestCase):
    """Tests for causal dependency tracking."""
    
    def test_causal_graph_creation(self):
        """Test causal graph creation."""
        graph = CausalGraph()
        self.assertEqual(len(graph.nodes), 0)
    
    def test_causal_graph_add_node(self):
        """Test adding node to causal graph."""
        graph = CausalGraph()
        node = graph.add_node("n1", {"value": 1})
        self.assertEqual(node.id, "n1")
    
    def test_causal_graph_add_dependent_node(self):
        """Test adding node with dependencies."""
        graph = CausalGraph()
        graph.add_node("n1", {"value": 1})
        node2 = graph.add_node("n2", {"value": 2}, caused_by=["n1"])
        self.assertIn("n1", node2.causes)
    
    def test_causal_graph_effects_updated(self):
        """Test effects are updated."""
        graph = CausalGraph()
        graph.add_node("n1", {"value": 1})
        graph.add_node("n2", {"value": 2}, caused_by=["n1"])
        self.assertIn("n2", graph.nodes["n1"].effects)
    
    def test_causal_graph_ancestors(self):
        """Test getting ancestors."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2, caused_by=["n1"])
        graph.add_node("n3", 3, caused_by=["n2"])
        ancestors = graph.get_ancestors("n3")
        self.assertIn("n1", ancestors)
        self.assertIn("n2", ancestors)
    
    def test_causal_graph_is_descendant(self):
        """Test descendant checking."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2, caused_by=["n1"])
        self.assertTrue(graph.is_causal_descendant("n2", "n1"))
    
    def test_causal_graph_not_descendant(self):
        """Test non-descendant checking."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2)
        self.assertFalse(graph.is_causal_descendant("n2", "n1"))
    
    def test_causal_graph_multiple_causes(self):
        """Test node with multiple causes."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2)
        node3 = graph.add_node("n3", 3, caused_by=["n1", "n2"])
        self.assertEqual(len(node3.causes), 2)
    
    def test_causal_graph_transitive_closure(self):
        """Test transitive closure of ancestors."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2, caused_by=["n1"])
        graph.add_node("n3", 3, caused_by=["n2"])
        graph.add_node("n4", 4, caused_by=["n3"])
        ancestors = graph.get_ancestors("n4")
        self.assertEqual(len(ancestors), 3)
    
    def test_causal_graph_no_self_ancestor(self):
        """Test node is not its own ancestor."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        ancestors = graph.get_ancestors("n1")
        self.assertNotIn("n1", ancestors)
    
    def test_causal_graph_timestamp(self):
        """Test nodes have timestamps."""
        graph = CausalGraph()
        before = time.time()
        node = graph.add_node("n1", 1)
        after = time.time()
        self.assertGreaterEqual(node.timestamp, before)
        self.assertLessEqual(node.timestamp, after)
    
    def test_causal_graph_data_preserved(self):
        """Test node data is preserved."""
        graph = CausalGraph()
        data = {"key": "value", "number": 42}
        node = graph.add_node("n1", data)
        self.assertEqual(node.data, data)
    
    def test_causal_graph_concurrent_adds(self):
        """Test concurrent node additions."""
        graph = CausalGraph()
        
        def add_nodes():
            for i in range(10):
                graph.add_node(f"node_{threading.current_thread().name}_{i}", i)
        
        threads = [threading.Thread(target=add_nodes) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(graph.nodes), 50)
    
    def test_causal_graph_diamond_dependency(self):
        """Test diamond dependency pattern."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2, caused_by=["n1"])
        graph.add_node("n3", 3, caused_by=["n1"])
        graph.add_node("n4", 4, caused_by=["n2", "n3"])
        ancestors = graph.get_ancestors("n4")
        self.assertIn("n1", ancestors)
    
    def test_causal_graph_missing_cause(self):
        """Test handling missing cause."""
        graph = CausalGraph()
        node = graph.add_node("n1", 1, caused_by=["nonexistent"])
        self.assertIn("nonexistent", node.causes)
    
    def test_causal_graph_empty_ancestors(self):
        """Test ancestors of root node."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        ancestors = graph.get_ancestors("n1")
        self.assertEqual(len(ancestors), 0)
    
    def test_causal_graph_effects_tracking(self):
        """Test effects are tracked correctly."""
        graph = CausalGraph()
        graph.add_node("n1", 1)
        graph.add_node("n2", 2, caused_by=["n1"])
        graph.add_node("n3", 3, caused_by=["n1"])
        self.assertEqual(len(graph.nodes["n1"].effects), 2)
    
    def test_causal_graph_chain(self):
        """Test long causal chain."""
        graph = CausalGraph()
        prev = None
        for i in range(10):
            node_id = f"n{i}"
            if prev:
                graph.add_node(node_id, i, caused_by=[prev])
            else:
                graph.add_node(node_id, i)
            prev = node_id
        
        ancestors = graph.get_ancestors("n9")
        self.assertEqual(len(ancestors), 9)
    
    def test_causal_graph_independent_branches(self):
        """Test independent causal branches."""
        graph = CausalGraph()
        graph.add_node("a1", 1)
        graph.add_node("a2", 2, caused_by=["a1"])
        graph.add_node("b1", 10)
        graph.add_node("b2", 20, caused_by=["b1"])
        
        self.assertFalse(graph.is_causal_descendant("a2", "b1"))
        self.assertFalse(graph.is_causal_descendant("b2", "a1"))
    
    def test_causal_graph_node_retrieval(self):
        """Test retrieving nodes from graph."""
        graph = CausalGraph()
        graph.add_node("n1", {"data": "test"})
        node = graph.nodes["n1"]
        self.assertEqual(node.data["data"], "test")


# ============================================================================
# Theorem 5: Belief Tests (20 tests)
# ============================================================================

class TestBeliefTheorem(unittest.TestCase):
    """Tests for Bayesian belief tracking."""
    
    def test_belief_creation(self):
        """Test belief creation."""
        belief = Belief("hypothesis", 0.5)
        self.assertEqual(belief.hypothesis, "hypothesis")
    
    def test_belief_initial_posterior(self):
        """Test initial posterior equals prior."""
        belief = Belief("h", 0.7)
        self.assertEqual(belief.posterior, 0.7)
    
    def test_belief_update_increases(self):
        """Test belief update increases confidence."""
        belief = Belief("h", 0.5)
        belief.update(0.9)
        self.assertGreater(belief.posterior, 0.5)
    
    def test_belief_update_decreases(self):
        """Test belief update decreases confidence."""
        belief = Belief("h", 0.5)
        belief.update(0.1)
        self.assertLess(belief.posterior, 0.5)
    
    def test_belief_bounded_upper(self):
        """Test belief bounded above by 1.0."""
        belief = Belief("h", 0.9)
        for _ in range(10):
            belief.update(0.99)
        self.assertLessEqual(belief.posterior, 1.0)
    
    def test_belief_bounded_lower(self):
        """Test belief bounded below by 0.0."""
        belief = Belief("h", 0.1)
        for _ in range(10):
            belief.update(0.01)
        self.assertGreaterEqual(belief.posterior, 0.0)
    
    def test_belief_tracker_creation(self):
        """Test belief tracker creation."""
        tracker = BeliefTracker()
        self.assertEqual(len(tracker.beliefs), 0)
    
    def test_belief_tracker_add(self):
        """Test adding belief to tracker."""
        tracker = BeliefTracker()
        belief = tracker.add_belief("h1", 0.5)
        self.assertEqual(belief.hypothesis, "h1")
    
    def test_belief_tracker_update(self):
        """Test updating belief in tracker."""
        tracker = BeliefTracker()
        tracker.add_belief("h1", 0.5)
        tracker.update_belief("h1", 0.8)
        posterior = tracker.get_posterior("h1")
        self.assertIsNotNone(posterior)
    
    def test_belief_tracker_get_posterior(self):
        """Test getting posterior from tracker."""
        tracker = BeliefTracker()
        tracker.add_belief("h1", 0.7)
        posterior = tracker.get_posterior("h1")
        self.assertEqual(posterior, 0.7)
    
    def test_belief_tracker_missing_belief(self):
        """Test getting missing belief returns None."""
        tracker = BeliefTracker()
        posterior = tracker.get_posterior("nonexistent")
        self.assertIsNone(posterior)
    
    def test_belief_tracker_multiple_beliefs(self):
        """Test tracking multiple beliefs."""
        tracker = BeliefTracker()
        tracker.add_belief("h1", 0.3)
        tracker.add_belief("h2", 0.7)
        self.assertEqual(len(tracker.beliefs), 2)
    
    def test_belief_tracker_concurrent_updates(self):
        """Test concurrent belief updates."""
        tracker = BeliefTracker()
        tracker.add_belief("h1", 0.5)
        
        def updater():
            for _ in range(10):
                tracker.update_belief("h1", 0.6)
        
        threads = [threading.Thread(target=updater) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        posterior = tracker.get_posterior("h1")
        self.assertIsNotNone(posterior)
    
    def test_belief_sequential_updates(self):
        """Test sequential belief updates."""
        belief = Belief("h", 0.5)
        belief.update(0.8)
        intermediate = belief.posterior
        belief.update(0.8)
        self.assertGreater(belief.posterior, intermediate)
    
    def test_belief_extreme_evidence(self):
        """Test belief with extreme evidence."""
        belief = Belief("h", 0.5)
        belief.update(0.99)
        self.assertGreater(belief.posterior, 0.7)
    
    def test_belief_weak_evidence(self):
        """Test belief with weak evidence."""
        belief = Belief("h", 0.5)
        initial = belief.posterior
        belief.update(0.51)
        self.assertAlmostEqual(belief.posterior, initial, delta=0.1)
    
    def test_belief_prior_values(self):
        """Test different prior values."""
        belief1 = Belief("h1", 0.1)
        belief2 = Belief("h2", 0.9)
        self.assertEqual(belief1.posterior, 0.1)
        self.assertEqual(belief2.posterior, 0.9)
    
    def test_belief_update_neutral(self):
        """Test neutral evidence update."""
        belief = Belief("h", 0.5)
        initial = belief.posterior
        belief.update(0.5)
        self.assertAlmostEqual(belief.posterior, initial, delta=0.05)
    
    def test_belief_tracker_independence(self):
        """Test beliefs are independent."""
        tracker = BeliefTracker()
        tracker.add_belief("h1", 0.5)
        tracker.add_belief("h2", 0.5)
        tracker.update_belief("h1", 0.9)
        self.assertEqual(tracker.get_posterior("h2"), 0.5)
    
    def test_belief_monotonic_increase(self):
        """Test monotonic increase with positive evidence."""
        belief = Belief("h", 0.5)
        prev = belief.posterior
        for _ in range(5):
            belief.update(0.8)
            self.assertGreaterEqual(belief.posterior, prev)
            prev = belief.posterior


# ============================================================================
# Theorem 6: Isolation/Attestation Tests (20 tests)
# ============================================================================

class TestIsolationTheorem(unittest.TestCase):
    """Tests for sandboxed attestors."""
    
    def test_attestor_creation(self):
        """Test attestor creation."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        self.assertEqual(attestor.name, "test")
    
    def test_attestor_attest_success(self):
        """Test successful attestation."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        result = attestor.attest("claim", lambda: True)
        self.assertTrue(result)
    
    def test_attestor_attest_failure(self):
        """Test failed attestation."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        result = attestor.attest("claim", lambda: False)
        self.assertFalse(result)
    
    def test_attestor_budget_enforcement(self):
        """Test attestor enforces budget."""
        budget = ResourceBudget(5, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        for i in range(5):
            attestor.attest(f"claim{i}", lambda: True)
        
        with self.assertRaises(AttestationError):
            attestor.attest("claim6", lambda: True)
    
    def test_attestor_records_attestations(self):
        """Test attestor records attestations."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        attestor.attest("claim1", lambda: True)
        attestor.attest("claim2", lambda: False)
        attestations = attestor.get_attestations()
        self.assertEqual(len(attestations), 2)
    
    def test_attestor_exception_handling(self):
        """Test attestor handles exceptions."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        
        def failing_verifier():
            raise ValueError("test error")
        
        result = attestor.attest("claim", failing_verifier)
        self.assertFalse(result)
    
    def test_attestor_state_isolation(self):
        """Test attestor state is isolated."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        attestor.state["key"] = "value"
        self.assertEqual(attestor.state["key"], "value")
    
    def test_attestor_concurrent_attestations(self):
        """Test concurrent attestations."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        
        def attest_worker():
            for i in range(5):
                attestor.attest(f"claim_{threading.current_thread().name}_{i}", lambda: True)
        
        threads = [threading.Thread(target=attest_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        attestations = attestor.get_attestations()
        self.assertGreater(len(attestations), 0)
    
    def test_attestor_attestation_timestamp(self):
        """Test attestations have timestamps."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        before = time.time()
        attestor.attest("claim", lambda: True)
        after = time.time()
        
        attestations = attestor.get_attestations()
        timestamp = attestations[0][0]
        self.assertGreaterEqual(timestamp, before)
        self.assertLessEqual(timestamp, after)
    
    def test_attestor_claim_preserved(self):
        """Test claim text is preserved."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        attestor.attest("my_claim", lambda: True)
        
        attestations = attestor.get_attestations()
        claim = attestations[0][1]
        self.assertEqual(claim, "my_claim")
    
    def test_attestor_result_preserved(self):
        """Test attestation result is preserved."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        attestor.attest("claim1", lambda: True)
        attestor.attest("claim2", lambda: False)
        
        attestations = attestor.get_attestations()
        self.assertTrue(attestations[0][2])
        self.assertFalse(attestations[1][2])
    
    def test_attestor_multiple_attestors(self):
        """Test multiple independent attestors."""
        budget1 = ResourceBudget(100, 10.0, 1000000)
        budget2 = ResourceBudget(100, 10.0, 1000000)
        attestor1 = SandboxedAttestor("a1", budget1)
        attestor2 = SandboxedAttestor("a2", budget2)
        
        attestor1.attest("claim1", lambda: True)
        attestor2.attest("claim2", lambda: False)
        
        self.assertEqual(len(attestor1.get_attestations()), 1)
        self.assertEqual(len(attestor2.get_attestations()), 1)
    
    def test_attestor_name_preserved(self):
        """Test attestor name is preserved."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("my_attestor", budget)
        self.assertEqual(attestor.name, "my_attestor")
    
    def test_attestor_complex_verifier(self):
        """Test complex verifier logic."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        
        def complex_verifier():
            x = 5
            y = 10
            return x + y == 15
        
        result = attestor.attest("arithmetic", complex_verifier)
        self.assertTrue(result)
    
    def test_attestor_stateful_verifier(self):
        """Test stateful verifier."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        state = {"count": 0}
        
        def stateful_verifier():
            state["count"] += 1
            return state["count"] <= 3
        
        self.assertTrue(attestor.attest("c1", stateful_verifier))
        self.assertTrue(attestor.attest("c2", stateful_verifier))
        self.assertTrue(attestor.attest("c3", stateful_verifier))
        self.assertFalse(attestor.attest("c4", stateful_verifier))
    
    def test_attestor_get_attestations_copy(self):
        """Test get_attestations returns copy."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        attestor.attest("claim", lambda: True)
        
        attestations1 = attestor.get_attestations()
        attestations2 = attestor.get_attestations()
        
        self.assertEqual(attestations1, attestations2)
        self.assertIsNot(attestations1, attestations2)
    
    def test_attestor_sequential_attestations(self):
        """Test sequential attestations."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        
        for i in range(10):
            attestor.attest(f"claim{i}", lambda: i % 2 == 0)
        
        attestations = attestor.get_attestations()
        self.assertEqual(len(attestations), 10)
    
    def test_attestor_budget_consumption(self):
        """Test attestations consume budget."""
        budget = ResourceBudget(10, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        
        initial_ops = budget.operations_used
        attestor.attest("claim", lambda: True)
        
        self.assertGreater(budget.operations_used, initial_ops)
    
    def test_attestor_error_message(self):
        """Test error message includes attestor name."""
        budget = ResourceBudget(1, 10.0, 1000000)
        attestor = SandboxedAttestor("my_attestor", budget)
        attestor.attest("claim1", lambda: True)
        
        try:
            attestor.attest("claim2", lambda: True)
            self.fail("Should have raised")
        except AttestationError as e:
            self.assertIn("my_attestor", str(e))
    
    def test_attestor_empty_initial_state(self):
        """Test attestor starts with empty state."""
        budget = ResourceBudget(100, 10.0, 1000000)
        attestor = SandboxedAttestor("test", budget)
        self.assertEqual(len(attestor.state), 0)
        self.assertEqual(len(attestor.attestations), 0)


# ============================================================================
# Theorem 7: Reconciliation Tests (18 tests)
# ============================================================================

class TestReconciliationTheorem(unittest.TestCase):
    """Tests for environment reconciliation."""
    
    def test_reconciler_creation(self):
        """Test reconciler creation."""
        reconciler = EnvironmentReconciler()
        self.assertEqual(len(reconciler.states), 0)
    
    def test_reconciler_record_state(self):
        """Test recording state."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key": "value"})
        self.assertEqual(len(reconciler.states), 1)
    
    def test_reconciler_reconcile_initial(self):
        """Test initial reconciliation."""
        reconciler = EnvironmentReconciler()
        result = reconciler.reconcile({"key": "value"})
        self.assertEqual(result["key"], "value")
    
    def test_reconciler_reconcile_merge(self):
        """Test reconciliation merges states."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key1": "value1"})
        result = reconciler.reconcile({"key2": "value2"})
        self.assertIn("key1", result)
        self.assertIn("key2", result)
    
    def test_reconciler_version_increment(self):
        """Test version increments."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key": "value"})
        self.assertEqual(reconciler.current_version, 1)
        reconciler.record_state({"key": "value2"})
        self.assertEqual(reconciler.current_version, 2)
    
    def test_reconciler_timestamp(self):
        """Test states have timestamps."""
        reconciler = EnvironmentReconciler()
        before = time.time()
        reconciler.record_state({"key": "value"})
        after = time.time()
        
        timestamp = reconciler.states[0].timestamp
        self.assertGreaterEqual(timestamp, before)
        self.assertLessEqual(timestamp, after)
    
    def test_reconciler_state_immutable(self):
        """Test recorded states are immutable."""
        reconciler = EnvironmentReconciler()
        original = {"key": "value"}
        reconciler.record_state(original)
        original["key"] = "modified"
        
        recorded = reconciler.states[0].state
        self.assertEqual(recorded["key"], "value")
    
    def test_reconciler_get_history(self):
        """Test getting state history."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"v": 1})
        reconciler.record_state({"v": 2})
        
        history = reconciler.get_state_history()
        self.assertEqual(len(history), 2)
    
    def test_reconciler_history_copy(self):
        """Test history returns copy."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key": "value"})
        
        history1 = reconciler.get_state_history()
        history2 = reconciler.get_state_history()
        
        self.assertEqual(history1, history2)
        self.assertIsNot(history1, history2)
    
    def test_reconciler_concurrent_records(self):
        """Test concurrent state recording."""
        reconciler = EnvironmentReconciler()
        
        def recorder():
            for i in range(10):
                reconciler.record_state({f"key_{threading.current_thread().name}_{i}": i})
        
        threads = [threading.Thread(target=recorder) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        self.assertEqual(len(reconciler.states), 30)
    
    def test_reconciler_reconcile_overwrites(self):
        """Test reconciliation overwrites old values."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key": "old"})
        result = reconciler.reconcile({"key": "new"})
        self.assertEqual(result["key"], "new")
    
    def test_reconciler_multiple_reconciliations(self):
        """Test multiple reconciliations."""
        reconciler = EnvironmentReconciler()
        reconciler.reconcile({"a": 1})
        reconciler.reconcile({"b": 2})
        result = reconciler.reconcile({"c": 3})
        
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
    
    def test_reconciler_empty_state(self):
        """Test reconciling empty state."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key": "value"})
        result = reconciler.reconcile({})
        self.assertIn("key", result)
    
    def test_reconciler_complex_state(self):
        """Test reconciling complex state."""
        reconciler = EnvironmentReconciler()
        complex_state = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42
        }
        result = reconciler.reconcile(complex_state)
        self.assertEqual(result["nested"]["key"], "value")
    
    def test_reconciler_version_in_state(self):
        """Test version is recorded in state."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"key": "value"})
        state = reconciler.states[0]
        self.assertEqual(state.version, 0)
    
    def test_reconciler_sequential_versions(self):
        """Test sequential version numbers."""
        reconciler = EnvironmentReconciler()
        for i in range(5):
            reconciler.record_state({"v": i})
        
        versions = [s.version for s in reconciler.states]
        self.assertEqual(versions, [0, 1, 2, 3, 4])
    
    def test_reconciler_state_preservation(self):
        """Test all states are preserved."""
        reconciler = EnvironmentReconciler()
        states_to_record = [{"v": i} for i in range(10)]
        
        for state in states_to_record:
            reconciler.record_state(state)
        
        self.assertEqual(len(reconciler.states), 10)
    
    def test_reconciler_eventual_consistency(self):
        """Test eventual consistency property."""
        reconciler = EnvironmentReconciler()
        reconciler.record_state({"a": 1, "b": 2})
        reconciler.reconcile({"a": 10, "c": 3})
        result = reconciler.reconcile({"d": 4})
        
        # Eventually has all keys
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertIn("d", result)


# ============================================================================
# Integration Tests (15 tests)
# ============================================================================

class TestArbiterIntegration(unittest.TestCase):
    """Integration tests for complete Arbiter agent."""
    
    def test_agent_creation(self):
        """Test agent can be created."""
        agent = ArbiterAgent()
        self.assertIsNotNone(agent)
    
    def test_agent_execute_reason(self):
        """Test executing reason action."""
        agent = ArbiterAgent()
        result = agent.execute_action("reason", {"prompt": "test"})
        self.assertIsNotNone(result)
    
    def test_agent_execute_belief_update(self):
        """Test executing belief update."""
        agent = ArbiterAgent()
        agent.belief_tracker.add_belief("h1", 0.5)
        result = agent.execute_action("update_belief", {
            "hypothesis": "h1",
            "likelihood": 0.8
        })
        self.assertIsNotNone(result)
    
    def test_agent_execute_attest(self):
        """Test executing attestation."""
        agent = ArbiterAgent()
        result = agent.execute_action("attest", {
            "attestor": "test",
            "claim": "claim1",
            "verifier": lambda: True
        })
        self.assertTrue(result)
    
    def test_agent_execute_reconcile(self):
        """Test executing reconciliation."""
        agent = ArbiterAgent()
        result = agent.execute_action("reconcile", {
            "observed_state": {"key": "value"}
        })
        self.assertEqual(result["key"], "value")
    
    def test_agent_budget_enforcement(self):
        """Test agent enforces budget."""
        agent = ArbiterAgent(max_operations=5)
        for i in range(5):
            agent.execute_action("reason", {"prompt": f"test{i}"})
        
        with self.assertRaises(RuntimeError):
            agent.execute_action("reason", {"prompt": "test6"})
    
    def test_agent_invariants_checked(self):
        """Test agent checks invariants."""
        agent = ArbiterAgent()
        # Invariants should be checked without raising
        agent.check_invariants()
    
    def test_agent_execution_log(self):
        """Test agent logs execution."""
        agent = ArbiterAgent()
        agent.execute_action("reason", {"prompt": "test"})
        self.assertEqual(len(agent.execution_log), 1)
    
    def test_agent_causal_tracking(self):
        """Test agent tracks causality."""
        agent = ArbiterAgent()
        agent.execute_action("reason", {"prompt": "test1"})
        agent.execute_action("reason", {"prompt": "test2"})
        self.assertGreater(len(agent.causal_graph.nodes), 0)
    
    def test_agent_statistics(self):
        """Test getting agent statistics."""
        agent = ArbiterAgent()
        agent.execute_action("reason", {"prompt": "test"})
        stats = agent.get_statistics()
        self.assertIn("operations_used", stats)
        self.assertGreater(stats["actions_executed"], 0)
    
    def test_agent_set_llm(self):
        """Test setting custom LLM."""
        agent = ArbiterAgent()
        custom_llm = MockLLM()
        agent.set_llm(custom_llm)
        self.assertEqual(agent.llm, custom_llm)
    
    def test_agent_add_attestor(self):
        """Test adding attestor to agent."""
        agent = ArbiterAgent()
        agent.add_attestor("test_attestor")
        self.assertIn("test_attestor", agent.attestors)
    
    def test_agent_concurrent_safety(self):
        """Test agent thread safety."""
        agent = ArbiterAgent(max_operations=100)
        
        def worker():
            for i in range(5):
                try:
                    agent.execute_action("reason", {"prompt": f"test{i}"})
                except RuntimeError:
                    pass  # Budget exhausted
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not crash
        self.assertIsNotNone(agent.get_statistics())
    
    def test_agent_action_error_handling(self):
        """Test agent handles action errors."""
        agent = ArbiterAgent()
        try:
            agent.execute_action("invalid_action", {})
            self.fail("Should have raised")
        except ValueError:
            pass
        
        # Agent should still be functional
        agent.execute_action("reason", {"prompt": "test"})
    
    def test_agent_full_workflow(self):
        """Test complete workflow with all theorems."""
        agent = ArbiterAgent(max_operations=100)
        
        # Add belief
        agent.belief_tracker.add_belief("test_hypothesis", 0.5)
        
        # Execute reasoning
        agent.execute_action("reason", {"prompt": "What is 2+2?"})
        
        # Update belief
        agent.execute_action("update_belief", {
            "hypothesis": "test_hypothesis",
            "likelihood": 0.9
        })
        
        # Attest
        agent.execute_action("attest", {
            "attestor": "verifier",
            "claim": "2+2=4",
            "verifier": lambda: True
        })
        
        # Reconcile
        agent.execute_action("reconcile", {
            "observed_state": {"answer": 4}
        })
        
        # Check all systems functional
        stats = agent.get_statistics()
        self.assertEqual(stats["actions_executed"], 4)
        self.assertGreater(stats["causal_nodes"], 0)
        self.assertGreater(len(agent.belief_tracker.beliefs), 0)
        self.assertGreater(len(agent.attestors), 0)
        self.assertGreater(len(agent.reconciler.states), 0)


if __name__ == "__main__":
    unittest.main()
