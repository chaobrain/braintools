import unittest
import jax.numpy as jnp
from braintools.metric import unitary_LFP


class TestUnitaryLFP(unittest.TestCase):
    def test_invalid_spike_type(self):
        times = jnp.arange(100) * 0.1
        spikes = jnp.ones((100, 10))
        with self.assertRaises(ValueError) as context:
            unitary_LFP(times, spikes, 'invalid_type')
        self.assertIn('"spike_type" should be "exc or ""inh".', str(context.exception))
