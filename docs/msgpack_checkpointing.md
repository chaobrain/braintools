# Save and Load Model States with `msgpack` Checkpointing

This tutorial demonstrates how to use BrainTools' checkpointing system to save and restore model states using the msgpack format. The checkpointing system provides efficient serialization of complex neural network states, including weights, biases, and custom objects.

## Core Functions

- `msgpack_save(filename, data)` - Save data to msgpack file
- `msgpack_load(filename, target=None)` - Load data from msgpack file
- `msgpack_register_serialization(cls, to_dict, from_dict)` - Register custom serialization

## Basic Checkpointing

### Simple Data Structures

The most basic use case involves saving and loading simple data structures:


```python
import jax.numpy as jnp

import braintools

# Create some model data
model_data = {
    'weights': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    'bias': jnp.array([0.1, 0.2]),
    'config': {
        'learning_rate': 0.001,
        'batch_size': 32
    }
}

# Save to checkpoint
checkpoint_path = "files/model_checkpoint.msgpack"
braintools.file.msgpack_save(checkpoint_path, model_data)
print(f"Model saved to {checkpoint_path}")

# Load from checkpoint
loaded_data = braintools.file.msgpack_load(checkpoint_path)
print("Model loaded successfully!")
print(f"Weights shape: {loaded_data['weights'].shape}")
print(f"Config: {loaded_data['config']}")
```

    Saving checkpoint into model_checkpoint.msgpack
    Model saved to model_checkpoint.msgpack
    Loading checkpoint from model_checkpoint.msgpack
    Model loaded successfully!
    Weights shape: (2, 2)
    Config: {'learning_rate': 0.001, 'batch_size': 32}
    

### Working with Templates

For structured restoration, you can provide a template that defines the expected structure:


```python
# Create a template with the expected structure
template = {
    'weights': jnp.zeros((2, 2)),  # Shape and dtype information
    'bias': jnp.zeros(2),
    'config': {'learning_rate': 0.0, 'batch_size': 0}
}

# Load with template to ensure type safety
loaded_data = braintools.file.msgpack_load(checkpoint_path, target=template)
print("Loaded with template validation")
```

    Loading checkpoint from model_checkpoint.msgpack
    Loaded with template validation
    

## Working with BrainState Objects

BrainTools provides special support for BrainState objects, which are commonly used in neural network implementations:


```python
import brainstate


# Create BrainState objects
class SimpleModel(brainstate.nn.Module):
    def __init__(self):
        self.weight = brainstate.ParamState(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        self.bias = brainstate.ParamState(jnp.array([0.1, 0.2]))
        self.running_mean = brainstate.State(jnp.array([0.0, 0.0]))


# Initialize model
model = SimpleModel()

# Create checkpoint data
checkpoint_data = {
    'model_state': model.states(),
    'training_step': 1000,
    'epoch': 10
}

# Save checkpoint
braintools.file.msgpack_save("files/model_state.msgpack", checkpoint_data)

# Create new model instance for loading
new_model = SimpleModel()
template = {
    'model_state': new_model.states(),
    'training_step': 0,
    'epoch': 0
}

# Load and restore state
restored_data = braintools.file.msgpack_load("files/model_state.msgpack", target=template)
print(f"Restored to epoch {restored_data['epoch']}, step {restored_data['training_step']}")
```

    Saving checkpoint into model_state.msgpack
    Loading checkpoint from model_state.msgpack
    Restored to epoch 10, step 1000
    

## Custom Object Serialization

For custom objects, you can register serialization handlers:


```python
from typing import Dict, Any


# Define a custom class
class CustomLayer:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = brainstate.random.randn(input_size, output_size)
        self.bias = jnp.zeros(output_size)

    def __call__(self, x):
        return jnp.dot(x, self.weights) + self.bias


# Register serialization for CustomLayer
def layer_to_state_dict(layer: CustomLayer) -> Dict[str, Any]:
    return {
        'input_size': layer.input_size,
        'output_size': layer.output_size,
        'weights': layer.weights,
        'bias': layer.bias
    }


def layer_from_state_dict(
    layer: CustomLayer,
    state_dict: Dict[str, Any],
    mismatch: str = 'error'
) -> CustomLayer:
    # Create new layer with restored parameters
    new_layer = CustomLayer(state_dict['input_size'], state_dict['output_size'])
    new_layer.weights = state_dict['weights']
    new_layer.bias = state_dict['bias']
    return new_layer


# Register the serialization
braintools.file.msgpack_register_serialization(
    CustomLayer,
    layer_to_state_dict,
    layer_from_state_dict
)

# Now you can save and load CustomLayer objects
layer = CustomLayer(10, 5)
data = {'my_layer': layer, 'metadata': {'version': '1.0'}}

braintools.file.msgpack_save("files/custom_layer.msgpack", data)

# Load with template
template = {'my_layer': CustomLayer(10, 5), 'metadata': {'version': ''}}
loaded = braintools.file.msgpack_load("files/custom_layer.msgpack", target=template)
print(f"Loaded layer with shape: {loaded['my_layer'].weights.shape}")
```

    Saving checkpoint into custom_layer.msgpack
    Loading checkpoint from custom_layer.msgpack
    Loaded layer with shape: (10, 5)
    

## Mismatch Handling

The checkpointing system provides flexible mismatch handling for cases where the saved state doesn't exactly match the target structure:


```python
# Create model with different structure
original_model = {
    'layer1': {'weights': jnp.ones((5, 3)), 'bias': jnp.zeros(3)},
    'layer2': {'weights': jnp.ones((3, 2)), 'bias': jnp.zeros(2)},
    'config': {'lr': 0.01}
}

braintools.file.msgpack_save("files/original.msgpack", original_model)

# New model with additional components
new_model = {
    'layer1': {'weights': jnp.zeros((5, 3)), 'bias': jnp.zeros(3)},
    'layer2': {'weights': jnp.zeros((3, 2)), 'bias': jnp.zeros(2)},
    'layer3': {'weights': jnp.zeros((2, 1)), 'bias': jnp.zeros(1)},  # New layer
    'config': {'lr': 0.001, 'momentum': 0.9}  # New parameter
}

# Different mismatch handling strategies:

# 1. Error on mismatch (default)
try:
    loaded = braintools.file.msgpack_load("files/original.msgpack", target=new_model, mismatch='error')
except ValueError as e:
    print(f"Error mode caught mismatch: {e}")

# 2. Warn on mismatch but continue
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    loaded = braintools.file.msgpack_load("files/original.msgpack", target=new_model, mismatch='warn')
    if w:
        print(f"Warning: {w[0].message}")
    print("Loaded with warnings - missing components kept from target")
    print(f"layer3 weights preserved: {jnp.allclose(loaded['layer3']['weights'], new_model['layer3']['weights'])}")

# 3. Ignore mismatches silently
loaded = braintools.file.msgpack_load("files/original.msgpack", target=new_model, mismatch='ignore')
print("Loaded silently - new components preserved from target")
```

    Saving checkpoint into original.msgpack
    Loading checkpoint from original.msgpack
    Error mode caught mismatch: The target dict keys and state dict keys do not match, target dict contains keys {'layer3'} which are not present in state dict at path .
    Loading checkpoint from original.msgpack
    Warning: The target dict keys and state dict keys do not match, target dict contains keys {'layer3'} which are not present in state dict at path .
    Loaded with warnings - missing components kept from target
    layer3 weights preserved: True
    Loading checkpoint from original.msgpack
    Loaded silently - new components preserved from target
    

## Advanced Usage

### Async Checkpointing

For large models, you can use asynchronous checkpointing to avoid blocking training:


```python
# Create async manager
async_manager = braintools.file.AsyncManager(max_workers=2)

# Large model data
large_model = {
    'embeddings': brainstate.random.randn(100000, 512),
    'weights': [brainstate.random.normal(512, 512) for i in range(10)],
    'step': 5000
}

# Save asynchronously
braintools.file.msgpack_save("files/large_model.msgpack", large_model, async_manager=async_manager)
print("Checkpoint initiated asynchronously")

# Continue with other work...
print("Doing other work while checkpoint saves...")

# Wait for completion when needed
async_manager.wait_previous_save()
print("Checkpoint completed!")
```

    Saving checkpoint into large_model.msgpack
    Checkpoint initiated asynchronously
    Doing other work while checkpoint saves...
    

    D:\codes\projects\braintools\braintools\file\msg_checkpoint.py:650: UserWarning: The previous async brainpy.checkpoints.save has not finished yet. Waiting for it to complete before the next save.
      warnings.warn(
    

    Checkpoint completed!
    

### Working with BrainUnit Quantities

BrainTools automatically handles BrainUnit quantities:


```python
import brainunit as u

# Model with physical units
physics_model = {
    'time_constant': 10.0 * u.ms,
    'resistance': 100.0 * u.mohm,
    'capacitance': 200.0 * u.pF,
    'voltage_threshold': -50.0 * u.mV
}
loaded_physics = physics_model.copy()

braintools.file.msgpack_save("files/physics_model.msgpack", physics_model)
loaded_physics = braintools.file.msgpack_load("files/physics_model.msgpack", target=loaded_physics)

print(f"Time constant: {loaded_physics['time_constant']}")
print(f"Units preserved: {loaded_physics['time_constant'].unit}")
```

    Saving checkpoint into physics_model.msgpack
    Loading checkpoint from physics_model.msgpack
    Time constant: 10.0 * second
    Units preserved: 10.0^-3 * s
    

### Handling Complex Nested Structures


```python
# Complex nested model
complex_model = {
    'encoder': {
        'layers': [
            {'weights': brainstate.random.randn(100, 64), 'bias': jnp.zeros(64)},
            {'weights': brainstate.random.randn(4, 32), 'bias': jnp.zeros(32)}
        ],
        'config': {'activation': 'relu', 'dropout': 0.1}
    },
    'decoder': {
        'layers': [
            {'weights': brainstate.random.randn(32, 64), 'bias': jnp.zeros(64)},
            {'weights': brainstate.random.randn(64, 100), 'bias': jnp.zeros(100)}
        ],
        'config': {'activation': 'sigmoid'}
    },
    'optimizer_state': {
        'momentum': [jnp.zeros_like(w) for w in [
            brainstate.random.randn(100, 64),
            brainstate.random.randn(64, 32),
            brainstate.random.randn(32, 64),
            brainstate.random.randn(64, 100)
        ]],
        'learning_rate': 0.001,
        'step': 1000
    }
}

# Save complex model
braintools.file.msgpack_save("files/complex_model.msgpack", complex_model)

# Load and verify structure
loaded_complex = braintools.file.msgpack_load("files/complex_model.msgpack")
print(f"Encoder layers: {len(loaded_complex['encoder']['layers'])}")
print(f"Optimizer step: {loaded_complex['optimizer_state']['step']}")
```

    Saving checkpoint into complex_model.msgpack
    Loading checkpoint from complex_model.msgpack
    Encoder layers: 2
    Optimizer step: 1000
    

## Best Practices

### 1. Version Compatibility

Include version information in your checkpoints:


```python
import braintools
import time

# Example training config (you would define this based on your needs)
training_config = {'lr': 0.001, 'batch_size': 32}
your_model_data = {'weights': jnp.ones((10, 10))}  # Example model data

checkpoint_data = {
    'model': your_model_data,
    'metadata': {
        'braintools_version': braintools.__version__,
        'model_version': '1.2.0',
        'timestamp': time.time(),
        'training_config': training_config
    }
}

braintools.file.msgpack_save("files/versioned_checkpoint.msgpack", checkpoint_data)
print(f"Checkpoint saved with BrainTools version {braintools.__version__}")
```

    Saving checkpoint into versioned_checkpoint.msgpack
    Checkpoint saved with BrainTools version 0.0.12
    

### 2. Error Handling

Always handle potential loading errors:


```python
def safe_load_checkpoint(checkpoint_path, template=None):
    try:
        if template is not None:
            return braintools.file.msgpack_load(checkpoint_path, target=template, mismatch='warn')
        else:
            return braintools.file.msgpack_load(checkpoint_path)
    except FileNotFoundError:
        print(f"Checkpoint {checkpoint_path} not found")
        return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


# Test the safe loading function
result = safe_load_checkpoint("files/nonexistent_checkpoint.msgpack")
print(f"Safe load result for missing file: {result}")

# Test with existing file
result = safe_load_checkpoint("files/model_checkpoint.msgpack")
if result:
    print("Successfully loaded existing checkpoint")
```

    Error loading checkpoint: Checkpoint not found: nonexistent_checkpoint.msgpack
    Safe load result for missing file: None
    Loading checkpoint from model_checkpoint.msgpack
    Successfully loaded existing checkpoint
    

### 3. Checkpoint Validation

Validate critical components after loading:


```python
def validate_checkpoint(loaded_data, expected_shapes):
    for key, expected_shape in expected_shapes.items():
        if key in loaded_data:
            actual_shape = loaded_data[key].shape
            if actual_shape != expected_shape:
                raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
        else:
            raise ValueError(f"Missing key in checkpoint: {key}")
    return True


# Example validation
expected_shapes = {
    'weights': (2, 2),
    'bias': (2,)
}

try:
    loaded_data = braintools.file.msgpack_load("files/model_checkpoint.msgpack")
    validate_checkpoint(loaded_data, expected_shapes)
    print("Checkpoint validation passed!")
except ValueError as e:
    print(f"Validation failed: {e}")
```

    Loading checkpoint from model_checkpoint.msgpack
    Checkpoint validation passed!
    


```python
import os
import jax


def training_loop_with_checkpointing():
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Example training parameters
    num_epochs = 100
    checkpoint_interval = 10

    # Mock model state and optimizer state
    model_state = {'weights': jnp.ones((10, 10)), 'bias': jnp.zeros(10)}
    optimizer_state = {'momentum': jnp.zeros((10, 10)), 'step': 0}

    def train_step(model_state, batch):
        # Mock training step
        return model_state

    for epoch in range(num_epochs):
        # Mock batch
        batch = jnp.ones((32, 10))

        # Training step
        model_state = train_step(model_state, batch)

        # Checkpoint every N epochs
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.msgpack"
            checkpoint_data = {
                'model_state': model_state,
                'epoch': epoch,
                'optimizer_state': optimizer_state,
                'rng_state': jax.random.PRNGKey(42)  # In practice, use current RNG state
            }
            braintools.file.msgpack_save(checkpoint_path, checkpoint_data)
            print(f"Checkpoint saved at epoch {epoch}")

        # Break early for demo
        if epoch >= 20:
            break


# Run the training loop
training_loop_with_checkpointing()
```

    Saving checkpoint into checkpoints/checkpoint_epoch_0.msgpack
    Checkpoint saved at epoch 0
    Saving checkpoint into checkpoints/checkpoint_epoch_10.msgpack
    Checkpoint saved at epoch 10
    Saving checkpoint into checkpoints/checkpoint_epoch_20.msgpack
    Checkpoint saved at epoch 20
    

### 4. Regular Checkpointing During Training

## Conclusion

BrainTools' checkpointing system provides a robust and flexible way to save and restore model states. Key features include:

- **Automatic serialization** of JAX arrays, BrainState objects, and BrainUnit quantities
- **Custom object support** through registration system
- **Flexible mismatch handling** for evolving model architectures
- **Asynchronous saving** for large models
- **Type safety** through template-based loading

This system enables reliable model persistence for long training runs, model deployment, and collaborative research workflows.
