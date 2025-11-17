# ComfyUI-UniRig Tests

This directory contains the test suite for ComfyUI-UniRig.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and ComfyUI mocks
├── smoke_test.py            # Basic import and structure tests
├── test_nodes.py            # Unit tests for node classes
├── assets/                  # Test assets (meshes, textures)
└── README.md
```

## Test Types

### Smoke Tests (`smoke_test.py`)
- Tests basic Python imports (torch, numpy, trimesh)
- Tests module structure and imports
- Tests config dataclasses
- **Can run standalone** without pytest

```bash
# Run standalone
python tests/smoke_test.py

# Run with pytest
pytest tests/smoke_test.py -v
```

### Unit Tests (`test_nodes.py`)
- Tests node INPUT_TYPES structure
- Tests RETURN_TYPES definitions
- Tests configuration dataclasses
- Tests parameter validation
- **No model loading required**

```bash
pytest tests/test_nodes.py -v -m unit
```

## Running Tests

### Prerequisites
```bash
pip install -r requirements-dev.txt
```

### Run All Tests
```bash
# From project root
pytest tests/ -v

# With GPU (if available)
pytest tests/ -v --use-gpu
```

### Run Specific Test Categories
```bash
# Smoke tests only
pytest tests/smoke_test.py -v

# Unit tests only
pytest tests/test_nodes.py -v -m unit
```

## Key Features

### 1. ComfyUI Mocking
The `conftest.py` mocks ComfyUI dependencies at module level:
- `folder_paths` - Path management
- `comfy.utils` - Utility functions
- `comfy.model_management` - Device management
- `server.PromptServer` - ComfyUI server

### 2. Pytest Detection in `__init__.py`
The main `__init__.py` detects when pytest is running and skips initialization:
```python
if 'pytest' not in sys.modules:
    # Normal ComfyUI initialization
else:
    # Dummy values for testing
```

### 3. Custom Pytest Options
- `--use-gpu` - Run tests on GPU instead of CPU

### 4. Test Markers
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests with mocks
- `@pytest.mark.real_model` - Tests requiring real models

## CI/CD

GitHub Actions workflow runs on push/PR:
- Smoke tests
- Unit tests
- Mocked integration tests

See `.github/workflows/test-linux.yml`

## Adding New Tests

1. **Unit tests**: Add to `test_nodes.py` with `@pytest.mark.unit`
2. **New node tests**: Follow the pattern in existing test classes
3. **Test assets**: Add to `tests/assets/` directory

## Troubleshooting

### Import Errors
Make sure `conftest.py` is loading before your imports. The mocks are set at module level.

### ComfyUI Dependencies
All ComfyUI modules are mocked. If you need a specific behavior, add it to `conftest.py`.

### Real Model Tests Failing
These require:
- Blender installed
- UniRig models downloaded
- Full UniRig dependencies installed

Skip with: `pytest -m "not real_model"`
