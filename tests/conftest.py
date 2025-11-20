"""
Pytest configuration and fixtures for ComfyUI-UniRig tests
"""
import sys
import os
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock, Mock


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Run tests on GPU instead of CPU (much faster for real model tests)"
    )


# Add the custom node directory to Python path so we can import modules
custom_nodes_dir = Path(__file__).parent.parent
sys.path.insert(0, str(custom_nodes_dir))

# Mock ComfyUI modules at module level BEFORE pytest starts
# This prevents import errors when pytest tries to load __init__.py files

# Mock folder_paths module (ComfyUI path management)
# Use cacheable location for models in CI (instead of /tmp)
test_models_dir = os.environ.get("TEST_MODELS_DIR", str(Path.home() / ".cache" / "test_models"))
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = test_models_dir
mock_folder_paths.get_folder_paths = lambda x: [test_models_dir]
mock_folder_paths.get_temp_directory = lambda: "/tmp/comfy_temp"
mock_folder_paths.get_output_directory = lambda: "/tmp/comfy_output"
mock_folder_paths.get_input_directory = lambda: "/tmp/comfy_input"
sys.modules["folder_paths"] = mock_folder_paths

# Mock comfy modules
mock_comfy = type("comfy", (), {})()
mock_comfy_utils = type("utils", (), {})()
mock_comfy_utils.load_torch_file = lambda x: {}
mock_comfy_utils.ProgressBar = MagicMock()
mock_comfy.utils = mock_comfy_utils

# Mock model_management
mock_comfy_mm = type("model_management", (), {})()


def _get_test_device():
    """Get device for testing - GPU if --use-gpu flag is set, else CPU"""
    use_gpu = os.environ.get("PYTEST_USE_GPU", "0") == "1"
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


mock_comfy_mm.get_torch_device = _get_test_device
mock_comfy_mm.soft_empty_cache = lambda: None
mock_comfy_mm.load_models_gpu = lambda x: None
mock_comfy_mm.unet_offload_device = lambda: torch.device("cpu")
mock_comfy_mm.is_device_mps = lambda x: False
mock_comfy_mm.get_autocast_device = lambda x: "cpu"
mock_comfy.model_management = mock_comfy_mm

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = mock_comfy_utils
sys.modules["comfy.model_management"] = mock_comfy_mm

# Mock server module (ComfyUI server)
mock_prompt_server_instance = MagicMock()
mock_prompt_server_instance.app = MagicMock()
mock_prompt_server_instance.app.add_routes = MagicMock()

mock_prompt_server = type("PromptServer", (), {})()
mock_prompt_server.instance = mock_prompt_server_instance

mock_server = type("server", (), {})()
mock_server.PromptServer = type("PromptServer", (), {"instance": mock_prompt_server_instance})()

sys.modules["server"] = mock_server

# Mock aiohttp.web
mock_aiohttp_web = type("web", (), {})()
mock_aiohttp_web.static = lambda *args: None

mock_aiohttp = type("aiohttp", (), {})()
mock_aiohttp.web = mock_aiohttp_web

sys.modules["aiohttp"] = mock_aiohttp
sys.modules["aiohttp.web"] = mock_aiohttp_web


def pytest_ignore_collect(collection_path, path, config):
    """Ignore __init__.py files during collection"""
    if collection_path.name == "__init__.py":
        return True
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_test_device(request):
    """Configure test device based on --use-gpu flag"""
    use_gpu = request.config.getoption("--use-gpu")
    if use_gpu:
        os.environ["PYTEST_USE_GPU"] = "1"
        if torch.cuda.is_available():
            print(f"\n[GPU] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n[WARN] --use-gpu specified but CUDA not available, using CPU")
    else:
        os.environ["PYTEST_USE_GPU"] = "0"
        print("\n[CPU] Using CPU (use --use-gpu for GPU acceleration)")

    yield

    # Cleanup
    os.environ.pop("PYTEST_USE_GPU", None)


@pytest.fixture(scope="session", autouse=True)
def setup_mock_comfy():
    """Set up mock ComfyUI modules for testing - runs once per session"""
    # Ensure mocks persist throughout test session
    return True


@pytest.fixture
def mock_comfy_environment():
    """Provide access to mocked ComfyUI environment (already set up at module level)"""
    return sys.modules["folder_paths"]


@pytest.fixture
def sample_mesh():
    """Create a simple test mesh (cube)"""
    try:
        import trimesh
        import numpy as np

        # Create a simple cube mesh
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ], dtype=np.int32)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    except ImportError:
        pytest.skip("trimesh not installed")


@pytest.fixture
def sample_skeleton():
    """Create a simple test skeleton dict"""
    import numpy as np

    return {
        "joints": np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32),
        "names": ["root", "spine", "head"],
        "parents": [None, 0, 1],
        "tails": np.array([[0, 0.5, 0], [0, 1.5, 0], [0, 2.5, 0]], dtype=np.float32),
        "mesh_vertices": np.random.randn(100, 3).astype(np.float32),
        "mesh_faces": np.random.randint(0, 100, (50, 3)).astype(np.int32),
        "mesh_vertex_normals": np.random.randn(100, 3).astype(np.float32),
        "mesh_face_normals": np.random.randn(50, 3).astype(np.float32),
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_directories():
    """Ensure test directories exist"""
    import tempfile

    # Create temporary directories for tests
    # Use cacheable location for models (not /tmp)
    test_models_dir = os.environ.get("TEST_MODELS_DIR", str(Path.home() / ".cache" / "test_models"))
    temp_dirs = [
        test_models_dir,
        "/tmp/comfy_temp",
        "/tmp/comfy_output",
        "/tmp/comfy_input"
    ]

    for dir_path in temp_dirs:
        os.makedirs(dir_path, exist_ok=True)

    yield

    # Note: We don't clean up to allow inspection of test outputs
    # GitHub Actions will clean up automatically


@pytest.fixture
def cpu_optimized_params():
    """Provide CPU-optimized parameters for integration tests"""
    return {
        "target_face_count": 10000,   # Reduced from 50000
        "voxel_grid_size": 128,       # Reduced from 196
        "num_samples": 16384,         # Reduced from 32768
        "vertex_samples": 4096,       # Reduced from 8192
        "cache_to_gpu": False,        # Force CPU
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no model loading)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with mocked models"
    )
    config.addinivalue_line(
        "markers", "real_model: Tests that download and use real models (slow)"
    )
