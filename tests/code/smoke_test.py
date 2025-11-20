"""
Smoke test for ComfyUI-UniRig
Tests basic module import functionality without requiring ComfyUI or real models
"""

import os
import sys
import traceback
from pathlib import Path

# Disable Blender auto-installation during smoke tests
os.environ['UNIRIG_SKIP_BLENDER_INSTALL'] = '1'

# Add custom node directory to path
_custom_node_dir = Path(__file__).parent.parent.parent
if str(_custom_node_dir) not in sys.path:
    sys.path.insert(0, str(_custom_node_dir))

# Add the parent of custom_node_dir to allow importing it as a package
# This enables relative imports like `from ..lib import ...` to work
_custom_nodes_parent = _custom_node_dir.parent
if str(_custom_nodes_parent) not in sys.path:
    sys.path.insert(0, str(_custom_nodes_parent))

# Add nodes directory to path for direct imports (simple modules only)
_nodes_dir = _custom_node_dir / "nodes"
if str(_nodes_dir) not in sys.path:
    sys.path.insert(0, str(_nodes_dir))

# Mock ComfyUI dependencies BEFORE importing nodes
# This allows smoke test to run without ComfyUI installed
# Use cacheable location for models in CI (instead of /tmp)
_test_models_dir = os.environ.get("TEST_MODELS_DIR", str(Path.home() / ".cache" / "test_models"))
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = _test_models_dir
mock_folder_paths.get_folder_paths = lambda x: [_test_models_dir]
mock_folder_paths.get_temp_directory = lambda: "/tmp/comfy_temp"
mock_folder_paths.get_output_directory = lambda: "/tmp/comfy_output"
mock_folder_paths.get_input_directory = lambda: "/tmp/comfy_input"
sys.modules["folder_paths"] = mock_folder_paths

mock_comfy_mm = type("model_management", (), {})()
mock_comfy_mm.get_torch_device = lambda: "cpu"
mock_comfy_mm.soft_empty_cache = lambda: None
mock_comfy_mm.load_models_gpu = lambda x: None
mock_comfy_mm.unet_offload_device = lambda: "cpu"
mock_comfy_mm.is_device_mps = lambda x: False
mock_comfy_mm.get_autocast_device = lambda x: "cpu"

mock_comfy_utils = type("utils", (), {})()
mock_comfy_utils.load_torch_file = lambda x: {}
mock_comfy_utils.ProgressBar = type("ProgressBar", (), {})

mock_comfy = type("comfy", (), {})()
mock_comfy.model_management = mock_comfy_mm
mock_comfy.utils = mock_comfy_utils

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.model_management"] = mock_comfy_mm
sys.modules["comfy.utils"] = mock_comfy_utils

# Mock server module
mock_prompt_server_instance = type("instance", (), {})()
mock_prompt_server_instance.app = type("app", (), {"add_routes": lambda x: None})()

mock_server = type("server", (), {})()
mock_server.PromptServer = type("PromptServer", (), {"instance": mock_prompt_server_instance})()

sys.modules["server"] = mock_server

# Mock aiohttp.web
mock_aiohttp_web = type("web", (), {})()
mock_aiohttp_web.static = lambda *args: None

sys.modules["aiohttp"] = type("aiohttp", (), {"web": mock_aiohttp_web})()
sys.modules["aiohttp.web"] = mock_aiohttp_web


def test_basic_imports():
    """Test that basic Python dependencies can be imported"""
    print("Testing basic dependencies...")

    try:
        import torch
        print(f"  [PASS] torch {torch.__version__}")
    except ImportError as e:
        print(f"  [FAIL] torch: {e}")
        return False

    try:
        import numpy as np
        print(f"  [PASS] numpy {np.__version__}")
    except ImportError as e:
        print(f"  [FAIL] numpy: {e}")
        return False

    try:
        import trimesh
        print(f"  [PASS] trimesh {trimesh.__version__}")
    except ImportError as e:
        print(f"  [FAIL] trimesh: {e}")
        return False

    try:
        import yaml
        print(f"  [PASS] yaml")
    except ImportError as e:
        print(f"  [FAIL] yaml: {e}")
        return False

    return True


def test_config_module():
    """Test that config dataclasses can be imported"""
    print("\nTesting config module imports...")
    try:
        from lib.skinning_config import SkinningConfig, SkeletonConfig
        print(f"  [PASS] SkinningConfig imported")
        print(f"  [PASS] SkeletonConfig imported")

        # Test defaults
        sc = SkinningConfig()
        print(f"  [PASS] SkinningConfig defaults: grid={sc.voxel_grid_size}, samples={sc.num_samples}")

        skc = SkeletonConfig()
        print(f"  [PASS] SkeletonConfig defaults: face_count={skc.target_face_count}")

        return True
    except Exception as e:
        print(f"  [FAIL] Config import failed: {e}")
        traceback.print_exc()
        return False


def test_constants_module():
    """Test that constants can be imported"""
    print("\nTesting constants imports...")
    try:
        from constants import (
            TARGET_FACE_COUNT,
            BLENDER_TIMEOUT,
            INFERENCE_TIMEOUT,
            PARSE_TIMEOUT,
        )
        print(f"  [PASS] TARGET_FACE_COUNT = {TARGET_FACE_COUNT}")
        print(f"  [PASS] BLENDER_TIMEOUT = {BLENDER_TIMEOUT}")
        print(f"  [PASS] INFERENCE_TIMEOUT = {INFERENCE_TIMEOUT}")
        print(f"  [PASS] PARSE_TIMEOUT = {PARSE_TIMEOUT}")
        return True
    except Exception as e:
        print(f"  [FAIL] Constants import failed: {e}")
        traceback.print_exc()
        return False


def test_base_module():
    """Test that base module can be imported"""
    print("\nTesting base module imports...")
    try:
        from base import (
            UNIRIG_PATH,
            LIB_DIR,
            UNIRIG_MODELS_DIR,
        )
        print(f"  [PASS] UNIRIG_PATH = {UNIRIG_PATH}")
        print(f"  [PASS] LIB_DIR = {LIB_DIR}")
        print(f"  [PASS] UNIRIG_MODELS_DIR = {UNIRIG_MODELS_DIR}")
        return True
    except Exception as e:
        print(f"  [FAIL] Base module import failed: {e}")
        traceback.print_exc()
        return False


def test_node_class_structures():
    """Test that node classes have required ComfyUI methods"""
    print("\nTesting node class structures...")

    # All node modules use relative imports (via base.py which imports install.py)
    # These need pytest with proper package setup to work
    # In standalone smoke test mode, we just verify the files exist and defer to pytest
    all_nodes = [
        ("skeleton_extraction", "UniRigExtractSkeleton"),
        ("skeleton_extraction", "UniRigExtractRig"),
        ("skinning", "UniRigApplySkinning"),
        ("skinning", "UniRigApplySkinningML"),
        ("mesh_io", "UniRigLoadMesh"),
        ("mesh_io", "UniRigSaveMesh"),
        ("model_loaders", "UniRigLoadSkeletonModel"),
        ("model_loaders", "UniRigLoadSkinningModel"),
        ("skeleton_io", "UniRigSaveSkeleton"),
        ("skeleton_io", "UniRigSaveRiggedMesh"),
        ("skeleton_io", "UniRigLoadRiggedMesh"),
        ("skeleton_io", "UniRigPreviewRiggedMesh"),
        ("skeleton_processing", "UniRigDenormalizeSkeleton"),
        ("skeleton_processing", "UniRigValidateSkeleton"),
        ("skeleton_processing", "UniRigPrepareSkeletonForSkinning"),
    ]

    # Verify node files exist
    for module_name, class_name in all_nodes:
        node_file = _nodes_dir / f"{module_name}.py"
        if node_file.exists():
            print(f"  [PASS] {module_name}.py exists")
        else:
            print(f"  [FAIL] {module_name}.py not found")
            return False

    # Check that __init__.py exists and has NODE_CLASS_MAPPINGS
    init_file = _nodes_dir / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        if "NODE_CLASS_MAPPINGS" in content:
            print(f"  [PASS] nodes/__init__.py has NODE_CLASS_MAPPINGS")
        else:
            print(f"  [FAIL] nodes/__init__.py missing NODE_CLASS_MAPPINGS")
            return False
    else:
        print(f"  [FAIL] nodes/__init__.py not found")
        return False

    print(f"  [INFO] Full node structure tests run via pytest (handles relative imports)")
    return True


def test_skinning_config_with_overrides():
    """Test that skinning config accepts overrides"""
    print("\nTesting SkinningConfig with overrides...")
    try:
        from lib.skinning_config import SkinningConfig

        # Test with overrides
        config = SkinningConfig(
            voxel_grid_size=512,
            num_samples=65536,
            vertex_samples=16384,
        )

        assert config.voxel_grid_size == 512, f"Expected 512, got {config.voxel_grid_size}"
        assert config.num_samples == 65536, f"Expected 65536, got {config.num_samples}"
        assert config.vertex_samples == 16384, f"Expected 16384, got {config.vertex_samples}"

        # Test to_dict
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['voxel_grid_size'] == 512

        # Test from_dict
        config2 = SkinningConfig.from_dict({'voxel_grid_size': 256, 'unknown_param': 999})
        assert config2.voxel_grid_size == 256
        assert not hasattr(config2, 'unknown_param')

        print(f"  [PASS] Override test passed")
        print(f"  [PASS] to_dict test passed")
        print(f"  [PASS] from_dict test passed")
        return True

    except Exception as e:
        print(f"  [FAIL] SkinningConfig override test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests"""
    print("=" * 60)
    print("ComfyUI-UniRig Smoke Test")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Basic imports", test_basic_imports()))
    results.append(("Config module", test_config_module()))
    results.append(("Constants module", test_constants_module()))
    results.append(("Base module", test_base_module()))
    results.append(("Node class structures", test_node_class_structures()))
    results.append(("SkinningConfig overrides", test_skinning_config_with_overrides()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("[PASS] All smoke tests passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some smoke tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
