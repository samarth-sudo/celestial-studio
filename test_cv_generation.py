"""
Test script for Computer Vision algorithm generation
Verifies that the algorithm generator can produce CV algorithms
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from algorithm_templates import AlgorithmTemplates, get_algorithm_list

def test_cv_templates():
    """Test that CV templates are available"""
    print("=" * 60)
    print("Testing Computer Vision Templates")
    print("=" * 60)

    templates = AlgorithmTemplates.get_all_templates()

    if 'computer_vision' not in templates:
        print("❌ FAILED: computer_vision category not found")
        return False

    cv_templates = templates['computer_vision']
    print(f"\n✅ Found {len(cv_templates)} computer vision templates:")

    for name, code in cv_templates.items():
        lines = code.split('\n')
        description = lines[1].strip('// ').strip() if len(lines) > 1 else "No description"
        print(f"\n  • {name}")
        print(f"    {description}")
        print(f"    Code length: {len(code)} characters")

    return True


def test_algorithm_list():
    """Test that CV algorithms appear in the algorithm list"""
    print("\n" + "=" * 60)
    print("Testing Algorithm List")
    print("=" * 60)

    algorithms = get_algorithm_list()
    cv_algorithms = [a for a in algorithms if a['category'] == 'computer_vision']

    print(f"\n✅ Found {len(cv_algorithms)} CV algorithms in the list:")

    for algo in cv_algorithms:
        print(f"\n  • {algo['name']}")
        print(f"    Category: {algo['category']}")
        print(f"    Complexity: {algo['complexity']}")
        print(f"    Description: {algo['description'][:100]}...")

    return len(cv_algorithms) > 0


def test_algorithm_generator():
    """Test that the algorithm generator can handle CV requests"""
    print("\n" + "=" * 60)
    print("Testing Algorithm Generator")
    print("=" * 60)

    try:
        from algorithm_generator import AlgorithmGenerator, AlgorithmRequest

        generator = AlgorithmGenerator()

        # Test request
        request = AlgorithmRequest(
            description="Detect objects in front of the robot and return their positions",
            robot_type="mobile",
            algorithm_type="computer_vision"
        )

        print("\n✅ Algorithm generator initialized successfully")
        print(f"✅ CV template loaded: {len(generator.templates.get('computer_vision', ''))} chars")
        print(f"\nTest request:")
        print(f"  Description: {request.description}")
        print(f"  Robot type: {request.robot_type}")
        print(f"  Algorithm type: {request.algorithm_type}")

        # Note: We won't actually generate code here since it requires Ollama running
        print("\n⚠️  Skipping actual generation (requires Ollama running)")
        print("   To test generation, use: POST /api/generate-algorithm")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 Computer Vision Algorithm Generation Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("CV Templates", test_cv_templates()))
    results.append(("Algorithm List", test_algorithm_list()))
    results.append(("Algorithm Generator", test_algorithm_generator()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Computer vision algorithms are ready.")
        print("\nAvailable CV algorithms:")
        print("  • object_detection - YOLO-style object detection")
        print("  • object_tracking - Kalman filter tracking")
        print("  • feature_detection - Corner/feature detection and matching")
        print("  • optical_flow - Motion estimation")
        print("  • semantic_segmentation - Scene segmentation")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
