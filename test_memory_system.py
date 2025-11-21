"""
Test script for TOON + ChromaDB hybrid memory system

Tests:
1. ChromaDB initialization
2. Conversation message storage
3. Algorithm storage
4. Scene storage
5. Context retrieval with TOON format
6. Token savings measurement
"""

import sys
sys.path.append('backend')

from backend.chat.memory_manager import get_memory_manager
from backend.utils.toon_service import get_toon_service
import json


def test_memory_system():
    print("=" * 60)
    print("Testing TOON + ChromaDB Hybrid Memory System")
    print("=" * 60)

    # Initialize managers
    print("\n1. Initializing memory manager...")
    memory = get_memory_manager()
    toon = get_toon_service()

    # Test user
    user_id = "test_user_123"

    # Test 1: Store conversation messages
    print("\n2. Testing conversation message storage...")
    messages = [
        "I want to create a mobile robot for warehouse navigation",
        "It should avoid obstacles using DWA algorithm",
        "Add path planning with A* algorithm",
        "The robot should operate in a 10x10 meter warehouse",
        "There should be 5 obstacles in the scene"
    ]

    for i, msg in enumerate(messages):
        memory.add_conversation_message(
            user_id=user_id,
            message=msg,
            role="user",
            metadata={
                "intent": "simulation_request",
                "timestamp_index": i
            }
        )

    print(f"✅ Stored {len(messages)} conversation messages")

    # Test 2: Store algorithm
    print("\n3. Testing algorithm storage...")
    test_algorithm = {
        "id": "algo-123",
        "name": "A* Path Planning",
        "type": "path_planning",
        "description": "Find shortest path using A* algorithm with heuristic optimization",
        "code": """
function findPath(start, goal, obstacles) {
  const openSet = [start]
  const closedSet = new Set()

  while (openSet.length > 0) {
    const current = openSet[0]
    if (current === goal) return reconstructPath(current)

    // A* logic here
  }
  return []
}
""",
        "complexity": "O(n log n)",
        "parameters": [
            {"name": "maxIterations", "type": "number", "value": 1000},
            {"name": "heuristicWeight", "type": "number", "value": 1.2}
        ]
    }

    memory.add_algorithm(user_id=user_id, algorithm=test_algorithm)
    print("✅ Stored algorithm")

    # Test 3: Store scene
    print("\n4. Testing scene storage...")
    test_scene = {
        "robot": {
            "type": "mobile_robot",
            "position": [0, 0.5, 0]
        },
        "environment": {
            "floor": {"texture": "warehouse", "size": [10, 10]},
            "lighting": {"type": "directional"}
        },
        "objects": [
            {"type": "box", "position": [2, 0.5, 2], "color": "red"},
            {"type": "box", "position": [5, 0.5, 3], "color": "blue"},
            {"type": "box", "position": [7, 0.5, 7], "color": "green"}
        ]
    }

    memory.add_scene(user_id=user_id, scene_config=test_scene)
    print("✅ Stored scene configuration")

    # Test 4: Retrieve relevant context
    print("\n5. Testing context retrieval with semantic search...")
    current_query = "Show me the path planning algorithm for obstacle avoidance"

    context = memory.get_relevant_context(
        user_id=user_id,
        current_message=current_query,
        max_messages=5,
        max_algorithms=2
    )

    print("\nRetrieved Context (TOON Format):")
    print("-" * 60)
    print(context)
    print("-" * 60)

    # Test 5: Test TOON serialization for conversation
    print("\n6. Testing TOON conversation serialization...")
    conversation_messages = [
        {
            "role": "user",
            "content": "Create a mobile robot",
            "timestamp": "2025-11-17T10:30:00"
        },
        {
            "role": "assistant",
            "content": "I'll create a mobile robot for you. What environment should it operate in?",
            "timestamp": "2025-11-17T10:30:05"
        },
        {
            "role": "user",
            "content": "A warehouse environment with obstacles",
            "timestamp": "2025-11-17T10:30:15"
        }
    ]

    toon_conversation = toon.serialize_conversation(conversation_messages)
    json_conversation = json.dumps(conversation_messages, indent=2)

    savings = toon.estimate_token_savings(json_conversation, toon_conversation)

    print(f"\nTOON Conversation Format:")
    print("-" * 60)
    print(toon_conversation)
    print("-" * 60)
    print(f"\nToken Savings: {savings['savings_percent']}%")
    print(f"JSON tokens: {savings['json_tokens']}")
    print(f"TOON tokens: {savings['toon_tokens']}")

    # Test 6: Get stats
    print("\n7. Memory Manager Statistics:")
    print("-" * 60)
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("-" * 60)

    # Test 7: Retrieve user's algorithms
    print("\n8. Testing algorithm retrieval...")
    algorithms = memory.get_user_algorithms(user_id=user_id)
    print(f"Found {len(algorithms)} algorithms for user")
    for algo in algorithms:
        print(f"  - {algo['name']} ({algo['type']})")

    # Test 8: Get conversation history
    print("\n9. Testing conversation history retrieval...")
    history = memory.get_conversation_history(user_id=user_id, limit=10)
    print(f"Found {len(history)} messages in conversation history")
    for msg in history[:3]:  # Show first 3
        print(f"  [{msg['role']}] {msg['content'][:50]}...")

    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)

    # Clean up test data
    print("\n10. Cleaning up test data...")
    memory.clear_user_data(user_id)
    print("✅ Test data cleared")


if __name__ == "__main__":
    try:
        test_memory_system()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
