"""
Robotics Memory Manager - Qdrant Cloud Edition

Replaces ChromaDB with Qdrant for production-ready cloud deployment.
Combines token-efficient TOON serialization with Qdrant vector search.

Architecture:
- Qdrant Cloud: Semantic search across conversations, algorithms, and scenes
- TOON Format: Token-efficient serialization for LLM context
- Ollama Embeddings: Local nomic-embed-text model for vectors

Expected Performance:
- 70-85% token reduction vs pure JSON
- 100+ message support (vs current 30-40)
- <500ms overhead for context retrieval
- Cloud-hosted, no local storage needed
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
import uuid
import requests
import os

try:
    from utils.toon_service import get_toon_service
except ImportError:
    from backend.utils.toon_service import get_toon_service


class RoboticsMemoryManager:
    """
    Manages long-term memory for robotics conversations using Qdrant Cloud + TOON format
    """

    def __init__(self):
        """
        Initialize the memory manager with Qdrant Cloud

        Credentials loaded from environment variables:
        - QDRANT_URL: Qdrant cloud endpoint
        - QDRANT_API_KEY: API authentication key
        """
        # Connect to Qdrant Cloud
        self.client = QdrantClient(
            url=os.getenv(
                "QDRANT_URL",
                "https://5c3ca410-23bb-49d9-abde-2e739e8efa84.us-west-1-0.aws.cloud.qdrant.io:6333"
            ),
            api_key=os.getenv(
                "QDRANT_API_KEY",
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.H61ZSxBcr_-Vx8CkJ_B9_7HS4SReUBsShmZTs-QfWvc"
            )
        )

        # Ollama URL for embeddings
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # TOON service for token-efficient serialization
        self.toon_service = get_toon_service()

        # Ensure collections exist
        self._ensure_collections()

        # Get collection stats
        try:
            conv_count = self.client.count(collection_name="conversations").count
            algo_count = self.client.count(collection_name="algorithms").count
            scene_count = self.client.count(collection_name="scenes").count

            print(f"✅ RoboticsMemoryManager initialized (Qdrant Cloud)")
            print(f"   Conversations: {conv_count} entries")
            print(f"   Algorithms: {algo_count} entries")
            print(f"   Scenes: {scene_count} entries")
        except Exception as e:
            print(f"✅ RoboticsMemoryManager initialized (Qdrant Cloud)")
            print(f"   Collections created (empty)")

    def _ensure_collections(self):
        """Create Qdrant collections if they don't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            # Create conversations collection
            if "conversations" not in collection_names:
                self.client.create_collection(
                    collection_name="conversations",
                    vectors_config=VectorParams(
                        size=768,  # nomic-embed-text dimension
                        distance=Distance.COSINE
                    )
                )

            # Create algorithms collection
            if "algorithms" not in collection_names:
                self.client.create_collection(
                    collection_name="algorithms",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

            # Create scenes collection
            if "scenes" not in collection_names:
                self.client.create_collection(
                    collection_name="scenes",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

        except Exception as e:
            print(f"⚠️  Warning: Could not verify collections: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector using Ollama nomic-embed-text model

        Args:
            text: Text to embed

        Returns:
            768-dimensional embedding vector
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text
                },
                timeout=10
            )
            return response.json()["embedding"]
        except Exception as e:
            print(f"⚠️  Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 768

    def add_conversation_message(
        self,
        user_id: str,
        message: str,
        role: str,  # 'user' or 'assistant'
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a conversation message with semantic search capability

        Args:
            user_id: User identifier
            message: Message content
            role: 'user' or 'assistant'
            metadata: Optional metadata (intent, algorithm_type, etc.)

        Returns:
            Message ID
        """
        try:
            # Generate unique message ID (UUID required for Qdrant Cloud)
            timestamp = datetime.now().isoformat()
            message_id = str(uuid.uuid4())

            # Generate embedding
            embedding = self._get_embedding(message)

            # Prepare payload - serialize complex types for Qdrant
            payload = {
                "user_id": user_id,
                "role": role,
                "message": message,
                "timestamp": timestamp,
            }

            # Qdrant accepts JSON in payload, but serialize complex nested structures
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        # Convert complex types (dicts, lists) to JSON strings
                        payload[key] = json.dumps(value)
                    elif isinstance(value, (str, int, float, bool)):
                        payload[key] = value
                    else:
                        # Convert other types to string representation
                        payload[key] = str(value)

            # Store in Qdrant
            self.client.upsert(
                collection_name="conversations",
                points=[
                    PointStruct(
                        id=message_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            print(f"💾 Stored {role} message: {message[:50]}...")
            return message_id

        except Exception as e:
            print(f"❌ Failed to store conversation: {e}")
            return ""

    def add_algorithm(
        self,
        user_id: str,
        algorithm: Dict[str, Any]
    ) -> str:
        """
        Store a generated algorithm with semantic search

        Args:
            user_id: User identifier
            algorithm: Algorithm dict with name, type, code, description, etc.

        Returns:
            Algorithm ID
        """
        try:
            # Use provided ID or generate UUID for Qdrant Cloud compatibility
            algorithm_id = algorithm.get('id', str(uuid.uuid4()))

            # Create searchable text from algorithm
            searchable_text = f"""
Algorithm: {algorithm.get('name', 'Unknown')}
Type: {algorithm.get('type', 'Unknown')}
Description: {algorithm.get('description', '')}
Complexity: {algorithm.get('complexity', 'Unknown')}
Code: {algorithm.get('code', '')[:500]}
""".strip()

            # Generate embedding
            embedding = self._get_embedding(searchable_text)

            # Store payload
            payload = {
                "user_id": user_id,
                "algorithm_id": algorithm_id,
                "name": algorithm.get('name', 'Unknown'),
                "type": algorithm.get('type', 'Unknown'),
                "complexity": algorithm.get('complexity', 'Unknown'),
                "timestamp": datetime.now().isoformat(),
                "code": algorithm.get('code', ''),
                "parameters": json.dumps(algorithm.get('parameters', []))
            }

            # Store in Qdrant
            self.client.upsert(
                collection_name="algorithms",
                points=[
                    PointStruct(
                        id=algorithm_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            print(f"💾 Stored algorithm: {algorithm.get('name')}")
            return algorithm_id

        except Exception as e:
            print(f"❌ Failed to store algorithm: {e}")
            return ""

    def add_scene(
        self,
        user_id: str,
        scene_config: Dict[str, Any]
    ) -> str:
        """
        Store a scene configuration with semantic search

        Args:
            user_id: User identifier
            scene_config: Scene configuration dict

        Returns:
            Scene ID
        """
        try:
            # Generate UUID for Qdrant Cloud compatibility
            scene_id = str(uuid.uuid4())

            # Create searchable text
            searchable_text = self.toon_service.serialize_scene_config(scene_config)

            # Generate embedding
            embedding = self._get_embedding(searchable_text)

            # Store payload
            payload = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "config": json.dumps(scene_config)
            }

            # Store in Qdrant
            self.client.upsert(
                collection_name="scenes",
                points=[
                    PointStruct(
                        id=scene_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            print(f"💾 Stored scene configuration")
            return scene_id

        except Exception as e:
            print(f"❌ Failed to store scene: {e}")
            return ""

    def get_relevant_context(
        self,
        user_id: str,
        current_message: str,
        max_messages: int = 10,
        max_algorithms: int = 3
    ) -> str:
        """
        Get relevant context for LLM using semantic search + recent messages

        This is the KEY method for token reduction:
        - Searches conversation history semantically
        - Returns only relevant past context
        - Formats in TOON for efficiency

        Args:
            user_id: User identifier
            current_message: Current user message
            max_messages: Max conversation messages to retrieve
            max_algorithms: Max relevant algorithms to include

        Returns:
            Token-efficient context string for LLM
        """
        try:
            context_parts = []

            # Generate query embedding
            query_embedding = self._get_embedding(current_message)

            # 1. Get recent conversation history (semantic search)
            recent_messages = self.client.search(
                collection_name="conversations",
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=max_messages
            )

            if recent_messages:
                context_parts.append("## Recent Conversation (TOON Format)")
                context_parts.append(f"messages: {len(recent_messages)}")

                # Format as TOON table
                for hit in recent_messages:
                    role = hit.payload.get('role', 'unknown')
                    timestamp = hit.payload.get('timestamp', '')[:19]
                    message = hit.payload.get('message', '')[:100].replace('\n', ' ')
                    context_parts.append(f"{timestamp} {role} {message}")

                context_parts.append("")

            # 2. Get relevant algorithms
            relevant_algos = self.client.search(
                collection_name="algorithms",
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=max_algorithms
            )

            if relevant_algos:
                context_parts.append("## Relevant Algorithms")
                for hit in relevant_algos:
                    context_parts.append(f"- {hit.payload.get('name')} ({hit.payload.get('type')})")
                    context_parts.append(f"  Complexity: {hit.payload.get('complexity')}")
                context_parts.append("")

            # 3. Get relevant scenes
            relevant_scenes = self.client.search(
                collection_name="scenes",
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=2
            )

            if relevant_scenes:
                context_parts.append("## Recent Scene Configurations")
                context_parts.append(f"scenes: {len(relevant_scenes)}")
                context_parts.append("")

            # Combine all context
            full_context = "\n".join(context_parts)

            # Log token savings
            json_equivalent = json.dumps({
                "messages": [msg.payload for msg in recent_messages],
                "algorithms": [algo.payload for algo in relevant_algos],
                "scenes": [scene.payload for scene in relevant_scenes]
            }, indent=2)

            savings = self.toon_service.estimate_token_savings(json_equivalent, full_context)
            print(f"📊 Context prepared: {savings['toon_tokens']} tokens ({savings['savings_percent']}% savings)")

            return full_context

        except Exception as e:
            print(f"❌ Failed to get context: {e}")
            return ""

    def get_conversation_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get full conversation history for a user (for UI display)

        Args:
            user_id: User identifier
            limit: Max messages to retrieve

        Returns:
            List of message dicts with role, content, timestamp
        """
        try:
            results = self.client.scroll(
                collection_name="conversations",
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=limit
            )

            messages = []
            if results and results[0]:
                for point in results[0]:
                    messages.append({
                        "role": point.payload.get('role', 'unknown'),
                        "content": point.payload.get('message', ''),
                        "timestamp": point.payload.get('timestamp', ''),
                        "metadata": point.payload
                    })

            # Sort by timestamp
            messages.sort(key=lambda x: x['timestamp'])
            return messages

        except Exception as e:
            print(f"❌ Failed to get conversation history: {e}")
            return []

    def get_user_algorithms(
        self,
        user_id: str,
        algorithm_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all algorithms for a user

        Args:
            user_id: User identifier
            algorithm_type: Optional filter by algorithm type

        Returns:
            List of algorithm dicts
        """
        try:
            filter_conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            if algorithm_type:
                filter_conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=algorithm_type))
                )

            results = self.client.scroll(
                collection_name="algorithms",
                scroll_filter=Filter(must=filter_conditions),
                limit=100
            )

            algorithms = []
            if results and results[0]:
                for point in results[0]:
                    algorithms.append({
                        "id": point.payload.get('algorithm_id', ''),
                        "name": point.payload.get('name', 'Unknown'),
                        "type": point.payload.get('type', 'Unknown'),
                        "complexity": point.payload.get('complexity', 'Unknown'),
                        "code": point.payload.get('code', ''),
                        "parameters": json.loads(point.payload.get('parameters', '[]')),
                        "timestamp": point.payload.get('timestamp', '')
                    })

            return algorithms

        except Exception as e:
            print(f"❌ Failed to get algorithms: {e}")
            return []

    def clear_user_data(self, user_id: str):
        """
        Clear all data for a user (GDPR compliance)

        Args:
            user_id: User identifier
        """
        try:
            # Delete from all collections
            for collection_name in ["conversations", "algorithms", "scenes"]:
                # Scroll to get all user points
                results = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                    ),
                    limit=1000
                )

                if results and results[0]:
                    point_ids = [point.id for point in results[0]]
                    if point_ids:
                        self.client.delete(
                            collection_name=collection_name,
                            points_selector=point_ids
                        )

            print(f"🗑️  Cleared all data for user: {user_id}")

        except Exception as e:
            print(f"❌ Failed to clear user data: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        try:
            return {
                "conversations": self.client.count(collection_name="conversations").count,
                "algorithms": self.client.count(collection_name="algorithms").count,
                "scenes": self.client.count(collection_name="scenes").count
            }
        except Exception as e:
            print(f"⚠️  Failed to get stats: {e}")
            return {
                "conversations": 0,
                "algorithms": 0,
                "scenes": 0
            }


# Singleton instance
_memory_manager: Optional[RoboticsMemoryManager] = None


def get_memory_manager() -> RoboticsMemoryManager:
    """Get or create RoboticsMemoryManager singleton"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = RoboticsMemoryManager()
    return _memory_manager
