#!/usr/bin/env python3
"""
Test Qdrant Cloud Connection

Verifies:
1. Connection to Qdrant Cloud cluster
2. Collection creation and listing
3. Embedding generation via Ollama
4. Vector insertion and search
"""

import os
import sys
import requests
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Load environment from .env file
try:
    from dotenv import load_dotenv
    load_dotenv('backend/.env')
except ImportError:
    print("⚠️  python-dotenv not installed, using environment variables only")

def test_qdrant_connection():
    """Test basic Qdrant Cloud connectivity"""
    print("\n" + "="*60)
    print("🧪 Testing Qdrant Cloud Connection")
    print("="*60 + "\n")

    # Get credentials
    qdrant_url = os.getenv(
        "QDRANT_URL",
        "https://5c3ca410-23bb-49d9-abde-2e739e8efa84.us-west-1-0.aws.cloud.qdrant.io:6333"
    )
    qdrant_api_key = os.getenv(
        "QDRANT_API_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.H61ZSxBcr_-Vx8CkJ_B9_7HS4SReUBsShmZTs-QfWvc"
    )

    print(f"📍 Qdrant URL: {qdrant_url}")
    print(f"🔑 API Key: {qdrant_api_key[:20]}..." + "\n")

    try:
        # Initialize client
        print("🔌 Connecting to Qdrant Cloud...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Test connection by listing collections
        print("📋 Listing existing collections...")
        collections = client.get_collections().collections
        print(f"✅ Connected! Found {len(collections)} collections:")
        for col in collections:
            # Get detailed collection info
            try:
                col_info = client.get_collection(col.name)
                count = col_info.points_count if hasattr(col_info, 'points_count') else 'unknown'
                print(f"   - {col.name} ({count} vectors)")
            except:
                print(f"   - {col.name}")

        return client, True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None, False


def test_ollama_embeddings():
    """Test Ollama embedding generation"""
    print("\n" + "="*60)
    print("🧪 Testing Ollama Embeddings")
    print("="*60 + "\n")

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"📍 Ollama URL: {ollama_url}\n")

    try:
        print("🤖 Generating embedding for test text...")
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "Test robotics simulation query"
            },
            timeout=10
        )

        if response.status_code == 200:
            embedding = response.json()["embedding"]
            print(f"✅ Embedding generated successfully!")
            print(f"   Dimension: {len(embedding)}")
            print(f"   Sample values: {embedding[:5]}")
            return embedding, True
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return None, False

    except Exception as e:
        print(f"❌ Ollama request failed: {e}")
        return None, False


def test_qdrant_operations(client, test_embedding):
    """Test Qdrant vector operations"""
    print("\n" + "="*60)
    print("🧪 Testing Qdrant Vector Operations")
    print("="*60 + "\n")

    test_collection = "test_deployment"

    try:
        # Create test collection
        print(f"📦 Creating test collection '{test_collection}'...")

        # Delete if exists
        try:
            client.delete_collection(collection_name=test_collection)
            print("   (Deleted existing test collection)")
        except:
            pass

        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print("✅ Collection created!\n")

        # Insert test point
        print("💾 Inserting test vector...")
        test_id = str(uuid.uuid4())  # Use UUID for Qdrant Cloud compatibility
        client.upsert(
            collection_name=test_collection,
            points=[
                PointStruct(
                    id=test_id,
                    vector=test_embedding,
                    payload={
                        "text": "Test robotics simulation query",
                        "timestamp": "2025-11-18T00:00:00",
                        "type": "test"
                    }
                )
            ]
        )
        print("✅ Vector inserted!\n")

        # Search test
        print("🔍 Performing similarity search...")
        results = client.search(
            collection_name=test_collection,
            query_vector=test_embedding,
            limit=1
        )

        if results and len(results) > 0:
            print(f"✅ Search successful!")
            print(f"   Found {len(results)} results")
            print(f"   Top match score: {results[0].score:.4f}")
            print(f"   Payload: {results[0].payload}")
        else:
            print("⚠️  Search returned no results")

        # Clean up
        print("\n🗑️  Cleaning up test collection...")
        client.delete_collection(collection_name=test_collection)
        print("✅ Test collection deleted")

        return True

    except Exception as e:
        print(f"❌ Vector operations failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("      QDRANT CLOUD DEPLOYMENT TEST SUITE")
    print("🚀 " * 20)

    # Test 1: Qdrant Connection
    client, conn_success = test_qdrant_connection()
    if not conn_success:
        print("\n❌ FAILED: Cannot proceed without Qdrant connection")
        sys.exit(1)

    # Test 2: Ollama Embeddings
    embedding, ollama_success = test_ollama_embeddings()
    if not ollama_success:
        print("\n⚠️  WARNING: Ollama embeddings failed")
        print("   This will need to be fixed before deployment")
        print("   Skipping vector operation tests...")
        sys.exit(1)

    # Test 3: Qdrant Operations
    ops_success = test_qdrant_operations(client, embedding)

    # Final summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Qdrant Connection: {'PASSED' if conn_success else 'FAILED'}")
    print(f"{'✅' if ollama_success else '❌'} Ollama Embeddings: {'PASSED' if ollama_success else 'FAILED'}")
    print(f"{'✅' if ops_success else '❌'} Vector Operations: {'PASSED' if ops_success else 'FAILED'}")
    print("="*60)

    if conn_success and ollama_success and ops_success:
        print("\n🎉 ALL TESTS PASSED! Ready for Vercel deployment.")
        print("\nNext steps:")
        print("  1. Set up ngrok tunnel for Ollama")
        print("  2. Fix hardcoded URLs in frontend")
        print("  3. Create vercel.json and deploy")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before deploying.")
        return 1


if __name__ == "__main__":
    exit(main())
