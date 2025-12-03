"""
Genesis Documentation Indexer
Parses Genesis documentation and extracts code examples and API references.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GenesisDocIndexer:
    """Index Genesis documentation for RAG retrieval"""

    def __init__(self, docs_path: str = "./genesis-doc"):
        self.docs_path = Path(docs_path)
        self.indexed_docs: List[Dict] = []
        self.code_examples: List[Dict] = []
        self.api_references: Dict[str, str] = {}

    def index_documentation(self) -> None:
        """Parse all markdown files and extract relevant information"""
        if not self.docs_path.exists():
            logger.warning(f"Genesis docs not found at {self.docs_path}")
            return

        logger.info(f"Indexing Genesis documentation from {self.docs_path}")

        # Find all markdown files
        md_files = list(self.docs_path.rglob("*.md"))
        logger.info(f"Found {len(md_files)} documentation files")

        for md_file in md_files:
            self._index_file(md_file)

        logger.info(f"Indexed {len(self.code_examples)} code examples")
        logger.info(f"Indexed {len(self.api_references)} API references")

    def _index_file(self, file_path: Path) -> None:
        """Index a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract metadata
            relative_path = file_path.relative_to(self.docs_path)
            category = str(relative_path.parts[0]) if relative_path.parts else "general"

            # Store full document
            self.indexed_docs.append({
                'path': str(relative_path),
                'category': category,
                'content': content,
                'title': self._extract_title(content)
            })

            # Extract code examples
            code_blocks = self._extract_code_blocks(content)
            for code in code_blocks:
                self.code_examples.append({
                    'code': code,
                    'file': str(relative_path),
                    'category': category
                })

            # Extract API references (class/function definitions)
            apis = self._extract_api_references(content)
            for api_name, api_doc in apis.items():
                self.api_references[api_name] = api_doc

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")

    def _extract_title(self, content: str) -> str:
        """Extract title from markdown content"""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1) if match else "Untitled"

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract Python code blocks from markdown"""
        # Match ```python ... ``` code blocks
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches

    def _extract_api_references(self, content: str) -> Dict[str, str]:
        """Extract API function/class references"""
        apis = {}

        # Look for function/class definitions in code blocks
        code_blocks = self._extract_code_blocks(content)
        for code in code_blocks:
            # Extract function calls like gs.Scene(), gs.init(), etc.
            function_calls = re.findall(r'(gs\.\w+(?:\.\w+)*)', code)
            for func in function_calls:
                if func not in apis:
                    apis[func] = code[:500]  # Store context (first 500 chars)

        return apis

    def search_by_keyword(self, keyword: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword search across all documents"""
        keyword_lower = keyword.lower()
        results = []

        for doc in self.indexed_docs:
            # Calculate relevance score (simple count)
            content_lower = doc['content'].lower()
            score = content_lower.count(keyword_lower)

            if score > 0:
                results.append({
                    **doc,
                    'score': score
                })

        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def search_code_examples(self, keyword: str, top_k: int = 5) -> List[str]:
        """Search for code examples containing keyword"""
        keyword_lower = keyword.lower()
        results = []

        for example in self.code_examples:
            if keyword_lower in example['code'].lower():
                results.append(example['code'])
                if len(results) >= top_k:
                    break

        return results

    def get_api_reference(self, api_name: str) -> str:
        """Get API reference documentation for a specific API"""
        return self.api_references.get(api_name, "")

    def get_examples_by_category(self, category: str) -> List[str]:
        """Get all code examples from a specific category"""
        examples = [ex['code'] for ex in self.code_examples
                    if ex['category'] == category]
        return examples

    def get_stats(self) -> Dict:
        """Get indexing statistics"""
        return {
            'total_docs': len(self.indexed_docs),
            'total_code_examples': len(self.code_examples),
            'total_apis': len(self.api_references),
            'categories': list(set(doc['category'] for doc in self.indexed_docs))
        }


# Singleton instance
_indexer_instance = None


def get_indexer(docs_path: str = "./genesis-doc") -> GenesisDocIndexer:
    """Get or create singleton indexer instance"""
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = GenesisDocIndexer(docs_path)
        try:
            _indexer_instance.index_documentation()
        except Exception as e:
            logger.error(f"Failed to index documentation: {e}")
    return _indexer_instance
