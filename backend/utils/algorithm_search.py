"""
Algorithm Research Search Utility

Uses Tavily API to search for latest algorithms, research papers, and techniques.
Focuses on robotics algorithms from 2024-2025 research.
"""

import os
from typing import List, Dict, Optional
from tavily import TavilyClient


class AlgorithmSearcher:
    """Search for latest algorithm research using Tavily API"""

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        self.client = TavilyClient(api_key=self.api_key)

    def search_algorithm(
        self,
        task_description: str,
        robot_type: str = "mobile",
        year_filter: str = "2024-2025"
    ) -> Dict[str, any]:
        """
        Search for relevant algorithms for a given task

        Args:
            task_description: What the algorithm should do (e.g., "path planning with obstacles")
            robot_type: Type of robot (mobile, arm, drone, humanoid)
            year_filter: Year range for research papers (default: 2024-2025)

        Returns:
            Dict with:
                - papers: List of relevant research papers
                - techniques: List of algorithm techniques found
                - summary: Brief summary of findings
                - sources: List of URLs for citation
        """

        # Build search query
        query = f"{task_description} {robot_type} robot algorithm {year_filter} research paper"

        try:
            # Search with Tavily (using basic depth for faster response)
            # Note: Tavily client has internal timeout handling
            response = self.client.search(
                query=query,
                search_depth="basic",  # Changed from "advanced" for faster search (10-15s faster)
                max_results=3,  # Reduced from 5 for faster response
                include_domains=[
                    "arxiv.org",
                    "github.com",
                    "paperswithcode.com"
                ],
                timeout=15  # 15 second timeout to prevent blocking algorithm generation
            )

            papers = []
            techniques = []
            sources = []

            # Parse results
            for result in response.get('results', []):
                title = result.get('title', '')
                url = result.get('url', '')
                content = result.get('content', '')

                # Extract paper info
                if any(domain in url for domain in ['arxiv', 'ieee', 'springer', 'paperswithcode']):
                    papers.append({
                        'title': title,
                        'url': url,
                        'summary': content[:300]  # First 300 chars
                    })

                # Extract technique names (simple heuristic)
                if 'algorithm' in content.lower() or 'method' in content.lower():
                    techniques.append({
                        'name': title,
                        'description': content[:200],
                        'source': url
                    })

                sources.append(url)

            # Generate summary
            summary = self._generate_summary(papers, techniques)

            return {
                'papers': papers[:3],  # Top 3 papers
                'techniques': techniques[:5],  # Top 5 techniques
                'summary': summary,
                'sources': sources,
                'search_query': query
            }

        except Exception as e:
            return {
                'papers': [],
                'techniques': [],
                'summary': f'Web search failed: {str(e)}',
                'sources': [],
                'search_query': query,
                'error': str(e)
            }

    def _generate_summary(self, papers: List[Dict], techniques: List[Dict]) -> str:
        """Generate a brief summary of search results"""

        if not papers and not techniques:
            return "No recent research papers found. Using classical algorithms."

        summary_parts = []

        if papers:
            summary_parts.append(f"Found {len(papers)} recent papers:")
            for paper in papers[:2]:
                summary_parts.append(f"- {paper['title']}")

        if techniques:
            summary_parts.append(f"\nRelevant techniques:")
            for tech in techniques[:3]:
                summary_parts.append(f"- {tech['name']}")

        return "\n".join(summary_parts)


def search_latest_algorithms(
    task: str,
    robot_type: str = "mobile",
    use_web_search: bool = True
) -> Optional[Dict]:
    """
    Convenience function to search for latest algorithms

    Args:
        task: Task description (e.g., "navigate warehouse with obstacles")
        robot_type: Type of robot
        use_web_search: Whether to use web search (requires Tavily API)

    Returns:
        Search results dict or None if search disabled/failed
    """

    if not use_web_search:
        return None

    try:
        searcher = AlgorithmSearcher()
        return searcher.search_algorithm(task, robot_type)
    except ValueError as e:
        # API key not set
        print(f"⚠️ Algorithm search disabled: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Algorithm search failed: {e}")
        return None
