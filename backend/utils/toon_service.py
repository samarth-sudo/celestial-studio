"""
TOON Format Service for Python Backend

Provides token-efficient serialization for LLM interactions.
Implements a simplified TOON format for uniform arrays.
Falls back to JSON for non-uniform or nested data.

TOON Format Example:
```
benchmarks: 3
algorithm_id execution_time_ms success_rate collision_count
algo_1 45.2 0.95 0
algo_2 52.1 0.87 1
algo_3 38.9 1.0 0
```
"""

import json
from typing import List, Dict, Any, Optional


class ToonService:
    """TOON format serialization service"""

    def serialize_benchmark_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Serialize benchmark results to TOON format

        TOON is highly efficient for uniform tabular data like benchmarks
        """
        try:
            if not results:
                return "benchmarks: 0\n"

            # Check if data is uniform (all dicts have same keys)
            if not self._is_uniform(results):
                print("⚠️  Data not uniform, using JSON fallback")
                return json.dumps(results, indent=2)

            # Extract field names
            fields = list(results[0].keys())

            # Build TOON format
            lines = []
            lines.append(f"benchmarks: {len(results)}")
            lines.append(" ".join(fields))

            # Add data rows
            for item in results:
                row_values = []
                for field in fields:
                    value = item.get(field)
                    # Format value
                    if value is None:
                        row_values.append("null")
                    elif isinstance(value, bool):
                        row_values.append(str(value).lower())
                    elif isinstance(value, (int, float)):
                        row_values.append(str(value))
                    elif isinstance(value, str):
                        # Escape spaces in strings
                        if " " in value:
                            row_values.append(f'"{value}"')
                        else:
                            row_values.append(value)
                    else:
                        row_values.append(json.dumps(value))

                lines.append(" ".join(row_values))

            toon_string = "\n".join(lines)

            # Log savings
            json_string = json.dumps(results, indent=2)
            savings = (1 - len(toon_string) / len(json_string)) * 100
            print(f"📊 TOON Serialization:")
            print(f"   Original JSON: {len(json_string)} chars")
            print(f"   TOON format: {len(toon_string)} chars")
            print(f"   Savings: {savings:.1f}%")

            return toon_string

        except Exception as e:
            print(f"⚠️  TOON serialization error: {e}, using JSON fallback")
            return json.dumps(results, indent=2)

    def serialize_scene_config(self, config: Dict[str, Any]) -> str:
        """Serialize scene configuration to TOON format"""
        try:
            # Scene config is typically nested, so JSON might be better
            # But we can flatten key values for LLM
            lines = []
            lines.append("scene_config:")

            for key, value in config.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"  {key}: {value}")
                elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                    lines.append(f"  {key}: {', '.join(value)}")
                else:
                    lines.append(f"  {key}: {json.dumps(value)}")

            return "\n".join(lines)

        except Exception as e:
            print(f"⚠️  TOON serialization error: {e}")
            return json.dumps(config, indent=2)

    def serialize_comparison_data(self, comparison: Dict[str, Any]) -> str:
        """Serialize algorithm comparison data for LLM analysis"""
        try:
            # Extract rankings (uniform array)
            rankings = comparison.get('rankings', [])

            if rankings and self._is_uniform(rankings):
                toon_rankings = self.serialize_benchmark_results(rankings)

                # Combine with other data
                output = "# Algorithm Comparison Data\n\n"
                output += "## Rankings (TOON Format)\n"
                output += toon_rankings + "\n\n"

                # Add recommendation
                recommendation = comparison.get('recommendation', {})
                output += "## Recommendation\n"
                output += f"Algorithm: {recommendation.get('algorithm_name', 'N/A')}\n"
                output += f"Confidence: {recommendation.get('confidence', 0):.2f}\n"
                output += "Reasons:\n"
                for reason in recommendation.get('reasons', []):
                    output += f"- {reason}\n"

                return output

            return json.dumps(comparison, indent=2)

        except Exception as e:
            print(f"⚠️  TOON serialization error: {e}")
            return json.dumps(comparison, indent=2)

    def prepare_for_llm(
        self,
        data: Any,
        data_type: str = 'benchmark'
    ) -> str:
        """
        Prepare data for LLM prompt with TOON format
        Returns formatted string ready to include in LLM context
        """
        if data_type == 'benchmark':
            toon_data = self.serialize_benchmark_results(data)
        elif data_type == 'scene':
            toon_data = self.serialize_scene_config(data)
        elif data_type == 'comparison':
            toon_data = self.serialize_comparison_data(data)
        else:
            toon_data = json.dumps(data, indent=2)

        return f"""
Data Format: TOON (Token-Oriented Object Notation)
Note: This is a compact format optimized for LLM parsing.
Each row represents one record with consistent field structure.

{toon_data}
"""

    def estimate_token_savings(self, original_json: str, toon_string: str) -> Dict[str, Any]:
        """
        Calculate token savings estimate
        Uses rough approximation of 1 token ≈ 4 characters
        """
        json_tokens = len(original_json) // 4
        toon_tokens = len(toon_string) // 4
        savings = json_tokens - toon_tokens
        savings_percent = (savings / json_tokens) * 100 if json_tokens > 0 else 0

        return {
            "json_tokens": json_tokens,
            "toon_tokens": toon_tokens,
            "savings": savings,
            "savings_percent": round(savings_percent, 1)
        }

    def serialize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Serialize conversation messages to TOON format

        Optimized for chat history with role, content, and timestamp

        Args:
            messages: List of message dicts with 'role', 'content', 'timestamp'

        Returns:
            TOON formatted string
        """
        try:
            if not messages:
                return "conversation: 0\n"

            lines = []
            lines.append(f"conversation: {len(messages)}")
            lines.append("timestamp role content_preview")

            for msg in messages:
                timestamp = msg.get('timestamp', '')[:19]  # YYYY-MM-DD HH:MM:SS
                role = msg.get('role', 'unknown')[:10]  # Limit role length
                content = msg.get('content', '')

                # Create preview (first 80 chars, replace newlines)
                preview = content[:80].replace('\n', ' ').replace('\r', ' ')
                if len(content) > 80:
                    preview += "..."

                # Escape spaces in preview
                if ' ' in preview:
                    preview = f'"{preview}"'

                lines.append(f"{timestamp} {role} {preview}")

            toon_string = "\n".join(lines)

            # Log savings
            json_string = json.dumps(messages, indent=2)
            savings = (1 - len(toon_string) / len(json_string)) * 100
            print(f"📊 TOON Conversation Serialization:")
            print(f"   Original JSON: {len(json_string)} chars")
            print(f"   TOON format: {len(toon_string)} chars")
            print(f"   Savings: {savings:.1f}%")

            return toon_string

        except Exception as e:
            print(f"⚠️  TOON conversation serialization error: {e}")
            return json.dumps(messages, indent=2)

    def _is_uniform(self, data: List[Dict]) -> bool:
        """Check if list of dicts has uniform structure"""
        if not data:
            return True

        first_keys = set(data[0].keys())
        return all(set(item.keys()) == first_keys for item in data)


# Singleton instance
_toon_service_instance: Optional[ToonService] = None


def get_toon_service() -> ToonService:
    """Get or create ToonService singleton"""
    global _toon_service_instance
    if _toon_service_instance is None:
        _toon_service_instance = ToonService()
    return _toon_service_instance
