"""
Code Validator for Celestial Studio

Validates AI-generated algorithm code for safety and correctness.
Prevents malicious code from executing in the browser.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    suggestions: List[str]


class CodeValidator:
    """Validates generated algorithm code for safety"""

    # Dangerous patterns that should never appear in generated code
    FORBIDDEN_PATTERNS = [
        (r'eval\s*\(', 'Use of eval() is forbidden'),
        (r'Function\s*\(', 'Use of Function constructor is forbidden'),
        (r'new\s+Function', 'Use of Function constructor is forbidden'),
        (r'constructor\s*\(.*\)\s*\{.*eval', 'Eval in constructor is forbidden'),
        (r'innerHTML\s*=', 'Direct innerHTML manipulation is forbidden (XSS risk)'),
        (r'document\.write', 'document.write is forbidden'),
        (r'__proto__', 'Prototype manipulation is forbidden'),
        (r'import\s+.*\s+from\s+[\'"]http', 'Remote imports are forbidden'),
        (r'fetch\s*\(', 'Network requests are forbidden'),
        (r'XMLHttpRequest', 'Network requests are forbidden'),
        (r'WebSocket', 'WebSocket connections are forbidden'),
        (r'localStorage', 'LocalStorage access is forbidden'),
        (r'sessionStorage', 'SessionStorage access is forbidden'),
        (r'indexedDB', 'IndexedDB access is forbidden'),
        (r'process\.', 'Process access is forbidden (Node.js API)'),
        (r'require\s*\(', 'CommonJS require is forbidden'),
        (r'global\.', 'Global object access is forbidden'),
        (r'window\[', 'Window object property access via bracket notation is forbidden'),
    ]

    # Warning patterns - allowed but potentially problematic
    WARNING_PATTERNS = [
        (r'while\s*\(true\)', 'Infinite while loop detected - ensure break condition exists'),
        (r'for\s*\([^;]*;;', 'Infinite for loop detected - ensure break condition exists'),
        (r'recursion', 'Recursive function detected - ensure base case exists'),
        (r'Array\s*\(\s*\d{5,}\s*\)', 'Very large array allocation detected'),
        (r'\.repeat\s*\(\s*\d{4,}\s*\)', 'Large string repeat operation detected'),
        (r'Math\.random', 'Random number generation - ensure deterministic behavior when needed'),
    ]

    # Required imports for algorithm types
    REQUIRED_IMPORTS = {
        'path_planning': ['THREE'],
        'obstacle_avoidance': ['THREE'],
        'inverse_kinematics': ['THREE'],
        'computer_vision': [],
        'motion_control': [],
    }

    @staticmethod
    def validate(code: str, algorithm_type: str = None) -> ValidationResult:
        """
        Validate generated algorithm code

        Args:
            code: TypeScript/JavaScript code to validate
            algorithm_type: Type of algorithm (for specific checks)

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        suggestions = []

        # 1. Check for forbidden patterns
        forbidden_found = CodeValidator._check_forbidden_patterns(code)
        errors.extend(forbidden_found)

        # 2. Check for warning patterns
        warning_found = CodeValidator._check_warning_patterns(code)
        warnings.extend(warning_found)

        # 3. Check syntax structure
        syntax_errors = CodeValidator._check_syntax(code)
        errors.extend(syntax_errors)

        # 4. Check for required imports (if algorithm type specified)
        if algorithm_type:
            import_warnings = CodeValidator._check_imports(code, algorithm_type)
            warnings.extend(import_warnings)

        # 5. Check for potential infinite loops
        loop_warnings = CodeValidator._check_loops(code)
        warnings.extend(loop_warnings)

        # 6. Check for excessive complexity
        complexity_warnings = CodeValidator._check_complexity(code)
        warnings.extend(complexity_warnings)

        # 7. Generate suggestions
        suggestions = CodeValidator._generate_suggestions(code, errors, warnings)

        # Determine risk level
        if len(errors) > 0:
            risk_level = 'high'
        elif len(warnings) > 3:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            risk_level=risk_level,
            suggestions=suggestions
        )

    @staticmethod
    def _check_forbidden_patterns(code: str) -> List[str]:
        """Check for forbidden code patterns"""
        errors = []

        for pattern, message in CodeValidator.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"FORBIDDEN: {message}")

        return errors

    @staticmethod
    def _check_warning_patterns(code: str) -> List[str]:
        """Check for warning patterns"""
        warnings = []

        for pattern, message in CodeValidator.WARNING_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                warnings.append(f"WARNING: {message}")

        return warnings

    @staticmethod
    def _check_syntax(code: str) -> List[str]:
        """Basic syntax checks"""
        errors = []

        # Check for balanced braces
        brace_count = code.count('{') - code.count('}')
        if brace_count != 0:
            errors.append(f"Unbalanced braces: {abs(brace_count)} {'opening' if brace_count > 0 else 'closing'} braces without match")

        # Check for balanced parentheses
        paren_count = code.count('(') - code.count(')')
        if paren_count != 0:
            errors.append(f"Unbalanced parentheses: {abs(paren_count)} {'opening' if paren_count > 0 else 'closing'} parentheses without match")

        # Check for balanced brackets
        bracket_count = code.count('[') - code.count(']')
        if bracket_count != 0:
            errors.append(f"Unbalanced brackets: {abs(bracket_count)} {'opening' if bracket_count > 0 else 'closing'} brackets without match")

        # Check for function declarations
        if not re.search(r'function\s+\w+|const\s+\w+\s*=\s*(\([^)]*\)|[^=]+)\s*=>', code):
            errors.append("No function declarations found - algorithm must define at least one function")

        return errors

    @staticmethod
    def _check_imports(code: str, algorithm_type: str) -> List[str]:
        """Check for required imports"""
        warnings = []

        required = CodeValidator.REQUIRED_IMPORTS.get(algorithm_type, [])

        for lib in required:
            if f"import" not in code or lib not in code:
                warnings.append(f"Missing import: {lib} is typically required for {algorithm_type}")

        return warnings

    @staticmethod
    def _check_loops(code: str) -> List[str]:
        """Check for potential infinite loops"""
        warnings = []

        # Find while loops
        while_loops = re.finditer(r'while\s*\([^)]+\)\s*\{([^}]*)\}', code, re.DOTALL)

        for match in while_loops:
            loop_body = match.group(1)
            # Check if loop body has break, return, or condition modification
            if not re.search(r'(break|return|\+\+|--|\+=|-=)', loop_body):
                warnings.append("Potential infinite loop: while loop without visible exit condition")

        # Find for loops without clear termination
        for_loops = re.finditer(r'for\s*\([^;]*;([^;]*);[^)]*\)\s*\{([^}]*)\}', code, re.DOTALL)

        for match in for_loops:
            condition = match.group(1).strip()
            loop_body = match.group(2)

            if not condition:  # Empty condition
                if not re.search(r'(break|return)', loop_body):
                    warnings.append("Potential infinite loop: for loop with empty condition and no break")

        return warnings

    @staticmethod
    def _check_complexity(code: str) -> List[str]:
        """Check for excessive code complexity"""
        warnings = []

        # Count lines of code
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('//')]
        loc = len(lines)

        if loc > 500:
            warnings.append(f"Code is very long ({loc} lines) - consider breaking into smaller functions")

        # Count nested loops (potential O(n²) or worse)
        nested_loop_count = 0
        for match in re.finditer(r'(for|while)\s*\([^)]+\)\s*\{[^}]*?(for|while)\s*\([^)]+\)', code, re.DOTALL):
            nested_loop_count += 1

        if nested_loop_count > 2:
            warnings.append(f"Deeply nested loops detected ({nested_loop_count}) - may impact performance")

        # Count function complexity (number of conditions)
        condition_count = len(re.findall(r'\bif\b|\bwhile\b|\bfor\b|\bswitch\b', code))

        if condition_count > 20:
            warnings.append(f"High cyclomatic complexity ({condition_count} branches) - code may be hard to test")

        return warnings

    @staticmethod
    def _generate_suggestions(code: str, errors: List[str], warnings: List[str]) -> List[str]:
        """Generate suggestions for improving code"""
        suggestions = []

        # Suggest type annotations if missing
        if 'function' in code and ':' not in code:
            suggestions.append("Add TypeScript type annotations for better type safety")

        # Suggest constants for magic numbers
        magic_numbers = re.findall(r'\b\d+\.?\d*\b', code)
        if len(set(magic_numbers)) > 10:
            suggestions.append("Consider extracting magic numbers as named constants")

        # Suggest comments if sparse
        comment_lines = len(re.findall(r'//.*|/\*.*?\*/', code, re.DOTALL))
        code_lines = len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('//')])

        if code_lines > 50 and comment_lines < code_lines * 0.1:
            suggestions.append("Add more comments to explain algorithm logic")

        # Suggest performance optimization if many array operations
        if code.count('.push') + code.count('.splice') + code.count('.shift') > 10:
            suggestions.append("Consider using more efficient data structures (e.g., linked lists, queues)")

        # If errors exist, suggest regeneration
        if errors:
            suggestions.append("Code has critical errors - consider regenerating with modified prompt")

        return suggestions


def validate_code(code: str, algorithm_type: str = None) -> ValidationResult:
    """
    Convenience function to validate code

    Args:
        code: Algorithm code to validate
        algorithm_type: Type of algorithm

    Returns:
        ValidationResult
    """
    return CodeValidator.validate(code, algorithm_type)
