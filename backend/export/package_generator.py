"""
Package Generator - Main export orchestrator

Generates complete simulation packages in various formats with all required files:
- Algorithm code
- 3D scene configuration
- Robot models
- Setup instructions (README)
"""

import os
import json
import zipfile
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# REMOVED: React and Isaac Lab exporters - using Genesis instead
# from .react_exporter import ReactExporter
# from .isaac_lab_exporter import IsaacLabExporter
from .ros_exporter import ROSExporter
from .python_exporter import PythonExporter


class PackageGenerator:
    """Main package generation engine"""

    SUPPORTED_FORMATS = ['ros', 'python', 'algorithms']  # Removed 'react', 'isaac_lab'

    def __init__(self):
        # REMOVED: React and Isaac Lab exporters
        # self.react_exporter = ReactExporter()
        # self.isaac_lab_exporter = IsaacLabExporter()
        self.ros_exporter = ROSExporter()
        self.python_exporter = PythonExporter()
        self.temp_dir = tempfile.mkdtemp(prefix='celestial_export_')

    def generate_package(
        self,
        export_format: str,
        algorithms: List[Dict[str, Any]],
        scene_config: Dict[str, Any],
        robots: List[Dict[str, Any]],
        project_name: str = "celestial_simulation"
    ) -> str:
        """
        Generate complete simulation package

        Args:
            export_format: 'react', 'ros', 'python', or 'algorithms'
            algorithms: List of algorithm objects with code and metadata
            scene_config: 3D scene configuration
            robots: List of robot configurations
            project_name: Name for the generated project

        Returns:
            Path to generated ZIP file
        """
        if export_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {export_format}. Must be one of {self.SUPPORTED_FORMATS}")

        print(f"🔄 Generating {export_format} package: {project_name}")

        # Create project directory
        project_dir = os.path.join(self.temp_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Generate files based on format
        # REMOVED: React and Isaac Lab export options
        # if export_format == 'react':
        #     files = self.react_exporter.export(algorithms, scene_config, robots, project_name)
        #     self._write_files(project_dir, files)
        # elif export_format == 'isaac_lab':
        #     self.isaac_lab_exporter.export(project_dir, scene_config, algorithms, robots, project_name)
        if export_format == 'ros':
            files = self.ros_exporter.export(algorithms, scene_config, robots, project_name)
            # Write files to disk
            self._write_files(project_dir, files)
        elif export_format == 'python':
            files = self.python_exporter.export(algorithms, scene_config, robots, project_name)
            # Write files to disk
            self._write_files(project_dir, files)
        elif export_format == 'algorithms':
            files = self._export_algorithms_only(algorithms)
            # Write files to disk
            self._write_files(project_dir, files)
        else:
            raise ValueError(f"Unsupported export format: {export_format}. Supported: {self.SUPPORTED_FORMATS}")

        # Create ZIP package
        zip_path = self._create_zip(project_dir, project_name)

        print(f"✅ Package generated: {zip_path}")
        return zip_path

    def _export_algorithms_only(self, algorithms: List[Dict[str, Any]]) -> Dict[str, str]:
        """Export only algorithm files"""
        files = {}

        # README
        files['README.md'] = self._generate_algorithms_readme(algorithms)

        # Export each algorithm
        for i, algo in enumerate(algorithms):
            filename = f"algorithm_{i+1}_{algo.get('type', 'unknown')}.ts"
            files[f"algorithms/{filename}"] = algo.get('code', '')

            # Export metadata
            metadata = {
                'name': algo.get('name', f'Algorithm {i+1}'),
                'type': algo.get('type', 'unknown'),
                'description': algo.get('description', ''),
                'complexity': algo.get('complexity', 'Unknown'),
                'parameters': algo.get('parameters', [])
            }
            files[f"algorithms/{filename}.json"] = json.dumps(metadata, indent=2)

        return files

    def _generate_algorithms_readme(self, algorithms: List[Dict[str, Any]]) -> str:
        """Generate README for algorithms-only export"""
        readme = f"""# Celestial Studio - Generated Algorithms

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Algorithms Included

"""
        for i, algo in enumerate(algorithms):
            readme += f"""
### {i+1}. {algo.get('name', f'Algorithm {i+1}')}
- **Type:** {algo.get('type', 'unknown')}
- **Complexity:** {algo.get('complexity', 'Unknown')}
- **Description:** {algo.get('description', 'No description')}
- **File:** `algorithms/algorithm_{i+1}_{algo.get('type', 'unknown')}.ts`

"""

        readme += """
## Usage

These TypeScript algorithms can be integrated into your robotics application.
Each algorithm file includes:
- Type definitions
- Implementation code
- Configurable parameters

## Integration Example

```typescript
import { findPath } from './algorithms/algorithm_1_path_planning.ts'

const start = { x: 0, z: 0 }
const goal = { x: 10, z: 10 }
const obstacles = [...]

const path = findPath(start, goal, obstacles, worldBounds)
```

## Next Steps

1. Review algorithm code and adjust parameters
2. Integrate into your application
3. Test in your simulation environment
4. Tune parameters for optimal performance

For questions or support, visit: https://github.com/samarth-sudo/celestial-studio
"""
        return readme

    def _write_files(self, base_dir: str, files: Dict[str, str]):
        """Write files to disk"""
        for filepath, content in files.items():
            full_path = os.path.join(base_dir, filepath)

            # Create subdirectories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

    def _create_zip(self, source_dir: str, project_name: str) -> str:
        """Create ZIP file from directory"""
        zip_filename = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = os.path.join(self.temp_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)

        return zip_path

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary files: {self.temp_dir}")
