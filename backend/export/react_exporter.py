"""
React/TypeScript Project Exporter

Generates complete standalone React project with:
- All algorithm code
- 3D scene with Three.js + Rapier physics
- Robot components
- Setup instructions
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class ReactExporter:
    """Exports simulation as React/TypeScript project"""

    def export(
        self,
        algorithms: List[Dict[str, Any]],
        scene_config: Dict[str, Any],
        robots: List[Dict[str, Any]],
        project_name: str
    ) -> Dict[str, str]:
        """
        Generate all files for React project

        Returns:
            Dictionary of {filepath: content}
        """
        files = {}

        # Package configuration
        files['package.json'] = self._generate_package_json(project_name)
        files['tsconfig.json'] = self._generate_tsconfig()
        files['vite.config.ts'] = self._generate_vite_config()
        files['index.html'] = self._generate_index_html(project_name)

        # README with setup instructions
        files['README.md'] = self._generate_readme(project_name, algorithms, robots)

        # Source files
        files['src/main.tsx'] = self._generate_main()
        files['src/App.tsx'] = self._generate_app(scene_config)
        files['src/App.css'] = self._generate_app_css()

        # Algorithm files
        for i, algo in enumerate(algorithms):
            algo_filename = f"{algo.get('type', 'algorithm')}_{i+1}.ts"
            files[f"src/algorithms/{algo_filename}"] = algo.get('code', '')

        files['src/algorithms/index.ts'] = self._generate_algorithms_index(algorithms)

        # Scene configuration
        files['src/scene/SceneConfig.json'] = json.dumps(scene_config, indent=2)
        files['src/scene/Scene.tsx'] = self._generate_scene_component(scene_config, robots)

        # Robot components
        for robot in robots:
            robot_file = self._generate_robot_component(robot)
            robot_name = robot.get('type', 'robot').capitalize()
            files[f"src/robots/{robot_name}Robot.tsx"] = robot_file

        # Utility files
        files['src/utils/physics.ts'] = self._generate_physics_utils()

        # Scripts
        files['scripts/start.sh'] = self._generate_start_script()
        files['scripts/build.sh'] = self._generate_build_script()

        # Git ignore
        files['.gitignore'] = self._generate_gitignore()

        return files

    def _generate_package_json(self, project_name: str) -> str:
        """Generate package.json"""
        package = {
            "name": project_name,
            "version": "1.0.0",
            "description": f"Celestial Studio simulation - Generated on {datetime.now().strftime('%Y-%m-%d')}",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "@react-three/fiber": "^8.15.0",
                "@react-three/drei": "^9.88.0",
                "@react-three/rapier": "^1.2.0",
                "three": "^0.158.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "@types/three": "^0.158.0",
                "@vitejs/plugin-react": "^4.2.0",
                "typescript": "^5.3.0",
                "vite": "^5.0.0"
            }
        }
        return json.dumps(package, indent=2)

    def _generate_tsconfig(self) -> str:
        """Generate TypeScript config"""
        config = {
            "compilerOptions": {
                "target": "ES2020",
                "useDefineForClassFields": True,
                "lib": ["ES2020", "DOM", "DOM.Iterable"],
                "module": "ESNext",
                "skipLibCheck": True,
                "moduleResolution": "bundler",
                "allowImportingTsExtensions": True,
                "resolveJsonModule": True,
                "isolatedModules": True,
                "noEmit": True,
                "jsx": "react-jsx",
                "strict": True,
                "noUnusedLocals": True,
                "noUnusedParameters": True,
                "noFallthroughCasesInSwitch": True
            },
            "include": ["src"],
            "references": [{"path": "./tsconfig.node.json"}]
        }
        return json.dumps(config, indent=2)

    def _generate_vite_config(self) -> str:
        """Generate Vite config"""
        return """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
})
"""

    def _generate_index_html(self, project_name: str) -> str:
        """Generate index.html"""
        return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""

    def _generate_readme(self, project_name: str, algorithms: List[Dict], robots: List[Dict]) -> str:
        """Generate comprehensive README"""
        algo_list = "\\n".join([f"- {algo.get('name', 'Algorithm')} ({algo.get('type', 'unknown')})" for algo in algorithms])
        robot_list = "\\n".join([f"- {robot.get('type', 'robot').capitalize()} Robot" for robot in robots])

        return f"""# {project_name}

Celestial Studio Simulation - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📦 What's Included

### Algorithms
{algo_list}

### Robots
{robot_list}

### 3D Scene
- Complete environment with physics simulation
- Obstacles and goals
- Real-time rendering with Three.js

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The simulation will open at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## 📁 Project Structure

```
{project_name}/
├── src/
│   ├── algorithms/      # Generated algorithm implementations
│   ├── robots/          # Robot components
│   ├── scene/           # 3D scene configuration
│   ├── utils/           # Utility functions
│   ├── App.tsx          # Main application
│   └── main.tsx         # Entry point
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 🎮 Controls

- **Rotate:** Left mouse drag
- **Pan:** Right mouse drag
- **Zoom:** Mouse wheel

## 🔧 Customization

### Modify Algorithm Parameters

Edit files in `src/algorithms/` to adjust algorithm behavior.

### Change Scene Configuration

Edit `src/scene/SceneConfig.json` to modify:
- Environment type
- Object positions
- Robot starting positions

### Add Custom Robots

Add new robot components in `src/robots/` following the existing patterns.

## 📚 Documentation

- [Three.js Docs](https://threejs.org/docs/)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)
- [Rapier Physics](https://rapier.rs/)

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5173
lsof -ti:5173 | xargs kill -9

# Or specify different port
npm run dev -- --port 3000
```

### Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## 📄 License

This project was generated by Celestial Studio.
Original simulation and algorithms © Your Name

## 🔗 Links

- [Celestial Studio GitHub](https://github.com/samarth-sudo/celestial-studio)
- [Report Issues](https://github.com/samarth-sudo/celestial-studio/issues)
"""

    def _generate_main(self) -> str:
        """Generate main.tsx entry point"""
        return """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
"""

    def _generate_app(self, scene_config: Dict) -> str:
        """Generate main App component"""
        return """import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Physics } from '@react-three/rapier'
import Scene from './scene/Scene'
import './App.css'

function App() {
  return (
    <div className="app">
      <div className="header">
        <h1>Celestial Studio Simulation</h1>
        <p>Generated with AI-powered robotics IDE</p>
      </div>

      <Canvas
        camera={{ position: [10, 10, 10], fov: 50 }}
        shadows
      >
        <color attach="background" args={['#1a1a2e']} />
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[10, 10, 5]}
          intensity={1}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />

        <Physics>
          <Scene />
        </Physics>

        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
        />
      </Canvas>

      <div className="info">
        <p>🖱️ Left click + drag to rotate | Right click + drag to pan | Scroll to zoom</p>
      </div>
    </div>
  )
}

export default App
"""

    def _generate_app_css(self) -> str:
        """Generate CSS"""
        return """* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.app {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
}

.header {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  padding: 20px;
  background: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent);
  color: white;
  z-index: 10;
}

.header h1 {
  font-size: 24px;
  margin-bottom: 5px;
}

.header p {
  font-size: 14px;
  opacity: 0.8;
}

.info {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  padding: 10px 20px;
  background: rgba(0,0,0,0.7);
  color: white;
  border-radius: 20px;
  font-size: 14px;
  z-index: 10;
}

canvas {
  width: 100%;
  height: 100%;
}
"""

    def _generate_algorithms_index(self, algorithms: List[Dict]) -> str:
        """Generate algorithms index file"""
        exports = []
        for i, algo in enumerate(algorithms):
            algo_name = algo.get('type', 'algorithm').replace('_', '')
            filename = f"{algo.get('type', 'algorithm')}_{i+1}"
            exports.append(f"export * from './{filename}'")

        return "\n".join(exports) + "\n"

    def _generate_scene_component(self, scene_config: Dict, robots: List[Dict]) -> str:
        """Generate Scene component"""
        robot_imports = "\n".join([
            f"import {r.get('type', 'robot').capitalize()}Robot from '../robots/{r.get('type', 'robot').capitalize()}Robot'"
            for r in robots
        ])

        robot_components = "\n      ".join([
            f"<{r.get('type', 'robot').capitalize()}Robot position={{[{r.get('position', [0,0,0])[0]}, {r.get('position', [0,0,0])[1]}, {r.get('position', [0,0,0])[2]}]}} />"
            for r in robots
        ])

        return f"""import {{ RigidBody }} from '@react-three/rapier'
{robot_imports}
import sceneConfig from './SceneConfig.json'

export default function Scene() {{
  return (
    <>
      {{/* Floor */}}
      <RigidBody type="fixed">
        <mesh receiveShadow position={{[0, -0.1, 0]}}>
          <boxGeometry args={{[20, 0.2, 20]}} />
          <meshStandardMaterial color="#444444" />
        </mesh>
      </RigidBody>

      {{/* Robots */}}
      {robot_components}

      {{/* Grid Helper */}}
      <gridHelper args={{[20, 20, '#666666', '#333333']}} position={{[0, 0, 0]}} />
    </>
  )
}}
"""

    def _generate_robot_component(self, robot: Dict) -> str:
        """Generate robot component"""
        robot_type = robot.get('type', 'mobile').capitalize()
        return f"""import {{ useRef }} from 'react'
import {{ useFrame }} from '@react-three/fiber'
import {{ RigidBody, RigidBodyApi }} from '@react-three/rapier'
import * as THREE from 'three'

export default function {robot_type}Robot({{ position = [0, 0.5, 0] }}) {{
  const robotRef = useRef<RigidBodyApi>(null)

  useFrame(() => {{
    // Add robot logic here
  }})

  return (
    <RigidBody ref={{robotRef}} position={{position}} colliders="cuboid">
      <mesh castShadow>
        <boxGeometry args={{[1, 0.5, 0.8]}} />
        <meshStandardMaterial color="#4a90e2" />
      </mesh>
    </RigidBody>
  )
}}
"""

    def _generate_physics_utils(self) -> str:
        """Generate physics utility functions"""
        return """import * as THREE from 'three'

export function calculateDistance(p1: THREE.Vector3, p2: THREE.Vector3): number {
  return p1.distanceTo(p2)
}

export function normalizeVector(v: THREE.Vector3): THREE.Vector3 {
  return v.clone().normalize()
}

export function clampVelocity(velocity: THREE.Vector3, maxSpeed: number): THREE.Vector3 {
  if (velocity.length() > maxSpeed) {
    return velocity.normalize().multiplyScalar(maxSpeed)
  }
  return velocity
}
"""

    def _generate_start_script(self) -> str:
        """Generate start.sh script"""
        return """#!/bin/bash
echo "🚀 Starting Celestial Studio Simulation..."
npm run dev
"""

    def _generate_build_script(self) -> str:
        """Generate build.sh script"""
        return """#!/bin/bash
echo "🔨 Building production bundle..."
npm run build
echo "✅ Build complete! Run 'npm run preview' to test the build."
"""

    def _generate_gitignore(self) -> str:
        """Generate .gitignore"""
        return """# Dependencies
node_modules/

# Build output
dist/
build/

# Environment files
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
npm-debug.log*

# Temp files
*.tmp
"""
