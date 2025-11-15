"""
URDF to Three.js Converter

Converts parsed URDF data to Three.js-compatible scene configuration
for rendering in Celestial Studio's 3D simulation.
"""

from typing import Dict, List, Any, Optional
import math


class ThreeJSConverter:
    """Convert URDF robot model to Three.js scene configuration"""

    def convert(self, urdf_data: Dict) -> Dict:
        """
        Convert URDF data to Celestial Studio scene format

        Args:
            urdf_data: Parsed URDF data from URDFParser

        Returns:
            Three.js-compatible scene configuration
        """
        robot_name = urdf_data.get('name', 'robot')
        links = urdf_data.get('links', [])
        joints = urdf_data.get('joints', [])
        materials = urdf_data.get('materials', {})

        print(f"🔄 Converting {robot_name} to Three.js format...")

        # Convert links to Three.js meshes
        three_links = []
        for link in links:
            if link.get('visual'):  # Only convert links with visual components
                three_link = self._convert_link(link, materials)
                if three_link:
                    three_links.append(three_link)

        # Convert joints to constraints/transforms
        three_joints = []
        for joint in joints:
            three_joint = self._convert_joint(joint)
            if three_joint:
                three_joints.append(three_joint)

        # Build kinematic tree
        kinematic_tree = self._build_kinematic_tree(joints, links)

        scene_config = {
            "robot": {
                "type": "urdf_custom",
                "name": robot_name,
                "links": three_links,
                "joints": three_joints,
                "kinematic_tree": kinematic_tree,
                "base_link": self._find_base_link(joints) if joints else (links[0]['name'] if links else None)
            }
        }

        print(f"✅ Converted to {len(three_links)} meshes, {len(three_joints)} joints")

        return scene_config

    def _convert_link(self, link: Dict, materials: Dict) -> Optional[Dict]:
        """Convert URDF link to Three.js mesh configuration"""
        visual = link.get('visual')
        if not visual:
            return None

        link_name = link.get('name', 'unnamed_link')
        geometry = visual.get('geometry')
        origin = visual.get('origin', {})
        material = visual.get('material')

        if not geometry:
            return None

        # Convert geometry
        three_geometry = self._convert_geometry(geometry)
        if not three_geometry:
            return None

        # Convert material/color
        color = self._convert_material(material, materials)

        # Convert origin (position + rotation)
        position = origin.get('xyz', [0, 0, 0])
        rotation = origin.get('rpy', [0, 0, 0])  # Roll, pitch, yaw

        # Convert RPY (roll-pitch-yaw) to Three.js Euler angles
        # URDF uses RPY (XYZ rotation order), Three.js uses XYZ Euler
        three_rotation = rotation  # Keep as-is, Three.js will interpret correctly

        return {
            "name": link_name,
            "geometry": three_geometry,
            "material": {
                "color": color,
                "metalness": 0.3,
                "roughness": 0.7
            },
            "position": position,
            "rotation": three_rotation,
            "castShadow": True,
            "receiveShadow": True
        }

    def _convert_geometry(self, geometry: Dict) -> Optional[Dict]:
        """Convert URDF geometry to Three.js geometry"""
        geom_type = geometry.get('type')

        if geom_type == 'box':
            size = geometry.get('size', [1, 1, 1])
            return {
                "type": "box",
                "args": size  # [width, height, depth]
            }

        elif geom_type == 'cylinder':
            radius = geometry.get('radius', 0.5)
            length = geometry.get('length', 1.0)
            return {
                "type": "cylinder",
                "args": [radius, radius, length, 32]  # [radiusTop, radiusBottom, height, segments]
            }

        elif geom_type == 'sphere':
            radius = geometry.get('radius', 0.5)
            return {
                "type": "sphere",
                "args": [radius, 32, 32]  # [radius, widthSegments, heightSegments]
            }

        elif geom_type == 'mesh':
            filename = geometry.get('filename', '')
            scale = geometry.get('scale', [1, 1, 1])
            return {
                "type": "mesh",
                "filename": filename,
                "scale": scale
            }

        return None

    def _convert_material(self, material: Optional[Dict], materials: Dict) -> str:
        """Convert URDF material to Three.js color (hex string)"""
        if not material:
            return "#808080"  # Default gray

        # Check if material has color
        color_rgba = material.get('color')
        if color_rgba:
            # Convert RGBA [r,g,b,a] (0-1 range) to hex color
            r = int(color_rgba[0] * 255)
            g = int(color_rgba[1] * 255)
            b = int(color_rgba[2] * 255)
            return f"#{r:02x}{g:02x}{b:02x}"

        # Try to look up material by name
        material_name = material.get('name')
        if material_name and material_name in materials:
            mat_data = materials[material_name]
            color_rgba = mat_data.get('color')
            if color_rgba:
                r = int(color_rgba[0] * 255)
                g = int(color_rgba[1] * 255)
                b = int(color_rgba[2] * 255)
                return f"#{r:02x}{g:02x}{b:02x}"

        return "#808080"  # Fallback gray

    def _convert_joint(self, joint: Dict) -> Dict:
        """Convert URDF joint to Three.js constraint/transform"""
        joint_name = joint.get('name', 'unnamed_joint')
        joint_type = joint.get('type', 'fixed')
        parent = joint.get('parent', '')
        child = joint.get('child', '')
        origin = joint.get('origin', {})
        axis = joint.get('axis', [1, 0, 0])

        # Get limits
        limit_lower = joint.get('limit_lower', 0)
        limit_upper = joint.get('limit_upper', 0)
        limit_effort = joint.get('limit_effort', 100)
        limit_velocity = joint.get('limit_velocity', 1.0)

        # Position and rotation of joint
        position = origin.get('xyz', [0, 0, 0])
        rotation = origin.get('rpy', [0, 0, 0])

        return {
            "name": joint_name,
            "type": joint_type,
            "parent_link": parent,
            "child_link": child,
            "position": position,
            "rotation": rotation,
            "axis": axis,
            "limits": {
                "lower": limit_lower,
                "upper": limit_upper,
                "effort": limit_effort,
                "velocity": limit_velocity
            },
            "current_angle": 0.0  # Initial joint angle
        }

    def _build_kinematic_tree(self, joints: List[Dict], links: List[Dict]) -> Dict:
        """
        Build kinematic tree showing parent-child relationships

        Returns tree structure like:
        {
            "base_link": {
                "children": {
                    "link1": {
                        "joint": "joint1",
                        "children": {...}
                    }
                }
            }
        }
        """
        # Build parent->children mapping
        tree = {}
        link_to_children = {}

        for joint in joints:
            parent = joint.get('parent', '')
            child = joint.get('child', '')
            joint_name = joint.get('name', '')

            if parent not in link_to_children:
                link_to_children[parent] = []

            link_to_children[parent].append({
                "link": child,
                "joint": joint_name
            })

        # Find base link (link with no parent)
        all_children = set()
        for joint in joints:
            all_children.add(joint.get('child', ''))

        all_links = {link.get('name', '') for link in links}
        base_links = all_links - all_children

        if base_links:
            base_link = list(base_links)[0]
        elif links:
            base_link = links[0].get('name', '')
        else:
            return {}

        # Recursively build tree
        def build_subtree(link_name: str) -> Dict:
            subtree = {"name": link_name, "children": {}}

            if link_name in link_to_children:
                for child_info in link_to_children[link_name]:
                    child_link = child_info["link"]
                    child_joint = child_info["joint"]

                    subtree["children"][child_link] = {
                        "joint": child_joint,
                        **build_subtree(child_link)
                    }

            return subtree

        tree[base_link] = build_subtree(base_link)

        return tree

    def _find_base_link(self, joints: List[Dict]) -> Optional[str]:
        """Find the base link (root of kinematic tree)"""
        if not joints:
            return None

        # Base link is a parent that is never a child
        all_parents = {joint.get('parent', '') for joint in joints}
        all_children = {joint.get('child', '') for joint in joints}

        base_links = all_parents - all_children

        if base_links:
            return list(base_links)[0]

        # Fallback: return first parent
        return joints[0].get('parent', None) if joints else None
