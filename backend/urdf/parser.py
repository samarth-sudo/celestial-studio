"""
URDF Parser

Parses URDF XML files and extracts robot structure including:
- Links (visual, collision, inertial properties)
- Joints (type, parent, child, limits, axis)
- Materials and colors
- Geometry (box, cylinder, sphere, mesh)
"""

from xml.etree import ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import os


@dataclass
class URDFGeometry:
    """Geometry definition"""
    type: str  # 'box', 'cylinder', 'sphere', 'mesh'
    # Box
    size: Optional[List[float]] = None  # [x, y, z]
    # Cylinder
    radius: Optional[float] = None
    length: Optional[float] = None  # cylinder length
    # Sphere
    # radius already defined
    # Mesh
    filename: Optional[str] = None
    scale: Optional[List[float]] = None


@dataclass
class URDFMaterial:
    """Material definition"""
    name: str
    color: Optional[List[float]] = None  # RGBA [r, g, b, a]
    texture: Optional[str] = None


@dataclass
class URDFOrigin:
    """Origin (position + orientation)"""
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # roll, pitch, yaw


@dataclass
class URDFVisual:
    """Visual properties"""
    name: Optional[str] = None
    origin: URDFOrigin = field(default_factory=URDFOrigin)
    geometry: Optional[URDFGeometry] = None
    material: Optional[URDFMaterial] = None


@dataclass
class URDFCollision:
    """Collision properties"""
    name: Optional[str] = None
    origin: URDFOrigin = field(default_factory=URDFOrigin)
    geometry: Optional[URDFGeometry] = None


@dataclass
class URDFInertial:
    """Inertial properties"""
    origin: URDFOrigin = field(default_factory=URDFOrigin)
    mass: float = 1.0
    inertia: Dict[str, float] = field(default_factory=dict)


@dataclass
class URDFLink:
    """Link definition (rigid body)"""
    name: str
    visual: Optional[URDFVisual] = None
    collision: Optional[URDFCollision] = None
    inertial: Optional[URDFInertial] = None


@dataclass
class URDFJoint:
    """Joint definition (connects two links)"""
    name: str
    type: str  # 'fixed', 'revolute', 'continuous', 'prismatic', 'floating', 'planar'
    parent: str  # parent link name
    child: str  # child link name
    origin: URDFOrigin = field(default_factory=URDFOrigin)
    axis: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    # Limits (for revolute/prismatic)
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None
    limit_effort: Optional[float] = None
    limit_velocity: Optional[float] = None
    # Dynamics
    damping: Optional[float] = None
    friction: Optional[float] = None


class URDFParser:
    """Parse URDF XML files"""

    def __init__(self):
        self.materials: Dict[str, URDFMaterial] = {}

    def parse_file(self, filepath: str) -> Dict:
        """
        Parse URDF file and return structured robot data

        Args:
            filepath: Path to URDF XML file

        Returns:
            Dictionary with robot name, links, joints, materials
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"URDF file not found: {filepath}")

        tree = ET.parse(filepath)
        root = tree.getroot()

        if root.tag != 'robot':
            raise ValueError("Invalid URDF: root element must be 'robot'")

        robot_name = root.get('name', 'unnamed_robot')

        print(f"📄 Parsing URDF: {robot_name}")

        # Parse materials first (they're referenced by links)
        self.materials = self._parse_materials(root)

        # Parse links and joints
        links = self._parse_links(root)
        joints = self._parse_joints(root)

        print(f"   Found {len(links)} links, {len(joints)} joints")

        return {
            'name': robot_name,
            'links': [asdict(link) for link in links],
            'joints': [asdict(joint) for joint in joints],
            'materials': {name: asdict(mat) for name, mat in self.materials.items()}
        }

    def parse_string(self, urdf_xml: str) -> Dict:
        """Parse URDF from XML string"""
        root = ET.fromstring(urdf_xml)

        if root.tag != 'robot':
            raise ValueError("Invalid URDF: root element must be 'robot'")

        robot_name = root.get('name', 'unnamed_robot')

        self.materials = self._parse_materials(root)
        links = self._parse_links(root)
        joints = self._parse_joints(root)

        return {
            'name': robot_name,
            'links': [asdict(link) for link in links],
            'joints': [asdict(joint) for joint in joints],
            'materials': {name: asdict(mat) for name, mat in self.materials.items()}
        }

    def _parse_materials(self, root: ET.Element) -> Dict[str, URDFMaterial]:
        """Parse material definitions"""
        materials = {}

        for material_elem in root.findall('material'):
            name = material_elem.get('name')
            if not name:
                continue

            material = URDFMaterial(name=name)

            # Parse color
            color_elem = material_elem.find('color')
            if color_elem is not None:
                rgba = color_elem.get('rgba', '0.8 0.8 0.8 1.0')
                material.color = [float(x) for x in rgba.split()]

            # Parse texture
            texture_elem = material_elem.find('texture')
            if texture_elem is not None:
                material.texture = texture_elem.get('filename')

            materials[name] = material

        return materials

    def _parse_links(self, root: ET.Element) -> List[URDFLink]:
        """Parse link definitions"""
        links = []

        for link_elem in root.findall('link'):
            name = link_elem.get('name')
            if not name:
                continue

            link = URDFLink(name=name)

            # Parse visual
            visual_elem = link_elem.find('visual')
            if visual_elem is not None:
                link.visual = self._parse_visual(visual_elem)

            # Parse collision
            collision_elem = link_elem.find('collision')
            if collision_elem is not None:
                link.collision = self._parse_collision(collision_elem)

            # Parse inertial
            inertial_elem = link_elem.find('inertial')
            if inertial_elem is not None:
                link.inertial = self._parse_inertial(inertial_elem)

            links.append(link)

        return links

    def _parse_visual(self, visual_elem: ET.Element) -> URDFVisual:
        """Parse visual element"""
        visual = URDFVisual()
        visual.name = visual_elem.get('name')

        # Parse origin
        origin_elem = visual_elem.find('origin')
        if origin_elem is not None:
            visual.origin = self._parse_origin(origin_elem)

        # Parse geometry
        geometry_elem = visual_elem.find('geometry')
        if geometry_elem is not None:
            visual.geometry = self._parse_geometry(geometry_elem)

        # Parse material
        material_elem = visual_elem.find('material')
        if material_elem is not None:
            material_name = material_elem.get('name')
            if material_name and material_name in self.materials:
                visual.material = self.materials[material_name]
            else:
                # Inline material definition
                material = URDFMaterial(name=material_name or 'unnamed')
                color_elem = material_elem.find('color')
                if color_elem is not None:
                    rgba = color_elem.get('rgba', '0.8 0.8 0.8 1.0')
                    material.color = [float(x) for x in rgba.split()]
                visual.material = material

        return visual

    def _parse_collision(self, collision_elem: ET.Element) -> URDFCollision:
        """Parse collision element"""
        collision = URDFCollision()
        collision.name = collision_elem.get('name')

        origin_elem = collision_elem.find('origin')
        if origin_elem is not None:
            collision.origin = self._parse_origin(origin_elem)

        geometry_elem = collision_elem.find('geometry')
        if geometry_elem is not None:
            collision.geometry = self._parse_geometry(geometry_elem)

        return collision

    def _parse_inertial(self, inertial_elem: ET.Element) -> URDFInertial:
        """Parse inertial properties"""
        inertial = URDFInertial()

        origin_elem = inertial_elem.find('origin')
        if origin_elem is not None:
            inertial.origin = self._parse_origin(origin_elem)

        mass_elem = inertial_elem.find('mass')
        if mass_elem is not None:
            inertial.mass = float(mass_elem.get('value', 1.0))

        inertia_elem = inertial_elem.find('inertia')
        if inertia_elem is not None:
            inertial.inertia = {
                'ixx': float(inertia_elem.get('ixx', 0)),
                'ixy': float(inertia_elem.get('ixy', 0)),
                'ixz': float(inertia_elem.get('ixz', 0)),
                'iyy': float(inertia_elem.get('iyy', 0)),
                'iyz': float(inertia_elem.get('iyz', 0)),
                'izz': float(inertia_elem.get('izz', 0))
            }

        return inertial

    def _parse_origin(self, origin_elem: ET.Element) -> URDFOrigin:
        """Parse origin (position + rotation)"""
        origin = URDFOrigin()

        xyz_str = origin_elem.get('xyz', '0 0 0')
        origin.xyz = [float(x) for x in xyz_str.split()]

        rpy_str = origin_elem.get('rpy', '0 0 0')
        origin.rpy = [float(x) for x in rpy_str.split()]

        return origin

    def _parse_geometry(self, geometry_elem: ET.Element) -> URDFGeometry:
        """Parse geometry element"""
        # Check for box
        box_elem = geometry_elem.find('box')
        if box_elem is not None:
            size_str = box_elem.get('size', '1 1 1')
            size = [float(x) for x in size_str.split()]
            return URDFGeometry(type='box', size=size)

        # Check for cylinder
        cylinder_elem = geometry_elem.find('cylinder')
        if cylinder_elem is not None:
            radius = float(cylinder_elem.get('radius', 0.5))
            length = float(cylinder_elem.get('length', 1.0))
            return URDFGeometry(type='cylinder', radius=radius, length=length)

        # Check for sphere
        sphere_elem = geometry_elem.find('sphere')
        if sphere_elem is not None:
            radius = float(sphere_elem.get('radius', 0.5))
            return URDFGeometry(type='sphere', radius=radius)

        # Check for mesh
        mesh_elem = geometry_elem.find('mesh')
        if mesh_elem is not None:
            filename = mesh_elem.get('filename', '')
            scale_str = mesh_elem.get('scale', '1 1 1')
            scale = [float(x) for x in scale_str.split()]
            return URDFGeometry(type='mesh', filename=filename, scale=scale)

        # Default to box if no geometry found
        return URDFGeometry(type='box', size=[1.0, 1.0, 1.0])

    def _parse_joints(self, root: ET.Element) -> List[URDFJoint]:
        """Parse joint definitions"""
        joints = []

        for joint_elem in root.findall('joint'):
            name = joint_elem.get('name')
            joint_type = joint_elem.get('type', 'fixed')

            if not name:
                continue

            joint = URDFJoint(name=name, type=joint_type, parent='', child='')

            # Parse parent
            parent_elem = joint_elem.find('parent')
            if parent_elem is not None:
                joint.parent = parent_elem.get('link', '')

            # Parse child
            child_elem = joint_elem.find('child')
            if child_elem is not None:
                joint.child = child_elem.get('link', '')

            # Parse origin
            origin_elem = joint_elem.find('origin')
            if origin_elem is not None:
                joint.origin = self._parse_origin(origin_elem)

            # Parse axis
            axis_elem = joint_elem.find('axis')
            if axis_elem is not None:
                xyz_str = axis_elem.get('xyz', '1 0 0')
                joint.axis = [float(x) for x in xyz_str.split()]

            # Parse limits (for revolute/prismatic joints)
            limit_elem = joint_elem.find('limit')
            if limit_elem is not None:
                joint.limit_lower = float(limit_elem.get('lower', 0))
                joint.limit_upper = float(limit_elem.get('upper', 0))
                joint.limit_effort = float(limit_elem.get('effort', 0))
                joint.limit_velocity = float(limit_elem.get('velocity', 0))

            # Parse dynamics
            dynamics_elem = joint_elem.find('dynamics')
            if dynamics_elem is not None:
                joint.damping = float(dynamics_elem.get('damping', 0))
                joint.friction = float(dynamics_elem.get('friction', 0))

            joints.append(joint)

        return joints
