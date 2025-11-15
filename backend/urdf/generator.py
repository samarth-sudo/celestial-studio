"""
URDF Generator

Programmatically generate URDF files from high-level robot descriptions.
Allows users to build custom robots by specifying links and joints.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from xml.dom import minidom


@dataclass
class LinkBuilder:
    """Builder for creating robot links"""
    name: str
    visual_geometry: Optional[Dict] = None
    visual_material: Optional[Dict] = None
    visual_origin: Optional[Dict] = None
    collision_geometry: Optional[Dict] = None
    collision_origin: Optional[Dict] = None
    mass: float = 1.0
    inertia: Optional[Dict] = None
    inertial_origin: Optional[Dict] = None

    def with_box_visual(self, size: List[float], color: List[float] = None, origin: Dict = None):
        """Add box visual geometry"""
        self.visual_geometry = {'type': 'box', 'size': size}
        if color:
            self.visual_material = {'color': color}
        if origin:
            self.visual_origin = origin
        return self

    def with_cylinder_visual(self, radius: float, length: float, color: List[float] = None, origin: Dict = None):
        """Add cylinder visual geometry"""
        self.visual_geometry = {'type': 'cylinder', 'radius': radius, 'length': length}
        if color:
            self.visual_material = {'color': color}
        if origin:
            self.visual_origin = origin
        return self

    def with_sphere_visual(self, radius: float, color: List[float] = None, origin: Dict = None):
        """Add sphere visual geometry"""
        self.visual_geometry = {'type': 'sphere', 'radius': radius}
        if color:
            self.visual_material = {'color': color}
        if origin:
            self.visual_origin = origin
        return self

    def with_mesh_visual(self, filename: str, scale: List[float] = None, origin: Dict = None):
        """Add mesh visual geometry"""
        self.visual_geometry = {'type': 'mesh', 'filename': filename, 'scale': scale or [1, 1, 1]}
        if origin:
            self.visual_origin = origin
        return self

    def with_collision(self, geometry: Dict, origin: Dict = None):
        """Add collision geometry (usually same as visual)"""
        self.collision_geometry = geometry
        if origin:
            self.collision_origin = origin
        return self

    def with_inertia(self, mass: float, inertia: Dict, origin: Dict = None):
        """Add inertial properties"""
        self.mass = mass
        self.inertia = inertia
        if origin:
            self.inertial_origin = origin
        return self


@dataclass
class JointBuilder:
    """Builder for creating robot joints"""
    name: str
    joint_type: str  # 'fixed', 'revolute', 'continuous', 'prismatic'
    parent: str
    child: str
    origin: Optional[Dict] = None
    axis: Optional[List[float]] = None
    limits: Optional[Dict] = None
    dynamics: Optional[Dict] = None

    @staticmethod
    def fixed(name: str, parent: str, child: str, origin: Dict = None):
        """Create a fixed joint"""
        return JointBuilder(name=name, joint_type='fixed', parent=parent, child=child, origin=origin)

    @staticmethod
    def revolute(name: str, parent: str, child: str, origin: Dict = None,
                 axis: List[float] = None, lower: float = -3.14, upper: float = 3.14,
                 effort: float = 100.0, velocity: float = 1.0):
        """Create a revolute (hinge) joint"""
        return JointBuilder(
            name=name,
            joint_type='revolute',
            parent=parent,
            child=child,
            origin=origin,
            axis=axis or [0, 0, 1],
            limits={'lower': lower, 'upper': upper, 'effort': effort, 'velocity': velocity}
        )

    @staticmethod
    def continuous(name: str, parent: str, child: str, origin: Dict = None,
                   axis: List[float] = None, effort: float = 100.0, velocity: float = 1.0):
        """Create a continuous rotation joint"""
        return JointBuilder(
            name=name,
            joint_type='continuous',
            parent=parent,
            child=child,
            origin=origin,
            axis=axis or [0, 0, 1],
            limits={'effort': effort, 'velocity': velocity}
        )

    @staticmethod
    def prismatic(name: str, parent: str, child: str, origin: Dict = None,
                  axis: List[float] = None, lower: float = 0.0, upper: float = 1.0,
                  effort: float = 100.0, velocity: float = 1.0):
        """Create a prismatic (sliding) joint"""
        return JointBuilder(
            name=name,
            joint_type='prismatic',
            parent=parent,
            child=child,
            origin=origin,
            axis=axis or [0, 0, 1],
            limits={'lower': lower, 'upper': upper, 'effort': effort, 'velocity': velocity}
        )


class URDFGenerator:
    """Generate URDF XML from programmatic robot descriptions"""

    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        self.links: List[LinkBuilder] = []
        self.joints: List[JointBuilder] = []

    def add_link(self, link: LinkBuilder):
        """Add a link to the robot"""
        self.links.append(link)
        return self

    def add_joint(self, joint: JointBuilder):
        """Add a joint to the robot"""
        self.joints.append(joint)
        return self

    def generate(self) -> str:
        """Generate URDF XML string"""
        robot = ET.Element('robot', name=self.robot_name)

        # Add links
        for link in self.links:
            link_elem = self._create_link_element(link)
            robot.append(link_elem)

        # Add joints
        for joint in self.joints:
            joint_elem = self._create_joint_element(joint)
            robot.append(joint_elem)

        # Pretty print XML
        xml_str = ET.tostring(robot, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent='  ')

    def _create_link_element(self, link: LinkBuilder) -> ET.Element:
        """Create XML element for a link"""
        link_elem = ET.Element('link', name=link.name)

        # Add visual
        if link.visual_geometry:
            visual = ET.SubElement(link_elem, 'visual')

            # Origin
            if link.visual_origin:
                self._add_origin(visual, link.visual_origin)

            # Geometry
            geometry = ET.SubElement(visual, 'geometry')
            self._add_geometry(geometry, link.visual_geometry)

            # Material
            if link.visual_material:
                material = ET.SubElement(visual, 'material', name='material')
                if 'color' in link.visual_material:
                    color = link.visual_material['color']
                    color_str = ' '.join(str(c) for c in color)
                    ET.SubElement(material, 'color', rgba=color_str)

        # Add collision
        if link.collision_geometry:
            collision = ET.SubElement(link_elem, 'collision')

            if link.collision_origin:
                self._add_origin(collision, link.collision_origin)

            geometry = ET.SubElement(collision, 'geometry')
            self._add_geometry(geometry, link.collision_geometry)

        # Add inertial
        if link.mass > 0 or link.inertia:
            inertial = ET.SubElement(link_elem, 'inertial')

            if link.inertial_origin:
                self._add_origin(inertial, link.inertial_origin)

            ET.SubElement(inertial, 'mass', value=str(link.mass))

            if link.inertia:
                ET.SubElement(inertial, 'inertia',
                             ixx=str(link.inertia.get('ixx', 0)),
                             ixy=str(link.inertia.get('ixy', 0)),
                             ixz=str(link.inertia.get('ixz', 0)),
                             iyy=str(link.inertia.get('iyy', 0)),
                             iyz=str(link.inertia.get('iyz', 0)),
                             izz=str(link.inertia.get('izz', 0)))

        return link_elem

    def _create_joint_element(self, joint: JointBuilder) -> ET.Element:
        """Create XML element for a joint"""
        joint_elem = ET.Element('joint', name=joint.name, type=joint.joint_type)

        # Parent and child
        ET.SubElement(joint_elem, 'parent', link=joint.parent)
        ET.SubElement(joint_elem, 'child', link=joint.child)

        # Origin
        if joint.origin:
            self._add_origin(joint_elem, joint.origin)

        # Axis
        if joint.axis:
            axis_str = ' '.join(str(a) for a in joint.axis)
            ET.SubElement(joint_elem, 'axis', xyz=axis_str)

        # Limits
        if joint.limits:
            limit_attrs = {}
            if 'lower' in joint.limits:
                limit_attrs['lower'] = str(joint.limits['lower'])
            if 'upper' in joint.limits:
                limit_attrs['upper'] = str(joint.limits['upper'])
            if 'effort' in joint.limits:
                limit_attrs['effort'] = str(joint.limits['effort'])
            if 'velocity' in joint.limits:
                limit_attrs['velocity'] = str(joint.limits['velocity'])

            if limit_attrs:
                ET.SubElement(joint_elem, 'limit', **limit_attrs)

        # Dynamics
        if joint.dynamics:
            dynamics_attrs = {}
            if 'damping' in joint.dynamics:
                dynamics_attrs['damping'] = str(joint.dynamics['damping'])
            if 'friction' in joint.dynamics:
                dynamics_attrs['friction'] = str(joint.dynamics['friction'])

            if dynamics_attrs:
                ET.SubElement(joint_elem, 'dynamics', **dynamics_attrs)

        return joint_elem

    def _add_origin(self, parent: ET.Element, origin: Dict):
        """Add origin element"""
        xyz = origin.get('xyz', [0, 0, 0])
        rpy = origin.get('rpy', [0, 0, 0])

        xyz_str = ' '.join(str(v) for v in xyz)
        rpy_str = ' '.join(str(v) for v in rpy)

        ET.SubElement(parent, 'origin', xyz=xyz_str, rpy=rpy_str)

    def _add_geometry(self, parent: ET.Element, geometry: Dict):
        """Add geometry element"""
        geom_type = geometry.get('type')

        if geom_type == 'box':
            size = geometry.get('size', [1, 1, 1])
            size_str = ' '.join(str(s) for s in size)
            ET.SubElement(parent, 'box', size=size_str)

        elif geom_type == 'cylinder':
            radius = geometry.get('radius', 0.5)
            length = geometry.get('length', 1.0)
            ET.SubElement(parent, 'cylinder', radius=str(radius), length=str(length))

        elif geom_type == 'sphere':
            radius = geometry.get('radius', 0.5)
            ET.SubElement(parent, 'sphere', radius=str(radius))

        elif geom_type == 'mesh':
            filename = geometry.get('filename', '')
            scale = geometry.get('scale', [1, 1, 1])
            scale_str = ' '.join(str(s) for s in scale)
            ET.SubElement(parent, 'mesh', filename=filename, scale=scale_str)


# Preset robot templates
class RobotPresets:
    """Pre-configured robot templates"""

    @staticmethod
    def simple_arm(name: str = "simple_arm") -> URDFGenerator:
        """Create a simple 2-DOF arm"""
        gen = URDFGenerator(name)

        # Base
        base = LinkBuilder("base_link").with_box_visual(
            [0.2, 0.2, 0.1],
            color=[0.5, 0.5, 0.5, 1.0]
        )
        gen.add_link(base)

        # Link 1
        link1 = LinkBuilder("link1").with_cylinder_visual(
            radius=0.05,
            length=0.5,
            color=[0.8, 0.2, 0.2, 1.0]
        )
        gen.add_link(link1)

        # Joint 1
        joint1 = JointBuilder.revolute(
            "joint1", "base_link", "link1",
            origin={'xyz': [0, 0, 0.05], 'rpy': [0, 0, 0]},
            axis=[0, 0, 1],
            lower=-1.57, upper=1.57
        )
        gen.add_joint(joint1)

        # Link 2
        link2 = LinkBuilder("link2").with_cylinder_visual(
            radius=0.04,
            length=0.4,
            color=[0.2, 0.8, 0.2, 1.0]
        )
        gen.add_link(link2)

        # Joint 2
        joint2 = JointBuilder.revolute(
            "joint2", "link1", "link2",
            origin={'xyz': [0, 0, 0.25], 'rpy': [0, 0, 0]},
            axis=[0, 1, 0],
            lower=-1.57, upper=1.57
        )
        gen.add_joint(joint2)

        return gen

    @staticmethod
    def mobile_base(name: str = "mobile_robot") -> URDFGenerator:
        """Create a differential drive mobile robot"""
        gen = URDFGenerator(name)

        # Base link
        base = LinkBuilder("base_link").with_box_visual(
            [0.4, 0.3, 0.1],
            color=[0.3, 0.3, 0.8, 1.0]
        )
        gen.add_link(base)

        # Left wheel
        left_wheel = LinkBuilder("left_wheel").with_cylinder_visual(
            radius=0.05,
            length=0.03,
            color=[0.2, 0.2, 0.2, 1.0],
            origin={'xyz': [0, 0, 0], 'rpy': [1.57, 0, 0]}
        )
        gen.add_link(left_wheel)

        # Left wheel joint
        left_joint = JointBuilder.continuous(
            "left_wheel_joint", "base_link", "left_wheel",
            origin={'xyz': [0, 0.15, 0], 'rpy': [0, 0, 0]},
            axis=[0, 1, 0]
        )
        gen.add_joint(left_joint)

        # Right wheel
        right_wheel = LinkBuilder("right_wheel").with_cylinder_visual(
            radius=0.05,
            length=0.03,
            color=[0.2, 0.2, 0.2, 1.0],
            origin={'xyz': [0, 0, 0], 'rpy': [1.57, 0, 0]}
        )
        gen.add_link(right_wheel)

        # Right wheel joint
        right_joint = JointBuilder.continuous(
            "right_wheel_joint", "base_link", "right_wheel",
            origin={'xyz': [0, -0.15, 0], 'rpy': [0, 0, 0]},
            axis=[0, 1, 0]
        )
        gen.add_joint(right_joint)

        return gen

    @staticmethod
    def quadruped_leg(name: str = "leg", parent: str = "base_link",
                     position: List[float] = None) -> Tuple[List[LinkBuilder], List[JointBuilder]]:
        """Create a simple quadruped leg (returns links and joints for assembly)"""
        position = position or [0, 0, 0]
        links = []
        joints = []

        # Hip link
        hip = LinkBuilder(f"{name}_hip").with_sphere_visual(
            radius=0.05,
            color=[0.3, 0.3, 0.3, 1.0]
        )
        links.append(hip)

        # Hip joint
        hip_joint = JointBuilder.revolute(
            f"{name}_hip_joint", parent, f"{name}_hip",
            origin={'xyz': position, 'rpy': [0, 0, 0]},
            axis=[0, 1, 0],
            lower=-0.5, upper=0.5
        )
        joints.append(hip_joint)

        # Upper leg
        upper = LinkBuilder(f"{name}_upper").with_cylinder_visual(
            radius=0.03,
            length=0.2,
            color=[0.6, 0.6, 0.6, 1.0]
        )
        links.append(upper)

        # Knee joint
        knee_joint = JointBuilder.revolute(
            f"{name}_knee_joint", f"{name}_hip", f"{name}_upper",
            origin={'xyz': [0, 0, -0.1], 'rpy': [0, 0, 0]},
            axis=[0, 1, 0],
            lower=-2.0, upper=0.0
        )
        joints.append(knee_joint)

        # Lower leg
        lower = LinkBuilder(f"{name}_lower").with_cylinder_visual(
            radius=0.02,
            length=0.2,
            color=[0.5, 0.5, 0.5, 1.0]
        )
        links.append(lower)

        # Ankle joint
        ankle_joint = JointBuilder.revolute(
            f"{name}_ankle_joint", f"{name}_upper", f"{name}_lower",
            origin={'xyz': [0, 0, -0.1], 'rpy': [0, 0, 0]},
            axis=[0, 1, 0],
            lower=-0.5, upper=0.5
        )
        joints.append(ankle_joint)

        return links, joints
