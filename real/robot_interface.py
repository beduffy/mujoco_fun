from typing import Optional, List

try:
    import rclpy  # type: ignore
    ROS2_AVAILABLE = True
except Exception:
    ROS2_AVAILABLE = False


class RobotInterface:
    def __init__(self, mode: str = "sim"):
        self.mode = mode
        if mode == "ros2" and not ROS2_AVAILABLE:
            raise RuntimeError("ROS2 not available in this environment")
        # Stub: in real usage, initialize ROS2 node, publishers/subscribers

    def send_joint_positions(self, joint_positions: List[float]):
        if self.mode == "sim":
            # no-op; simulated via pybullet elsewhere
            return
        # Stub: publish to robot controller

    def get_joint_positions(self) -> List[float]:
        if self.mode == "sim":
            return []
        # Stub: query robot state
        return []