from typing import Any, Optional
import time
import signal

import rclpy
import rclpy.executors
import rclpy.logging


from robomaster_ros.client import RoboMasterROS


def main(args: Any = None) -> None:
    rclpy.init(args=args)
    executor = rclpy.executors.MultiThreadedExecutor()
    # TODO(Jerome): currently not triggered by ctrl+C
    # rclpy.get_default_context().on_shutdown(...)

    node: Optional[RoboMasterROS] = None
    while rclpy.ok():
        try:
            node = RoboMasterROS(executor=executor)
        except KeyboardInterrupt:
            break


    def shutdown(sig, _):
        if node:
            if rclpy.ok():
                node.abort()
                rclpy.spin_once(node, executor=executor, timeout_sec=0.1)
            if rclpy.ok():
                node.stop()
                rclpy.spin_once(node, executor=executor, timeout_sec=0.1)
        else:
            raise KeyboardInterrupt
        rclpy.try_shutdown()

    signal.signal(signal.SIGINT, shutdown)
    rclpy.try_shutdown()
    if node:
        node.destroy_node()
