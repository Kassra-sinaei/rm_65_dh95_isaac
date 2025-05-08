import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

class TfListener(Node):
    def __init__(self):
        super().__init__('tf_listener')
        # Subscribe to /tf, which publishes TFMessage (a list of TransformStamped)
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

    def tf_callback(self, msg: TFMessage):
        for transform in msg.transforms:  # each is a TransformStamped
            self.print_transform(transform)

    def print_transform(self, t: TransformStamped):
        header = t.header
        trans = t.transform.translation
        rot   = t.transform.rotation
        self.get_logger().info(
            f"[{header.stamp.sec}.{header.stamp.nanosec:09d}] "
            f"{header.frame_id} → {t.child_frame_id}  |  "
            f"pos = ({trans.x:.3f}, {trans.y:.3f}, {trans.z:.3f})  |  "
            f"rot = ({rot.x:.3f}, {rot.y:.3f}, {rot.z:.3f}, {rot.w:.3f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = TfListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()