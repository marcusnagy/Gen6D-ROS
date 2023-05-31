
import cv2
from eval import visualize_intermediate_results
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft

import numpy as np

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp


def depth_pt(mat, pt):
    # the value is an integer and we assumed it might be in milimetre
    # and the depth matrix is arranged y by x, so assess it (y,x) not (x,y)
    val = mat[pt[1]][pt[0]]
    val /= 1000 # to metre
    return val


def image_pt(a):
    return [int(x) for x in (a/(a[-1]))[0:-1]]


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts


class PredictorArgs:
    config: str
    database: str
    num: int
    std: float


class Gen6DPredictor:
    def __init__(self, args:PredictorArgs):
        self.cfg = load_cfg(args.config)
        self.ref_database = parse_database_name(args.database)
        self.estimator = name2estimator[self.cfg['type']](self.cfg)
        self.estimator.build(self.ref_database, split_type='all')

        self.object_pts = get_ref_point_cloud(self.ref_database)
        self.object_bbox_3d = pts_range_to_bbox_pts(
            np.max(self.object_pts,0),
            np.min(self.object_pts,0)
        )
        self.object_center = np.mean(self.object_bbox_3d, axis=0)
        self.object_rotation = self.ref_database.rotation
        self.object_scale = self.ref_database.scale_ratio
        print("Object rotation: %s" % self.object_rotation)
        print("Object scale: %.3f" % self.object_scale)

        self.depth_matrix = None

        self.num = args.num
        self.std = args.std

        self.pose_init = None
        self.hist_pts = []

        self.publisher = rospy.Publisher(
            "gen6d/tetra/pose",
            PoseStamped,
            queue_size=1
        )

    def ros_image_topic_hook(self, msg: Image):
        color = CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")
        img = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        self.predict(img)

    def ros_depth_topic_hook(self, msg: Image):
        matrix = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_matrix = matrix

    def draw_axes(self, img, R, t, K, o=(0,0,0)):
        p = R @ o + t
        img = cv2.line(img, image_pt(K @ p), image_pt(K @ (R @ ((1,0,0) + o) + t)), (255,0,0), 5)
        img = cv2.line(img, image_pt(K @ p), image_pt(K @ (R @ ((0,1,0) + o) + t)), (0,255,0), 5)
        img = cv2.line(img, image_pt(K @ p), image_pt(K @ (R @ ((0,0,1) + o) + t)), (0,0,255), 5)
        return img

    def draw_preview(self, image, box_3d, inter, K):
        image = visualize_intermediate_results(image, K, inter, self.estimator.ref_info, box_3d)
        cv2.imshow("preview", image)
        cv2.waitKey(1)

    def predict(self, image: np.ndarray):
        img = image
        # generate a pseudo K
        h, w, _ = img.shape
        f=np.sqrt(h**2+w**2)
        K = np.asarray([[f,0,w/2],
                        [0,f,h/2],
                        [0,0,1]],np.float32)
        K = np.asarray([[607.91, 0     , 435.50],
                        [0     , 606.40, 222.05],
                        [0     , 0     , 1     ]], np.float32)
        # K_new = cv2.getOptimalNewCameraMatrix() ## Could possibly use this

        if self.depth_matrix is None:
            return

        if self.pose_init is not None:
            self.estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
        pose_pr, inter_results = self.estimator.predict(img, K, pose_init=self.pose_init)
        self.pose_init = pose_pr

        pts, _ = project_points(self.object_bbox_3d, pose_pr, K)

        self.hist_pts.append(pts)
        pts_ = weighted_pts(self.hist_pts, weight_num=self.num, std_inv=self.std)
        pose_ = pnp(self.object_bbox_3d, pts_, K)

        R = pose_[0:3,0:3]
        t = pose_[:,-1]

        # Acquire depth
        centre_img_p = image_pt(K @ (R @ self.object_center +t))
        px, py = centre_img_p
        rx = np.arange(px - 2, px + 2, 1)
        ry = np.arange(py - 2, py + 2, 1)

        depths = list()
        for i in rx:
            for j in ry:
                val = depth_pt(self.depth_matrix, (i,j))
                if val == 0.0:
                    continue
                depths.append(val)
        # if len(depths) == 0:
        #     # We wish to have depth to get a proper metric estimate
        #     return
        
        depth_val = np.mean(depths)
        # Offset the depth to move the value deeper
        depth_val += 0.05
        print(depth_val)

        out_pose = PoseStamped()
        out_pose.header.stamp = rospy.Time(0)
        out_pose.header.frame_id = "skiros:TransformationPose-49"

        s = (1/(10 * self.object_scale))
        p = (R @ self.object_center + t) * s

        # Rescale depth distance
        p = p * depth_val / p[2]
        out_pose.pose.position.x = p[0]
        out_pose.pose.position.y = p[1]
        out_pose.pose.position.z = p[2]
        # out_pose.pose.position.x = p[2]
        # out_pose.pose.position.y = -p[0]
        # out_pose.pose.position.z = -p[1]
        
        padded = np.eye(4)
        padded[0:3,0:3] = R
        qR = tft.quaternion_from_matrix(padded)
        out_pose.pose.orientation.x = qR[0]
        out_pose.pose.orientation.y = qR[1]
        out_pose.pose.orientation.z = qR[2]
        out_pose.pose.orientation.w = qR[3]

        self.publisher.publish(out_pose)

        self.draw_axes(img, R, t, K, self.object_center)
        self.draw_preview(img, self.object_bbox_3d, inter_results, K)


if __name__ == "__main__":
    rospy.init_node("gen_6d_tracker", anonymous=True)

    print("Initializing")
    args = PredictorArgs()
    args.config = 'configs/gen6d_pretrain.yaml'
    args.database = "custom/tetra"
    args.num = 5
    args.std = 2.5
    predictor = Gen6DPredictor(args)

    print("Subscribing")
    subscriber = rospy.Subscriber(
        "/realsense/rgb/image_raw",
        Image,
        predictor.ros_image_topic_hook,
        queue_size=1
    )
    subscriber = rospy.Subscriber(
        "/realsense/aligned_depth_to_color/image_raw",
        Image,
        predictor.ros_depth_topic_hook,
        queue_size=1
    )

    rate = rospy.Rate(2.0)

    while not rospy.is_shutdown():
        rate.sleep()

    subscriber.unregister()
