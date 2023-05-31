
import json
import cv2
from eval import visualize_intermediate_results
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from std_msgs.msg import String as StringMessage

import numpy as np

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp


CUSTOM_OBJECTS = [
    "custom/tetra"
]


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

        self.num = args.num
        self.std = args.std
        self.depth_radius = 20

        self.pose_init = None
        self.hist_pts = []

        self.publisher_topic = "gen6d/{}/pose".format(
            args.database
        )
        self.publisher = rospy.Publisher(
            self.publisher_topic,
            PoseStamped,
            queue_size=1
        )
        print("- '%s' publishes to '%s" % (args.database, self.publisher_topic))

    def clear_cache(self):
        self.pose_init = None
        self.hist_pts = []

    def draw_axes(self, img, R, t, K, o=(0,0,0)):
        p = R @ o + t
        img = cv2.line(img, image_pt(K @ p), image_pt(K @ (R @ ((1,0,0) + o) + t)), (255,0,0), 5)
        img = cv2.line(img, image_pt(K @ p), image_pt(K @ (R @ ((0,1,0) + o) + t)), (0,255,0), 5)
        img = cv2.line(img, image_pt(K @ p), image_pt(K @ (R @ ((0,0,1) + o) + t)), (0,0,255), 5)
        return img

    def draw_preview(self, image, box_3d, inter, K):
        image = visualize_intermediate_results(
            image, 
            K, 
            inter, 
            self.estimator.ref_info, 
            box_3d
        )
        return image

    def predict(self, colour: np.ndarray, depth: np.ndarray):
        img = colour
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

        if depth is None:
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
        rx = np.arange(
            px - self.depth_radius, 
            px + self.depth_radius, 
            1
        )
        ry = np.arange(
            py - self.depth_radius, 
            py + self.depth_radius, 
            1
        )

        depths = list()
        for i in rx:
            for j in ry:
                val = depth_pt(depth, (i,j))
                if val == 0.0:
                    continue
                depths.append(val)
        
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
        
        padded = np.eye(4)
        padded[0:3,0:3] = R
        qR = tft.quaternion_from_matrix(padded)
        out_pose.pose.orientation.x = qR[0]
        out_pose.pose.orientation.y = qR[1]
        out_pose.pose.orientation.z = qR[2]
        out_pose.pose.orientation.w = qR[3]

        self.publisher.publish(out_pose)

        preview = self.draw_axes(img, R, t, K, self.object_center)
        preview = self.draw_preview(preview, self.object_bbox_3d, inter_results, K)

        return preview


def initialize_predictor(database) -> Gen6DPredictor:
    args = PredictorArgs()
    args.config = 'configs/gen6d_pretrain.yaml'
    args.database = database
    args.num = 5
    args.std = 2.5
    return Gen6DPredictor(args)


class Coordinator():
    def __init__(self) -> None:
        self.colour_image = None
        self.depth_image = None

        self.predictors = dict()
        self.should_predict = dict()

    def initialize(self):
        for database in CUSTOM_OBJECTS:
            self.should_predict[database] = False
            self.predictors[database] = initialize_predictor(database)

    def ros_image_topic_hook(self, msg: Image):
        color = CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")
        img = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        self.colour_image = np.asarray(img)

    def ros_depth_topic_hook(self, msg: Image):
        matrix = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_image = matrix

    def ros_command_topic_hook(self, msg: StringMessage):
        command = json.loads(msg.data)
        
        should_predict = command["predict"]
        predict_id = command["object"]

        print("{} predicting {}".format(
            "Started" if should_predict else "Stopped",
            predict_id
        ))

        if not predict_id in self.predictors:
            print("Error - Unknown database: '%s'" % predict_id)
            return

        if should_predict:
            self.should_predict[predict_id] = True
        else:
            self.should_predict[predict_id] = False
            self.predictors[predict_id].clear_cache()

    def start_listen(self):
        self.command_listener = rospy.Subscriber(
            "/gen6d/command",
            StringMessage,
            self.ros_command_topic_hook,
            queue_size=1
        )
        self.rgb_subscriber = rospy.Subscriber(
            "/realsense/rgb/image_raw",
            Image,
            self.ros_image_topic_hook,
            queue_size=1
        )
        self.depth_subscriber = rospy.Subscriber(
            "/realsense/aligned_depth_to_color/image_raw",
            Image,
            self.ros_depth_topic_hook,
            queue_size=1
        )

    def stop_listen(self):
        self.rgb_subscriber.unregister()
        self.depth_subscriber.unregister()
        self.command_listener.unregister()

    def tick(self):
        image = None
        for key in self.predictors:
            if not self.should_predict[key]: 
                continue
            predictor: Gen6DPredictor = self.predictors[key]
            image = predictor.predict(self.colour_image, self.depth_image)
        
        if not image is None:
            cv2.imshow("preview", image)
            cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("gen_6d_tracker", anonymous=True)

    print("Initializing")
    coordinator = Coordinator()
    coordinator.initialize()

    print("Subscribing")
    coordinator.start_listen()
    rate = rospy.Rate(6.0)

    while not rospy.is_shutdown():
        coordinator.tick()
        rate.sleep()

    print("Good bye")
    coordinator.stop_listen()
