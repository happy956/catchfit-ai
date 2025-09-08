from shapely.geometry import LineString, Polygon, MultiLineString
from modules import *
from math import dist
import numpy as np
import cv2

class BodyPart:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    @classmethod
    def shoulder(cls):
        return ['shoulder', cls.LEFT_SHOULDER, cls.RIGHT_SHOULDER]
    @classmethod
    def waist(cls):
        return ['waist', cls.LEFT_SHOULDER, cls.RIGHT_SHOULDER, cls.LEFT_HIP, cls.RIGHT_HIP]

    @classmethod ### 6
    def left_lower_arm(cls):
        return ['arm', cls.LEFT_ELBOW, cls.LEFT_WRIST]
    @classmethod ### 10
    def left_upper_arm(cls):
        return ['arm', cls.LEFT_ELBOW, cls.LEFT_SHOULDER]
    @classmethod ### 15
    def right_lower_arm(cls):
        return ['arm', cls.RIGHT_ELBOW, cls.RIGHT_WRIST]
    @classmethod ### 19
    def right_upper_arm(cls):
        return ['arm', cls.RIGHT_ELBOW, cls.RIGHT_SHOULDER]
    
    @classmethod ### 7
    def left_lower_leg(cls):
        return ['leg', cls.LEFT_ANKLE, cls.LEFT_KNEE]
    @classmethod ### 11
    def left_upper_leg(cls):
        return ['leg', cls.LEFT_KNEE, cls.LEFT_HIP]
    @classmethod ### 16
    def right_lower_leg(cls):
        return ['leg', cls.RIGHT_ANKLE, cls.RIGHT_KNEE]
    @classmethod ### 20
    def right_upper_leg(cls):
        return ['leg', cls.RIGHT_KNEE, cls.RIGHT_HIP]

class BodyPartValue:
    def __init__(self):
        self.left_upper_arm_width: float = 0.0
        self.right_upper_arm_width: float = 0.0
        
        self.left_lower_arm_width: float = 0.0
        self.right_lower_arm_width: float = 0.0
        
        self.left_upper_leg_width: float = 0.0
        self.right_upper_leg_width: float = 0.0
        
        self.left_lower_leg_width: float = 0.0
        self.right_lower_leg_width: float = 0.0
        
        self.shoulder_width: float = 0.0
        self.waist_width: float = 0.0
    
    def get_data(self):
        return [
            self.left_upper_arm_width,
            self.right_upper_arm_width,
            self.left_lower_arm_width,
            self.right_lower_arm_width,
            
            self.left_upper_leg_width,
            self.right_upper_leg_width,
            self.left_lower_leg_width,
            self.right_lower_leg_width,
            
            self.shoulder_width,
            self.waist_width
        ]

class Agent:
    def __init__(self, args):
        self.args = args
        self.device = self.args['device']
        self.pose_estimator = PoseEstimator(self.args)
        self.bbox_detector = BBOXDetector(self.args)
        self.segmentator = Segmentator(self.args)
        self.outliner = OutLiner(self.args)
        
        self.img = None
        self.bbox = None
    
    def get_keypoints(self, img=None):
        if img is not None:
            xyxy, keypoint = self.pose_estimator.get_keypoints(img)
        else:
            xyxy, keypoint = self.pose_estimator.get_keypoints(self.img)
        return xyxy, keypoint
    
    def get_bbox(self):
        bbox = self.bbox_detector.get_bbox(self.img)
        return bbox
    
    def get_segment(self):
        segment = self.segmentator.get_segment(self.img)
        return segment
    
    def get_outline_mask(self, img=None):
        if img is not None:
            outline_mask = self.outliner.get_outline_mask(img, self.bbox)
        else:
            outline_mask = self.outliner.get_outline_mask(self.img, self.bbox)
        return outline_mask
    
    def get_outline(self, img=None):
        outline_mask = self.get_outline_mask(img)
        contour = self.get_contour(outline_mask).reshape(-1, 2)
        outline = [c.tolist() for c in contour]
        return outline
    
    def set_img(self, img):
        self.img = img
        self.xyxy, self.keypoints = self.get_keypoints()
        self.bbox = self.get_bbox()
        self.segment = self.get_segment()
        self.outline = self.get_outline_mask()
    
    def get_euclidean_length(self, x1, y1, x2, y2):
        pixel_length = dist((x1, y1), (x2, y2))
        return pixel_length
    
    def get_middle_point(self, x1, y1, x2, y2):
        x, y = np.mean([x1, x2]), np.mean([y1, y2])
        return (x, y)
    
    def get_vertical_line(self, x1, y1, x2, y2, centroid, xyxy):
        origin_m = (y2 - y1) / (x2 - x1)
        m = - 1 / origin_m
        d_x, d_y = centroid
        b = d_y - m * d_x
        
        min_x, max_x = xyxy[0], xyxy[2]
        dummy_x = np.arange(min_x, max_x, 1)
        vertical_line = LineString([(x, m * x + b) for x in dummy_x])
        return vertical_line

    def get_horizontal_line(self, x1, y1, x2, y2, centroid, xyxy):
        m = (y2 - y1) / (x2 - x1)
        d_x, d_y = centroid
        b = d_y - m * d_x
        
        min_x, max_x = xyxy[0], xyxy[2]
        dummy_x = np.arange(min_x, max_x, 1)
        horizontal_line = LineString([(x, m * x + b) for x in dummy_x])
        return horizontal_line
    
    def get_intersection_line(self, polygon, vertical_line):
        return vertical_line.intersection(polygon)
    
    def get_contour(self, mask):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 1:
            contour = None
            max_contour_area = -np.inf
            for c in contours:
                area = cv2.contourArea(c)
                if area > max_contour_area:
                    contour = c
                    max_contour_area = area
        else:
            contour = contours[0]
        return contour
    
    def get_vertical_width(self, label, target_points, mask, xyxy):
        x1, y1, x2, y2 = target_points
        mask_ = np.where(mask == label, 255, 0).astype(np.uint8)
        _, thresh = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY)
        contour = self.get_contour(thresh).reshape(-1, 2)
        if len(contour) < 5:
            return 0
        polygon = Polygon(contour)
        centroid = list(polygon.centroid.coords)[0]
        vertical_line = self.get_vertical_line(x1, y1, x2, y2, centroid, xyxy)
        
        res_vertical_line = self.get_intersection_line(polygon, vertical_line)
        if res_vertical_line.is_empty:
            raise NotImplementedError
        
        vertical_points = list(res_vertical_line.coords)
        v_x1, v_y1, v_x2, v_y2 = vertical_points[0][0], vertical_points[0][1], vertical_points[-1][0], vertical_points[-1][1]
        value = self.get_euclidean_length(v_x1, v_y1, v_x2, v_y2)
        return value
    
    def get_waist_width(self, points):
        left_shoulder, right_shoulder, left_hip, right_hip = points
        left_shoulder_x, left_shoulder_y = left_shoulder
        right_shoulder_x, right_shoulder_y = right_shoulder
        left_hip_x, left_hip_y = left_hip
        right_hip_x, right_hip_y = right_hip
        
        left_waist_y = left_shoulder_y + (left_hip_y - left_shoulder_y) * 0.6
        right_waist_y = right_shoulder_y + (right_hip_y - right_shoulder_y) * 0.6
        
        centroid = (left_hip_x, left_waist_y)
        horizontal_line = self.get_horizontal_line(right_hip_x, right_waist_y, left_hip_x, left_waist_y, centroid, self.xyxy)
        
        ### arm mask
        mask = np.where(np.isin(self.segment, [5, 6, 10, 14, 15, 19]), 255, 0)
        new_outline = np.clip(self.outline - mask, 0, 255).astype('uint8')
        contour = self.get_contour(new_outline).reshape(-1, 2)
        polygon = Polygon(contour)
        
        res_vertical_line = self.get_intersection_line(polygon, horizontal_line)
        
        if isinstance(res_vertical_line, MultiLineString):
            res_vertical_line = res_vertical_line.geoms[1]
        
        if res_vertical_line.is_empty:
            raise NotImplementedError
        else:
            points = list(res_vertical_line.coords)
            v_x1, v_y1, v_x2, v_y2 = points[0][0], points[0][1], points[-1][0], points[-1][1]
            value = self.get_euclidean_length(v_x1, v_y1, v_x2, v_y2)
            return value
    
    def calculate(self):
        results = BodyPartValue()
        for label in np.unique(self.segment):
            if label == 6:
                indexes = BodyPart.left_lower_arm()
            elif label == 10:
                indexes = BodyPart.left_upper_arm()
            elif label == 15:
                indexes = BodyPart.right_lower_arm()
            elif label == 19:
                indexes = BodyPart.right_upper_arm()
            
            elif label == 7:
                indexes = BodyPart.left_lower_leg()
            elif label == 11:
                indexes = BodyPart.left_upper_leg()
            elif label == 16:
                indexes = BodyPart.right_lower_leg()
            elif label == 20:
                indexes = BodyPart.right_upper_leg()
            else: continue
            
            category, *index = indexes[0], [part for part in indexes[1:]]
            target_points = self.keypoints[index].reshape(-1)
            if 0 in target_points: continue
            
            if category == 'arm' or category == 'leg':
                value = self.get_vertical_width(label, target_points, self.segment, self.xyxy)
                if label == 6:
                    results.left_lower_arm_width = value
                elif label == 10:
                    results.left_upper_arm_width = value
                elif label == 15:
                    results.right_lower_arm_width = value
                elif label == 19:
                    results.right_upper_arm_width = value
                
                elif label == 7:
                    results.left_lower_leg_width = value
                elif label == 11:
                    results.left_upper_leg_width = value
                elif label == 16:
                    results.right_lower_leg_width = value
                elif label == 20:
                    results.right_upper_leg_width = value
        
        ### shoulder
        indexes = BodyPart.shoulder()
        category, *index = indexes[0], [part for part in indexes[1:]]
        target_points = self.keypoints[index].reshape(-1)
        if 0 in target_points: pass
        else:
            results.shoulder_width = self.get_euclidean_length(*target_points)
        
        ### waist
        indexes = BodyPart.waist()
        category, *index = indexes[0], [part for part in indexes[1:]]
        target_points = self.keypoints[index][0]
        if 0 in target_points: pass
        else:
            results.waist_width = self.get_waist_width(target_points)
        
        return results.get_data()
    
    def get_outline_error(self, img1):
        prev_outline = self.get_outline_mask(img1)
        prev_contour = self.get_contour(prev_outline)
        curr_contour = self.get_contour(self.outline)
        prev_area = cv2.contourArea(prev_contour)
        curr_area = cv2.contourArea(curr_contour)
        error = (curr_area - prev_area) / (prev_area) * 100
        return error
    
    def get_keypoints_error(self, img1):
        H, W, C = img1.shape
        _, prev_keypoints = self.get_keypoints(img1)
        curr_keypoints = self.keypoints
        
        error_sum = 0
        for p, c in zip(prev_keypoints, curr_keypoints):
            p_x, p_y = p
            c_x, c_y = c
            dist = self.get_euclidean_length(p_x, p_y, c_x, c_y)
            error_sum += dist
        error = error_sum / (W * len(prev_keypoints)) * 100
        return error
        
# if __name__ == '__main__':
#     with open('config.yaml', 'r') as f:
#         args = yaml.safe_load(f)
#     args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
#     agent = Agent(args)
    
#     img = cv2.imread('data/dataset1/IMG_6849.jpg')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     agent.set_img(img)
#     agent.get_width()