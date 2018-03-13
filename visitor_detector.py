#coding: utf-8

from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
import numpy as np
import freenect
import pickle
import scipy
import cv2
import os

def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

def of_like_map(value, start1, end1, start2, end2):
    result = start2 + (end2 - start2) * (value - start1) / (end1 - start1)
    return result

class Visitor_detector:
    """
    Calculate the mahalanobis distance of the current frame
    using normal time as the population.
    """
    def __init__(self, horizonY):
        if "train.pickle" in os.listdir("."):
            with open("train.pickle", "br") as f:
                self.train_depth = pickle.load(f)
                
        else:
            train_frame = 1000
            train_depth = [get_depth()[horizonY] for _ in tqdm(range(train_frame))]
            train_depth = np.array(train_depth)

            noise = 0.00001*np.random.rand(640, len(train_depth))
            
            sx = np.cov(train_depth.T + noise)
            self.sx = scipy.linalg.inv(sx)

            self.mean = train_depth.mean()
            
            """
            with open("train.pickle", "bw") as f:
                pickle.dump(train_depth)
            """

    def get_mahalanobs(self, new_depth):
        new_depth = new_depth.reshape((1, -1)).astype("float64")
        distance = mahalanobis(new_depth, self.mean, self.sx)

        return distance
    
if __name__ == "__main__":
    depth = get_depth()
    height, width = get_depth().shape

    horizonY = height/2

    detector = Visitor_detector(horizonY)
    
    while True:
        depth = get_depth()
        img = depth
        new_depth = depth[horizonY]

        distance = detector.get_mahalanobs(new_depth)
        print "mahalanobis distance:", distance

        radar = np.zeros(depth.shape)
        for i in range(width):
            radar[new_depth[i], i] = 1

        cv2.imshow("Radar", radar)
        cv2.line(img, (0, horizonY), (width, horizonY), (0), 2)
        cv2.imshow("Depth", img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.destroyAllWindows()
