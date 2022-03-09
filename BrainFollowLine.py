import numpy as np
from pyrobot.brain import Brain

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class BrainFollowLine(Brain):
 
  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0

  NO_ERROR = 0

  def setup(self):
    self.image_sub = rospy.Subscriber("/image",Image,self.callback)
    self.bridge = CvBridge()

  def callback(self,data):
    self.rosImage = data

  def destroy(self):
    cv2.destroyAllWindows()

  def obtain_function(self, image):
    image = cv2.resize(image, (60, 60))
    image = np.array(
      np.apply_along_axis(lambda x: x if (x <= np.array([255, 10, 10])).all() else [255, 255, 255], 2, image),
      dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (320, 240))
    row_centres = []
    rows = []
    for i in range(image.shape[0]):
      if np.any(image[i] < 100):
        rowcentre = np.median(np.argwhere(image[i] < 100))
        row_centres.append(rowcentre)
        rows.append(i)

    imagec = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
    p = np.poly1d(np.lib.polynomial.polyfit(rows, row_centres, 2))
    for i in rows:
      cv2.circle(imagec, (int(round(p(i))), i), 0, (255, 255, 0), thickness=6)
    cv2.imshow("Stage Camera Image", imagec)
    cv2.waitKey(1)
    return p, image.shape[1]

  def step(self):
    # take the last image received from the camera and convert it into
    # opencv format
    try:
      cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
    except CvBridgeError as e:
      print(e)

    # display the robot's camera's image using opencv
    # cv2.imshow("Stage Camera Image", cv_image)
    # cv2.waitKey(1)

    # write the image to a file, for debugging etc.
    # cv2.imwrite("debug-capture.png", cv_image)

    # convert the image into grayscale
    imageGray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # determine the robot's deviation from the line.
    try:
      p, d = self.obtain_function(cv_image)
    except Exception as e:
      print(e)
      print('Line not found')
      p = None

    ps = []
    if p is not None:
      for i in range(imageGray.shape[0]):
        ps.append(p(i) - d/2)

      mean_err = np.mean(ps)/100
      self.move(abs(1 - min(1, abs(mean_err))), - max(min(4,mean_err*4), -4))
    else:
      exit()


def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  return BrainFollowLine('BrainFollowLine', engine)
