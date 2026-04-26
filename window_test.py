import cv2
import numpy as np

print("Opening window...")

img = np.zeros((500,500,3), dtype=np.uint8)

cv2.imshow("Test Window", img)

cv2.waitKey(5000)  # show for 5 seconds

cv2.destroyAllWindows()

print("Window closed")