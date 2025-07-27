import cv2
import numpy as np

# Create a simple image
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.putText(img, 'OpenCV Test', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Show image
cv2.imshow('Test Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
