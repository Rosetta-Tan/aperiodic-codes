import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from timeit import default_timer as timer

start = timer()
image = cv2.imread('/Users/yitan/Desktop/psi_quad_gen8.png', 0)
# image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2)) 
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(image, threshold1=100, threshold2=200)
edges = cv2.Canny(image, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Iterate over contours and approximate them to polygons
polygons = []
for cnt in contours:
    # Approximate the contour to a polygon
    epsilon = 0.08 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # If the polygon has a reasonable number of sides, consider it
    if len(approx) == 4:
        # Draw the polygon on the original image (optional)
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)

        polygon = []
        # Extract and print the coordinates
        vertices = approx.reshape(-1, 2)
        polygon.append(vertices)
        print(f"Polygon with {len(approx)} sides, vertices: {vertices}")
        polygons.append(polygon)


print('len polygons: ', len(polygons))
# # Display the image with drawn polygons (optional)
# cv2.imshow('Polygons Detected', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Display the original and edge-detected images
# plt.subplot(121), plt.imshow(image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()
end = timer()
print('time: ', end - start)