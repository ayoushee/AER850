import cv2
import numpy as np

image_path = "motherboard_image.JPEG"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

edges = cv2.Canny(thresholded, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated_edges = cv2.dilate(edges, kernel, iterations=2)

contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(gray)

cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

extracted_pcb = cv2.bitwise_and(image, image, mask=mask)

background = np.zeros_like(image)
final_output = cv2.addWeighted(extracted_pcb, 1, background, 0, 0)

cv2.imwrite("thresholded_image.jpg", thresholded)
cv2.imwrite("dilated_edges.jpg", dilated_edges)
cv2.imwrite("extracted_pcb.jpg", final_output)
    
#cv2.imshow("Original Image", image)
#cv2.imshow("Thresholded Image", thresholded)
#cv2.imshow("Dilated Edges", dilated_edges)
#cv2.imshow("Final Extracted PCB", final_output)
#cv2.waitKey(0)
#cv2.destroyAllWindows()