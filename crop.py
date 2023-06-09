import cv2
import numpy as np

img = cv2.imread("images/30.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur image
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# do otsu threshold on gray image
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
inverseThresh = cv2.bitwise_not(thresh)
contours, hierarchy = cv2.findContours(inverseThresh, 1, 2)
print("Number of contours detected:", len(contours))

allx = []
ally = []
widths = []
heights = []
filteredCnts = []

for cnt in contours:
    x1, y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        if cv2.contourArea(cnt) > 1000:
            filteredCnts.append(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if ratio >= 0.8 and ratio <= 1.2:
                img = cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
                allx.append(x)
                ally.append(y)
                widths.append(w)
                heights.append(h)

averageWidth = sum(widths) / len(widths)
averageHeight = sum(heights) / len(heights)

topleft = (
    min(allx),
    min(ally),
)
bottomright = (
    max(allx) + int(averageWidth),
    max(ally) + int(averageHeight),
)
cv2.rectangle(img, topleft, bottomright, (0, 255, 0), 5)

crop_img = img[topleft[1] : bottomright[1], topleft[0] : bottomright[0]]


cv2.imshow("uncropped", img)
cv2.waitKey()
cv2.destroyWindow("uncropped")
cv2.imshow("cropped", crop_img)
cv2.imwrite("crop_img.jpg", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
