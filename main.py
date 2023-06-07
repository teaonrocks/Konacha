import cv2
import numpy as np

img = cv2.imread("27.jpeg")

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
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h
            if ratio >= 0.8 and ratio <= 1.2:
                # img = cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
                allx.append(x)
                ally.append(y)
                widths.append(w)
                heights.append(h)

averageWidth = sum(widths) / len(widths)
averageHeight = sum(heights) / len(heights)

topleft = (
    min(allx) - int(averageWidth) + int(averageWidth / 2),
    min(ally) - int(averageHeight) + int(averageWidth / 2),
)
bottomright = (
    max(allx) + int(averageWidth) + int(averageWidth / 2),
    max(ally) + int(averageHeight) + int(averageHeight / 2),
)
cv2.rectangle(img, topleft, bottomright, (0, 255, 0), 5)

crop_img = img[topleft[1] : bottomright[1], topleft[0] : bottomright[0]]


# def part2(img):
#     im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # threshold gray image to b and w
#     ret, thresh2 = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)

#     # dilate and erode image
#     kernel = np.ones((5, 5), np.uint8)
#     img_dilation = cv2.dilate(thresh2, kernel, iterations=2)

#     kernel = np.ones((10, 10), np.uint8)
#     img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

#     # detect corners
#     gray = np.float32(img_erosion)
#     dst = cv2.cornerHarris(gray, 5, 19, 0.07)

#     dst = cv2.dilate(dst, None, iterations=2)

#     img[dst > 0.01 * dst.max()] = [0, 0, 255]
#     return img


# part2img = part2(crop_img)
# cv2.imshow("img2.png", part2img)
# cv2.waitKey()
# cv2.destroyWindow("img2.png")
cv2.imshow("uncropped", img)
cv2.imwrite("uncropped.jpg", img)
cv2.waitKey()
cv2.destroyWindow("uncropped")
cv2.imshow("cropped", crop_img)
cv2.imwrite("crop_img.jpg", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
