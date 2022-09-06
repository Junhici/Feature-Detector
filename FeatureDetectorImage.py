import cv2

img1 = cv2.imread('Carte/2 Cuori.jpg')
img2 = cv2.imread('ImagesTrain/3Cy.jpg')

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance > 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
print(len(good))

cv2.imshow('img3', img3)
cv2.waitKey(0)