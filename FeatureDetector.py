import os

import cv2

path = 'ImagesQuery'
orb = cv2.ORB_create(nfeatures=1000)

images = []
class_names = []
my_list = os.listdir(path)

for cl in my_list:
    img_cur = cv2.imread(f'{path}/{cl}', 0)
    images.append(img_cur)
    class_names.append(os.path.splitext(cl)[0])


def find_desc(images):
    desc_list = []
    for img in images:
        kp, desc = orb.detectAndCompute(img, None)
        desc_list.append(desc)
    return desc_list


def find_ID(img, desc_list, thres=15):
    kp2, desc2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    match_list = []
    final_val = -1
    try:
        for desc in desc_list:
            matches = bf.knnMatch(desc, desc2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            match_list.append(len(good))
        print(match_list)
    except:
        pass

    if len(match_list) != 0:
        if max(match_list) > thres:
            final_val = match_list.index(max(match_list))
    return final_val


desc_list = find_desc(images)

cam = cv2.VideoCapture(0)

while True:
    success, img2 = cam.read()
    img_original = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = find_ID(img2, desc_list)
    if id != -1:
        cv2.putText(img_original, class_names[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        ####Togliere commento per vedere processo

    kp1, des1 = orb.detectAndCompute(img2, None)
    kp2, des2 = orb.detectAndCompute(images[id], None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    game_image = cv2.cvtColor(cv2.resize(images[id], (960, 840)), cv2.COLOR_GRAY2BGR)
    img3 = cv2.drawMatchesKnn(img_original, kp1, game_image, kp2, good, None, flags=2)
    ims = cv2.resize(img3, (960, 540))
    cv2.putText(ims, str(len(good)), (50, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('img3', ims)
    cv2.imshow('img_original', img_original)
    cv2.waitKey(1)
