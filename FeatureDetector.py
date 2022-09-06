# Import modules
import os
import cv2

# Define path
path = 'ImagesQuery'

# Create an ORB (Oriented BRIEF)
orb = cv2.ORB_create(nfeatures=1000)

# Define a list of images and class names
images = []
class_names = []

# Find samples in path
my_list = os.listdir(path)

# For each class in path
for cl in my_list:
    # Read current image
    img_cur = cv2.imread(f'{path}/{cl}', 0)
    
    # Append image
    images.append(img_cur)
    
    # Append filename
    class_names.append(os.path.splitext(cl)[0])

# Method to find descriptors
def find_desc(images):
    # Define descriptors list
    desc_list = []
    
    # For each image in images
    for img in images:
        # Define keypoints and descriptors in image
        kp, desc = orb.detectAndCompute(img, None)
        
        # Append descriptor in descriptors list
        desc_list.append(desc)
    return desc_list

# Method to find IDs
def find_ID(img, desc_list, thres=15):
    # Define keypoints and descriptors
    kp2, desc2 = orb.detectAndCompute(img, None)
    
    # Create a brute force desciptor matcher
    bf = cv2.BFMatcher()
    
    # Define a match list
    match_list = []
    final_val = -1
    try:
        # For each descriptor in descriptor list
        for desc in desc_list:
            # Find the best matches for descriptors
            matches = bf.knnMatch(desc, desc2, k=2)
            
            # Define good matches list
            good = []
            
            # For each m and n in matches
            for m, n in matches:
                # If m distance is less than 0.75 multiplied for n distance
                if m.distance < 0.75 * n.distance:
                    # Append M in good list
                    good.append([m])
            # Append length of good in match list
            match_list.append(len(good))
        print(match_list)
    except:
        pass

    # If there is a match in match list
    if len(match_list) != 0:
        # If maximum of the match list is higher than threshold
        if max(match_list) > thres:
            # Final value is equals to the index of match list maximum value
            final_val = match_list.index(max(match_list))
    return final_val

# Find descriptor in image
desc_list = find_desc(images)

# Configure camera
cam = cv2.VideoCapture(0)

while True:
    # Get camera's frame
    success, img2 = cam.read()
    
    # Copy image
    img_original = img2.copy()
    
    # Convert image from BGR to GRAY
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find ID
    id = find_ID(img2, desc_list)
    
    if id != -1:
        # Create a text to show name
        cv2.putText(img_original, class_names[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Get keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img2, None)
    kp2, des2 = orb.detectAndCompute(images[id], None)

    # Brute force matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Get good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Get sample image
    game_image = cv2.cvtColor(cv2.resize(images[id], (960, 840)), cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    img3 = cv2.drawMatchesKnn(img_original, kp1, game_image, kp2, good, None, flags=2)
    
    # Resize image
    ims = cv2.resize(img3, (960, 540))
    
    # Write how many good matches there were
    cv2.putText(ims, str(len(good)), (50, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    # Show images
    cv2.imshow('img3', ims)
    cv2.imshow('img_original', img_original)
    cv2.waitKey(1)
