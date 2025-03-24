import cv2
import argparse
import numpy as np

def extract_blue_rois(image):
    """Extracts all blue ROIs from the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the blue color range in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Mask blue regions
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of the blue regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Remove all small detections
            rois.append((x, y, w, h))

    return rois

def detect_camera_movement(img1, img2):
    """Detect if the camera moved using ORB feature matching and homography."""
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        # No keypoints found, return -1, meaning uncertain        
        return -1 

    # Use BFMatcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        # If only a few matches found, return 1 meaning possible camera movement        
        return 1 

    # Compute Homography to check transformation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)    
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if matrix is None or abs(matrix[0, 2]) > 10 or abs(matrix[1, 2]) > 10:
        # Significant translation detected, return 2        
        return 2

    # Otherwise, Camera did not move!
    return 0

def detect_new_objects(img1, img2, rois):
    """Detect if new obstructions are found in the ROIs."""
    new_objects_detected = []
    
    for roi in rois:
        x, y, w, h = roi

        # Extract ROI from both images
        roi1 = img1[y:y+h, x:x+w]
        roi2 = img2[y:y+h, x:x+w]

        # Convert to grayscale and compute absolute difference
        roi1_gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(roi1_gray, roi2_gray)

        # Apply threshold to highlight differences
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Update if new contours have been found (meaning obstructions)
        if len(contours) > 0:
            new_objects_detected.append(roi)

    # Return list bounding boxes for new objects
    return new_objects_detected  

def is_image_corrupted(img1, img2, blur_ratio_threshold=0.4, color_drop_threshold=0.2):
    """ Detects if img2 is significantly more blurry or grayed out compared to img1."""
    
    # Convert both images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compare Laplacian Variance to check for substantial blur
    laplacian1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
    laplacian2 = cv2.Laplacian(gray2, cv2.CV_64F).var()

    if laplacian2 < blur_ratio_threshold * laplacian1:
        # img2 is significantly blurry
        return True  

    # Check to see if the image has been grayed out or had significant loss in color
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    color_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    if color_similarity < color_drop_threshold:
        # img2 has lost an significant amount of color        
        return True

    # img2 is cleared!    
    return False  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze two images for camera movement, new objects in a given ROI, and blurriness")
    parser.add_argument("before_path", type=str, help="Path to the first image.")
    parser.add_argument("after_path", type=str, help="Path to the second image.")
    args = parser.parse_args()

    # Load images
    before_img = cv2.imread(args.before_path)
    after_img = cv2.imread(args.after_path)

    # Check to see if after image is corrupted
    corrupted_tag = is_image_corrupted(before_img, after_img)
    if corrupted_tag == True:
        # If so, other checks are not necessary. Return.
        camera_moved = False
        objects_in_rois = []
        corrupted_str = ''
    else:
        # If not, check to see if the camera moved
        camera_moved = detect_camera_movement(before_img, after_img)

        # For now, only reporting very significant changes                    
        camera_moved = False if camera_moved <= 0 else True

        # Next extract ROIs from the first image and see if anything new has appeared
        rois = extract_blue_rois(before_img)
        objects_in_rois = detect_new_objects(before_img, after_img, rois)    
        corrupted_str = ' NOT'

    # Print results
    print(f"---RESULT COMPARING {args.before_path} to {args.after_path}---")
    print(f"Camera moved: {camera_moved}")
    print(f"{len(objects_in_rois)} new objects found in ROI(s)")    
    print(f"{args.after_path} is{corrupted_str} corrupted\n")

    # Save results
    outf = args.before_path.replace('.png', '') + '_' + args.after_path.replace('.png', '.txt')
    with open(outf, "w") as FH:        
        FH.write(f"Camera moved: {camera_moved}\n")
        FH.write(f"{len(objects_in_rois)} new objects found in ROI(s)\n")
        FH.write(f"{args.after_path} is{corrupted_str} corrupted\n")
