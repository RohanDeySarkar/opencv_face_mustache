import cv2

faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "haarcascade_mcs_nose.xml"

faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)

imgMustache = cv2.imread('mustache.png', -1)

orig_mask = imgMustache[:, :, 3]  # Here we take just the alpha layer and create a new single-layer image that we will use for masking.

orig_mask_inv = cv2.bitwise_not(orig_mask)# The initial mask will define the area for the mustache, and the inverse mask will be for the region around the mustache

imgMustache = imgMustache[:, :, 0:3] # Here we convert the mustache image to a 3-channel BGR image

origMustacheHeight, origMustacheWidth = imgMustache.shape[:2] # Save the original mustache image sizes, which we will use later when re-sizing the mustache image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        
        )

    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w] # We create a greyscale ROI for the area where the face was discovered
                                      # (remember that we will be looking for a nose within this face, and Haar cascade classifiers operate on greyscale images.
                                      
        roi_color = frame[y:y+h, x:x+w] # We also keep a color ROI for the area where the face is, as we will draw our mustache over the color ROI. 

        nose = noseCascade.detectMultiScale(roi_gray)

        for (nx, ny, nw, nh) in nose:

            mustacheWidth = 3 * nw # keep The mustache should be three times the width of the nose

            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

            
            x1 = int(nx - (mustacheWidth/4))
            x2 = int(nx + nw + (mustacheWidth/4))
            y1 = int(ny + nh - (mustacheHeight/2))
            y2 = int(ny + nh + (mustacheHeight/2))

            if x1 < 0:
                x1 = 0

            if y1 < 0:
                y1 = 0

            if x2 > w:
                x2 = w

            if y2 > h:
                y2 = h

            mustacheWidth = int(x2 - x1)

            mustacheHeight = int(y2 - y1)

            mustache = cv2.resize(imgMustache, (mustacheWidth, mustacheHeight), interpolation = cv2.INTER_AREA)

            mask = cv2.resize(orig_mask, (mustacheWidth, mustacheHeight), interpolation = cv2.INTER_AREA)

            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth, mustacheHeight), interpolation = cv2.INTER_AREA)

            roi = roi_color[y1:y2, x1:x2]

            roi_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

            roi_fg = cv2.bitwise_and(mustache, mustache, mask = mask)

            dst = cv2.add(roi_bg, roi_fg)

            roi_color[y1:y2, x1:x2] = dst

            break

        cv2.imshow('frame', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()


    
