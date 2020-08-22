import cv2
import numpy as np

vs = cv2.VideoCapture("/home/ram/Desktop/Anukai solutions/projects/automatic number plate detection/samples/Rec 0044.mp4")

print("taking frames...")
c=1
while True:

    grabbed, image = vs.read()
    if not grabbed:
        break
    src = image
    org = src
    if c%5==0:
        # 1
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray image', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 2
        # apply guassian blur on src image
        dst = cv2.GaussianBlur(src, (3, 3), cv2.BORDER_DEFAULT)

        # display input and output image
        # cv2.imshow("Gaussian Smoothing", np.hstack((src, dst)))
        # cv2.waitKey(0)  # waits until a key is pressed
        # cv2.destroyAllWindows()  # destroys the window showing image

        # 3
        edge = cv2.Sobel(dst, -1, 1, 0)
        # 4
        threshold, threshold_image = cv2.threshold(edge, 32, 255, cv2.THRESH_BINARY)
        # 5
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.morphologyEx(threshold_image, cv2.MORPH_GRADIENT, kernel)
    #     6
        i = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY);
        contours, hierarchy = cv2.findContours(i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ROI_number = 0
        for i in range(0, len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # print(area)
            if (area > 200):
                # rect = cv2.minAreaRect(cnt)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # im = cv2.drawContours(src, [box], 0, (0, 0, 255), 2)
                # x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                extent = float(area) / rect_area
                aspect_ratio = float(w) / h
                if (extent > 0.4 and h > 20 and h < 25 and w > 70 and w<90):
                    im = cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(src, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    print("{} : {} : {} : {} :{}".format(c,i, extent, w, h))
                    ROI = src[y:y + h, x:x + w]
                    dsize = (150,50)
                    output = cv2.resize(ROI, dsize)
                    cv2.imwrite('/home/ram/Desktop/Anukai solutions/projects/automatic number plate detection/samples/re/{}_{}_ROI_{}.png'.format(c,i,ROI_number), output)
                    ROI_number += 1
    c+=1
print("complwted")
