import cv2
 
vc = cv2.VideoCapture('C:\\Vomit\\Vomit2.mp4')
c = 143
 
 
if vc.isOpened():
 
    rval,frame = vc.read()
else:
    rval = False
 
timeF = 1
 
while rval:
    rval,frame = vc.read()
    if (c%timeF == 0):
        cv2.imwrite('C:\\Vomit\\Test'+str(c)+'.jpg',frame)
 
    c = c + 1
    cv2.waitKey(1)
vc.release()
