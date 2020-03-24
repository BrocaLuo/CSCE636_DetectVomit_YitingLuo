import cv2
 
vc = cv2.VideoCapture('C:\\Vomit\\Whole3.mp4')
c = 1
 
 
if vc.isOpened():
 
    rval,frame = vc.read()
else:
    rval = False
 
timeF = 1
 
while rval:
    rval,frame = vc.read()
    if (c%timeF == 0):
        cv2.imwrite('C:\\Vomit\\Dataset'+str(c)+'.jpg',frame)
 
    c = c + 1
    cv2.waitKey(1)
vc.release()

vc = cv2.VideoCapture('C:\\Vomit\\Whole1.mp4')
c = 1
 
 
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
