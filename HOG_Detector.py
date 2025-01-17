
import cv2

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
       
        pad_w, pad_h = int(0.1*w), int(0.1*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (255, 0, 255), 1)
        
hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

cap= cv2.VideoCapture('image/video.mp4')
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    
    ret, img = cap.read() 
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05) 
    found_filtered = [] 
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
        draw_detections(img, found) 
        draw_detections(img, found_filtered, 3)
        print('%d (%d) found' % (len(found_filtered), len(found))) 
    cv2.imshow('img', img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows() 