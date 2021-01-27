import cv2, time
from datetime import datetime
import pandas

first_frame = None
status_list = [0]
times = []
df = pandas.DataFrame(columns=["Start","End"])
# capture video from web camera
# cv2.VideoCapture(...) the parameter can be the index of number of cameras, or the video file (ex: .mp4)
video = cv2.VideoCapture(0)

while True:
    # check- returns boolean variable whether the video is running or not
    # frame- returns a numpy array 3-d if colored video
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # change video to gray scale 
    gray = cv2.GaussianBlur(gray,(21,21),0) # blur the image, need to pass the gaussian curnnel

    if first_frame is None:
        first_frame = gray
        continue # continue to the beginning of the loop (second iteration) and do not execute the step after this if statement

    delta_frame = cv2.absdiff(first_frame,gray)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
    # use thresh_frame.copy() will not chang original thresh_frame

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1

        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)

    status_list.append(status)

    status_list = status_list[-2:] # memory improve 
    
    if status_list[-1] != status_list[-2]:
        times.append(datetime.now())
    
    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame", frame)




    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break 
print(times)
print(status_list)

for i in range(0,len(times),2):
    df = df.append({"Start":times[i], "End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()