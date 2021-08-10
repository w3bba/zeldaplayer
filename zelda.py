import cv2 as cv
import numpy as np
import time

#IMG_RGB = cv.imread('testimages/zelda5.png')


def createNumberSlices():
  img_numbers = cv.imread('masks/numbers.png', 0)
  return [img_numbers[0:7,int(i*7):int(i*7)+7] for i in range(0,10)]

NUMBERS = createNumberSlices()

def findHearts(frame, threshold):
  img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  template = cv.imread('masks/heart_full.png',0)
  mask = cv.imread('masks/heart_mask.png',cv.IMREAD_GRAYSCALE)
  w, h = template.shape[::-1]
  y, x = img_gray.shape[::]
  res = cv.matchTemplate(img_gray[0:50,int(x/2):x],template,cv.TM_CCOEFF_NORMED,mask)
  loc = np.where( res >= threshold)
  full_h=0;all_h=0
  for pt in zip(*loc[::-1]):
    all_h+=1 
    if (img_gray[pt[1]+(h//2),pt[0]+(w//2)+int(x/2)]>20):
      full_h+=1
  return full_h,all_h

def findHalfHeart(frame, threshold):
  img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  template = cv.imread('masks/heart_half.png',0)
  mask = cv.imread('masks/heart_full_mask.png',cv.IMREAD_GRAYSCALE)
  y, x = img_gray.shape[::]
  res = cv.matchTemplate(img_gray[0:50,int(x/2):x],template,cv.TM_CCOEFF_NORMED,mask)
  loc = np.where( res >= threshold)
  return loc[0].size!=0

def findItemCount(img_slice):
  loc_array = np.empty(10, dtype=object)
  loc_list = []
  tr, bw_slice = cv.threshold(img_slice,200,255,cv.THRESH_BINARY)
  cv.imshow('items',bw_slice)
  cv.imwrite('masks/items.png',bw_slice)
  for i in range(0,10):
    loc_array = np.where( cv.matchTemplate(bw_slice,NUMBERS[i],cv.TM_CCOEFF_NORMED,NUMBERS[i]) >= .95)
    for pt in zip(*(loc_array)[::-1]):
      loc_list.append([pt[0],pt[1],i])
  loc_list.sort(key=lambda loc:loc[0])
  i=1
  rupee = 0; bomb = 0; arrow = 0; key = 0
  for loc in loc_list:
    if i==1:
      rupee += loc[2]*100
    elif i==2:
      rupee += loc[2]*10
    elif i==3:
      rupee += loc[2]*1
    elif i==4:
      bomb += loc[2]*10
    elif i==5:
      bomb += loc[2]*1
    elif i==6:
      arrow += loc[2]*10
    elif i==7:
      arrow += loc[2]*1
    else:
      key = loc[2]
    i+=1
  return rupee, bomb, arrow, key

def getMagicMeter(frame):
  ret, thres = cv.threshold(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)[23:55,25:26],20,255,cv.THRESH_BINARY)
  return int((cv.countNonZero(thres)/thres.size)*10000)/100


frames = []
cap = cv.VideoCapture('testimages/vid1.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    start_time=time.monotonic()
    fh, ah = findHearts(frame,.8)
    magic = getMagicMeter(frame)
    hh = findHalfHeart(frame,.8)
    rupee,bomb,arrow,key =findItemCount(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)[24:31,60:160])
    if (hh):
      fh+=.5
    font = cv.FONT_HERSHEY_SIMPLEX
    #cv.rectangle(frame,(60,20),(160,35),(0,255,0),3)
    cv.putText(frame,'HP   : '+str(fh)+'/'+str(ah),(180,50), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame,'Rupee: '+str(rupee),(180,65), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame,'Bomb : '+str(bomb),(180,80), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame,'Arrow: '+str(arrow),(180,95), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame,'Key  : '+str(key),(180,110), font, 0.3,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame,'Magic: '+str(magic)+'%',(180,125), font, 0.3,(255,255,255),1,cv.LINE_AA)
    end_time=time.monotonic()
    # Display the resulting frame
    cv.imshow('Frame',frame)
    frames.append(1/(end_time-start_time))
    # Press Q on keyboard to  exit
    if cv.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Calculate average FPS
print('Average FPS: ',(sum(frames)/len(frames)))

# Closes all the frames
cv.destroyAllWindows()