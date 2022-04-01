import os
import cv2



print(os.getcwd())
print()

c = 0
for folder in os.listdir('images1024x1024/'):
    if '.txt' in folder:
        continue
    # print(folder)
    for img_name in os.listdir('images1024x1024/'+folder):
        if '.tmp.' in img_name:
            continue
        # print(img_name)
        c += 1
        if c%100==0:
            print(c)
        img = cv2.imread('images1024x1024/'+folder+'/'+img_name)
        resized = cv2.resize(img, (512,512), interpolation = cv2.INTER_AREA)
        cv2.imwrite('dataset/'+img_name, resized)



