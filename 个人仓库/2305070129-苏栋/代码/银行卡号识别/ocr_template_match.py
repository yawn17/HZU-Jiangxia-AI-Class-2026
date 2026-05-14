from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser() #创建参数解析器
ap.add_argument("-i", "--image", required=True, 
                help="path to input image") # 添加参数
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A image") # 添加参数
args = vars(ap.parse_args()) # 解析参数

# 指定信用卡类型
FIRST_NUMBER={
    "3": "American Express", # 3开头的是American Express卡
    "4": "Visa",  # 4开头的是Visa卡
    "5": "MasterCard",  # 5开头的是MasterCard卡
    "6": "Discover"  # 6开头的是Discover卡
}

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取一个模板图像
img=cv2.imread(args["template"])
cv_show("img",img)

# 转换为灰度图像
ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show("ref",ref)

# 对图像进行二值化处理
ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show("ref",ref)

# 计算轮廓
# cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图），cv2.RETR_EXTERNAL只检测外围轮廓，cv2.CHAIN_APPROX_SIMPLE只保留轮廓的拐点坐标
# 返回的contours是一个list，每个元素都是一个轮廓，用Numpy的array表示

ref_,refCnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show("img",img)
print(np.array(refCnts).shape)
refCnts=myutils.sort_contours(refCnts,method="left-to-right")[0] # 排序，从左到右，从上到下
digits={}

for (i,c) in enumerate(refCnts):
    (x,y,w,h)=cv2.boundingRect(c)
    roi=ref[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88))

    digits[i]=roi # 每一个数字对应每一个模板

# 初始化卷积核
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# 读取输入图像，预处理
image=cv2.imread(args["image"])
cv_show("image",image)

image=cv2.resize(image,(300,300))
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show("gray",gray)

# 礼貌操作，突出更明亮的区域
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show("tophat",tophat)

# 计算图像梯度
gradX=cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX=np.absolute(gradX)
(minVal,maxVal)=(np.min(gradX),np.max(gradX))
gradX=(255*((gradX-minVal)/(maxVal-minVal))) # 归一化
gradX=gradX.astype("uint8")
print(np.array(gradX).shape)
cv_show("gradX",gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel)
cv_show("gradX",gradX)
# THRESH_OTSU会自动寻找合适的阈值，将图像分为目标和背景
# 对于双峰图像，Otsu算法可以找到两个峰之间的最优阈值
thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
cv_show("thresh",thresh)

# 再使用闭操作，将数字连在一起
thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show("thresh",thresh)

# 计算轮廓
threshCnts,_=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=contours.sort_contours(threshCnts,method="left-to-right")[0] # 排序，从左到右，从上到下

output=[] # 输出结果

# 遍历轮廓
for (i,c) in enumerate(cnts):
    # 计算最小外接矩形
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar>2.5 and ar<4:
        
        if(w>40 and w<55) and (h>10 and h<20):
            
            locs.append((x,y,w,h))
    
# 将符合的轮廓从左到右排序
locs=sorted(locs,key=lambda x:x[0])
output=[]

# 遍历轮廓中的每一个区域
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    # 提取组内的数字
    groupOutput=[]

    # 根据坐标提取每一个组
    group=gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    cv_show("group",group)
    # 预处理
    group=cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv_show("group",group)

    # 计算每一组的轮廓
    group_,digitCnts,hierarchy=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts=contours.sort_contours(digitCnTs,method="left-to-right")[0]
    # 计算每一个数字的组
    for c in digitCnts:
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(57,88))
        cv_show("roi",roi)

        # 计算匹配得分
        scores=[]

        # 在模板中做匹配
        for (digit,digitROI) in digits.items():
            # 模板匹配
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF_NORMED)
            (_,score,_,_)=cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最匹配的数字
        groupOutput.append(str(np.argmax(scores)))
    # 画出来
    cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)

    # 得到结果
    output.extend(groupOutput)

print("credit card type:{}".format(FIRST_NUMBER[output[0]]))
print("credit card number:{}".format("".join(output)))
cv_show("image",image)
