import cv2
import numpy as np

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover"
}

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 对轮廓进行排序，默认从左到右
def sort_contours(cnts,method="left-to-right"):
    reverse=False # 不反转
    i=0 # x

    if method=="right-to-left" or method=="bottom-to-top":
        reverse=True

    if method=="bottom-to-top" or method=="top-to-bottom":
        i=1 # y

    boundingBoxes=[cv2.boundingRect(c) for c in cnts] 
    # cv2.boundingRect(c)表示计算给定轮廓c的最小外接矩形，返回左上角的x和y，以及矩形的w和h

    (cnts,boundingBoxes)=zip(*sorted(zip(cnts,boundingBoxes),key=lambda b:b[1][i],reverse=reverse))
    # zip将两个列打包成一个元组
    # key表示排序依据，b[1]表示边界框，b[1][i]表示边界框的第i个元素，(x,y,w,h)

    return cnts,boundingBoxes

# 调整图像的大小
def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    # cv2.INTER_AREA是一种插值方法，用于缩小图像
    dim=None # 初始化输入图像尺寸
    (h,w)=image.shape[:2] # 只取前两个元素
    if width is None and height is None:
        return image
    if width is None:
        r=height/float(h) # 比例
        dim=(int(w*r),height)
    else:
        r=width/float(w)
        dim=(width,int(h*r))
    resized=cv2.resize(image,dim,interpolation=inter)
    return resized

def credit_card_recognition(image_path,template_path):
    img=cv2.imread(template_path)
    if img is None:
        raise ValueError(f"photo have error:{template_path}")
    cv_show("original template image",img)

    # 灰度
    ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv_show("template grayscale image",ref)

    # 二值
    ref=cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
    # 10是阈值，高于10的设置为黑色，低于10的设置为白色（白变黑，黑变白）
    # 255是最大值，表示高于255的将被设置为255
    # cv2.THRESH_BINARY_INV是阈值处理的类型，表示反转二值化。在这种模式下，低于阈值的像素值将被设置为最大值（白色），高于阈值的像素值将被设置为0（黑色）。
    # cv2.threshold函数返回两个值，一是阈值，二是处理后的图像，这里需要处理后的图像，故取[1]
    cv_show("template binary image",ref)

    # 轮廓列表
    contours,_=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # find_Contours返回两个值，一是轮廓列表，二是层次结构。使用_占位符表示忽略层次结构
    # cv2.RETR_EXTERNAL：这是轮廓检索模式，表示只检测外部轮廓。
    # cv2.CHAIN_APPROX_SIMPLE：这是轮廓近似方法，表示只保留轮廓的拐点信息。

    # 轮廓排序
    refCnts=sort_contours(contours,method="left-to-right")[0]
    # 返回排序后的轮廓列表

    # 提取数字区域
    digits={}
    for (i,c) in enumerate(refCnts): 
    # enumerate函数为每个轮廓提供索引i和轮廓c
        (x,y,w,h)=cv2.boundingRect(c)
        roi=ref[y:y+h,x:x+w]
        # 使用切片操作来获取图像的一个子区域
        roi=cv2.resize(roi,(57,88))
        digits[i]=roi

    image=cv2.imread(image_path)
    if img is None:
        raise ValueError(f"photo have error:{template_path}")
    cv_show("original template image",img)

    image=resize(image,width=300)
    cv_show("resized input image",image)

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv_show("grayscale input image",gray)

    # 初始化卷积核
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    # 礼貌操作：原始图像-开运算（先腐蚀后膨胀），突出图像中明亮的区域
    tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
    cv_show("tophat image",tophat)

    # 使用Sobel算子计算图像梯度
    gradX=cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
    gradX=np.absolute(gradX)
    # 计算gradX的绝对值，以确保梯度值都是正数
    (minVal,maxVal)=(np.min(gradX),np.max(gradX))
    gradX=(255*((gradX-minVal)/(maxVal-minVal))) # 归一化
    gradX=gradX.astype("uint8")
    cv_show("gradient image",gradX) #展示有梯度的部分

    # 闭操作
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    cv_show("closed gradient image", gradX) # 将数字连在一起

    # 二值化
    thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    # 0是阈值参数，用于cv2.THRESH_BINARY | cv2.THRESH_OTSU模式
    # 在这两种模式下，阈值参数被忽略，实际的阈值由Otsu的方法自动确定
    # cv2.THRESH_BINARY：表示二值化类型为常规二值化
    # cv2.THRESH_OTSU：表示使用Otsu的方法自动确定最佳阈值
    # cv2.threshold函数返回两个值，一是实际使用的阈值，二是处理后的二值图像

    # 再次闭操作
    sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 创建一个5x5的矩形卷积核
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv_show("Closed threshold image again", thresh) 
    # 用于填充前景物体中的小洞，帮助将数字的各个部分连接起来，使得数字成为一个整体

    # 轮廓（银行卡）
    threshCnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sort_contours(threshCnts, method="left-to-right")[0]

    locs=[]
    for (i,c) in enumerate(cnts):
        (x,y,w,h)=cv2.boundingRect(c)
        ar=w/float(h) # 宽高比
        if ar>2.5 and ar<4: # 检查宽高比是否在一定的范围内
            if (w>40 and w<55) and (h>10 and h<20): # 检查宽度和高度是否在一定的范围内
                locs.append((x,y,w,h))

    # 轮廓重排
    locs=sorted(locs,key=lambda x:x[0])
    # 将locs中的边框进行重排，从左到右，通过比较每一组的第一个元素进行实现
    
    output=[]
    for (i,(gX,gY,gW,gH)) in enumerate(locs):
        groupOutput = []
        
        group=gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
        # 从gray图的像素矩阵中提取出比(x,y,w,h)确定的区域还要稍大5的区域
        # 索引取y-5到y+h+5，x-5到x+w+5，确保即使数字边缘接近边界框边缘，也能被完整提取
        cv_show(f"e{i}",group)

        # 二值化
        group=cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        cv_show(f"b{i}",group)

        contours,_=cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        digitCnts=sort_contours(contours,method="left-to-right")[0]

        for c in digitCnts:
            (x,y,w,h)=cv2.boundingRect(c)
            roi=group[y:y+h,x:x+w]
            roi=cv2.resize(roi,(57,88))

            scores=[]
            for (digit,digitROI) in digits.items(): # digits是模板 
            # digits是前面定义的区域，是字典形式，里面包括多个小区域
                # 模板匹配
                result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF_NORMED)
                # cv2.TM_CCOEFF_NORMED是匹配方法的参数，表示使用归一化相关系数匹配算法

                (_,score,_,_)=cv2.minMaxLoc(result)
                # minMaxLoc(minVal,maxVal,minLoc,maxLoc)是用于找到矩阵中的最小值和最大值，以及它们的位置
                # 这里只需要最大值，故其他使用占位符进行忽略

                scores.append(score)

            groupOutput.append(str(np.argmax(scores)))
            # np.argmax 是 NumPy 库中的一个函数，用于返回数组中最大值的索引
            # 这里返回的是scores中最大值的索引，并将其转化为字符串

        cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
        # cv2.rectangle用于在图像上绘制矩形
        # 左上角坐标，右下角坐标
        # (0,0,255)表示矩形颜色，这里表示红色

        cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # cv2.putText用于在图像上添加文本
        # cv2.FONT_HERSHEY_SIMPLEX指定字体类型。
        # 0.65表示字体大小。  
        # (0, 0, 255)表示文本的颜色，这里表示红色
        # 2表示文本线条的粗细

        output.extend(groupOutput)
        # output.extend是列表的方法，用于将一个列表中的所有元素添加到另一个列表的末尾

    print(output)
    print("credit card type:{}".format(FIRST_NUMBER[output[0]]))
    # output[0]是识别出的信用卡号的第一个数字，用来从FIRST_NUMBER字典中查找对应的信用卡类型

    print("credit card number:{}".format("".join(output)))
    cv_show("Result", image)

if __name__=="__main__":
    input_image_path = r"template-matching-ocr\images\credit_card_01.jpg"
    template_image_path = r"template-matching-ocr\images\ocr_a_reference.jpg"
    credit_card_recognition(input_image_path, template_image_path)

    





    
