import cv2
import numpy as np
import argparse

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover"
}


def cv_show(name, img): # 显示图片
    cv2.imshow(name, img)
    cv2.waitKey(0) # 等待键盘输入
    cv2.destroyAllWindows() # 关闭所有窗口


def sort_contours(cnts, method="left-to-right"): # 对输入的轮廓进行排序，输入轮廓列表和排序方式（默认为左到右）
    reverse = False # 是否反转排序，默认情况下不反转
    i = 0 # 排序依据，0表示左上角x坐标，1表示左上角y坐标

    if method == "right-to-left" or method == "bottom-to-top": # 如果排序方式为右到左或下到上，则反转排序
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top": # 如果排序方式为上到下或下到上，则排序依据为左上角y坐标
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] # 使用列表推导式获取每个轮廓的边界框，cv2.boundingRect(c)返回一个包含左上角坐标和宽高的元组
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse)) 
    # 首先使用zip函数将轮廓和边界框打包成元组列表，然后使用sorted函数对元组列表进行排序，排序依据为边界框的左上角坐标，排序方式为reverse参数指定的方式

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA): # 调整图像大小
    # 接受四个参数：输入图像，宽度，高度，插值方法（默认为cv2.INTER_AREA）
    dim = None # 初始化输出图像的尺寸
    (h, w) = image.shape[:2] # 获取输入图像的高度和宽度，[:2]表示取前两个元素，即高度和宽度
    if width is None and height is None:
        return image
    if width is None: # 是否只提供了高度而没有宽度
        r = height / float(h) # 计算缩放比例
        dim = (int(w * r), height) # 计算输出图像的宽度和高度
    else: # 是否只提供了宽度而没有高度
        r = width / float(w) # 计算缩放比例
        dim = (width, int(h * r)) # 计算输出图像的宽度和高度
    resized = cv2.resize(image, dim, interpolation=inter) # 调整图像大小
    return resized


def credit_card_recognition(image_path, template_path):
    # 读取一个模板图像
    img = cv2.imread(template_path)
    if img is None:
        raise ValueError(f"无法读取模板图像：{template_path}")
    cv_show("Original template image", img)

    # 转换为灰度图像
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show("template grayscale image", ref)

    # 对图像进行二值化处理
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show("template binary image", ref)

    # 计算轮廓
    contours, _ = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.findContours()函数返回两个值，第一个是轮廓列表，第二个是层次结构

    refCnts = sort_contours(contours, method="left-to-right")[0] # 对轮廓进行排序，返回排序后的轮廓列表
    digits = {}

    for (i, c) in enumerate(refCnts): # 遍历排序后的轮廓列表，i表示索引，c表示轮廓
        (x, y, w, h) = cv2.boundingRect(c) # 获取轮廓的边界框，返回一个包含左上角坐标和宽高的元组，x和y表示左上角坐标，w和h表示宽度和高度
        roi = ref[y:y + h, x:x + w] # 提取轮廓所在的区域
        roi = cv2.resize(roi, (57, 88)) # 调整区域大小为57x88，与信用卡上的数字大小相同

        digits[i] = roi # 将提取的区域保存到字典中，键为索引，值为区域

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)) # 创建一个9x3的矩形卷积核
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 创建一个5x5的矩形卷积核

    # 读取输入图像，预处理
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取输入图像：{image_path}")
    cv_show("Original input image", image)

    image = resize(image, width=300) # 调整图像大小，宽度为300，高度按比例调整
    cv_show("Resized input image", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv_show("Grayscale input image", gray)

    # 礼貌操作，突出更明亮的区域
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel) # MORPH_TOPHAT表示礼貌操作，突出更明亮的区域
    cv_show("Tophat image", tophat) # Tophat操作，突出图像中的明亮区域

    # 计算图像梯度
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1) # 计算图像梯度，使用Sobel算子，dx=1表示在x方向求导，dy=0表示在y方向求导，ksize=-1表示使用3x3的卷积核
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX)) # 获取图像梯度的最小值和最大值
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    cv_show("Gradient image", gradX) # 计算图像梯度，使用Sobel算子，dx=1表示在x方向求导，dy=0表示在y方向求导，ksize=-1表示使用3x3的卷积核

    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel) # 先膨胀，再腐蚀，将数字连在一起，使得数字成为一个整体
    cv_show("Closed gradient image", gradX)

    # THRESH_OTSU 会自动寻找合适的阈值，将图像分为目标和背景
    # 对于双峰图像，Otsu 算法可以找到两个峰之间的最优阈值
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
    # 对图像进行二值化处理，使用Otsu算法自动寻找合适的阈值，0表示使用Otsu算法，255表示将大于阈值的像素值设置为255，cv2.THRESH_BINARY表示使用二值化处理
    cv_show("Threshold image", thresh)

    # 再使用闭操作，将数字连在一起
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv_show("Closed threshold image again", thresh)

    # 计算轮廓
    threshCnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sort_contours(threshCnts, method="left-to-right")[0]

    output = []

    locs = []
    # 遍历轮廓
    for (i, c) in enumerate(cnts):
        # 计算最小外接矩形
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if ar > 2.5 and ar < 4:
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                locs.append((x, y, w, h))

    # 将符合的轮廓从左到右排序
    locs = sorted(locs, key=lambda x: x[0])
    output = []

    # 遍历轮廓中的每一个区域
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # 提取组内的数字
        groupOutput = []

        # 根据坐标提取每一个组
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        cv_show(f"extracted number group {i}", group)

        # 预处理
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv_show(f"preprocessed number group {i}", group)

        # 计算每一组的轮廓
        contours, _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = sort_contours(contours, method="left-to-right")[0]
        # 计算每一个数字的组
        for c in digitCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            # 计算匹配得分
            scores = []

            # 在模板中做匹配
            for (digit, digitROI) in digits.items():
                # 模板匹配
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF_NORMED) # 使用模板匹配算法，计算模板图像和目标图像之间的匹配得分，cv2.TM_CCOEFF_NORMED表示使用归一化相关系数匹配算法
                (_, score, _, _) = cv2.minMaxLoc(result) # 得到匹配得分的最小值和最大值，以及对应的坐标
                scores.append(score)

            # 得到最匹配的数字
            groupOutput.append(str(np.argmax(scores)))
        # 画出来
        cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1) # 在原图上画出矩形框，表示数字组的位置
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2) 
        # 在原图上画出数字组的内容，(gX, gY - 15)表示数字组的位置，cv2.FONT_HERSHEY_SIMPLEX表示字体，0.65表示字体大小，(0, 0, 255)表示颜色，2表示线条粗细

        # 得到结果
        output.extend(groupOutput)

    print("credit card type:{}".format(FIRST_NUMBER[output[0]]))
    print("credit card number:{}".format("".join(output)))
    cv_show("Result", image)


if __name__ == "__main__":
    input_image_path = r"template-matching-ocr\images\credit_card_02.jpg"
    template_image_path = r"template-matching-ocr\images\ocr_a_reference.jpg"
    credit_card_recognition(input_image_path, template_image_path)