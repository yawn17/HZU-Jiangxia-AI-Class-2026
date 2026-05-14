# 导入工具包
import numpy as np
import argparse # 导入argparse模块，用于处理命令行参数
import cv2

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned") 
args = vars(ap.parse_args())

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts): # 获取4个坐标点
	# 获取输入坐标点
	rect = order_points(pts) 
	# order_points为自定义函数对输入坐标点进行排序
	(tl, tr, br, bl) = rect # 左上，右上，右下，左下

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	# 已知两个点，利用距离公式计算距离（欧式距离）
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0], # 左上
		[maxWidth - 1, 0], # 右上
		[maxWidth - 1, maxHeight - 1], # 右下
		[0, maxHeight - 1]], dtype = "float32") # 左下

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	# getPerspectiveTransform函数计算3x3透视变换矩阵，两个参数，一个是4个点，一个是希望变换后的4个点
	# 具体计算过程是[x,y,z]=['3*3矩阵'*[x,y,1]]，即[x,y,1] * [a,b,c,d,e,f,g,h,i] = [ax+by+c, dx+ey+f, gx+hy+i]

	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# 参数：输入图片，变换矩阵，变换后图片大小
	# warpPerspective函数计算透视变换，即计算变换后的坐标位置

	# 返回变换后结果
	return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

# 读取输入
image = cv2.imread(args["image"])
#坐标也会相同变化
ratio = image.shape[0] / 500.0 # 缩放比例
orig = image.copy() # 拿copy的进行操作


image = resize(orig, height = 500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 灰度化
gray = cv2.GaussianBlur(gray, (5, 5), 0) # 高斯模糊，剔除噪声
edged = cv2.Canny(gray, 75, 200) # 边缘检测

# 展示预处理结果
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
# 最外面的轮廓，从大到小排序
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# 遍历轮廓
for c in cnts:
	# 计算轮廓近似
	peri = cv2.arcLength(c, True)
	# C表示输入的点集
	# epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
	# True表示封闭的
	approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
	# 参数：轮廓序号，轮廓近似精度（越大是矩形，越小是多边形），是否闭合
	# 轮廓点组成的结果，不好算，进行近似

	# 4个点的时候就拿出来
	if len(approx) == 4: # 如果检测到4个点，说明找到了纸张
		screenCnt = approx
		break

# 展示结果
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# 参数：坐标点，缩放比例
# 透视变换，将图片进行变换，变成我们想要的矩形
# orig是没进行resize，screenCnt是进行resize后的，需要还原回去，所以乘以ratio
# screenCnt.reshape(4, 2) * ratio: 将坐标点还原到原图大小

# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('scan.jpg', ref)
# 展示结果
print("STEP 3: 变换")
cv2.imshow("Original", resize(orig, height = 650))
cv2.imshow("Scanned", resize(ref, height = 650))
cv2.waitKey(0)