import unittest

import cv2
import numpy as np
from matplotlib import pyplot as plt

COLOR_MAP = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0)
}


def read():
    img = cv2.imread('axis.png', cv2.IMREAD_UNCHANGED)

    # 读取图片
    cv2.imshow('1',img)
    # 获取图片像素数
    rows, clos, ch = img.shape
    print(rows)  # x
    print(clos)  # y
    print(ch)
    # cv2.imwrite() 图片写入
    print(img.shape)  # (763, 865, 3)
    cv2.waitKey(0) # 等待时间
    cv2.destroyWindow()


def open_video_demo():
    video = cv2.VideoCapture('')
    if video.isOpened():
        open_video, frame = video.read()
    else:
        open_video = False
    while open_video:
        ret, frame = video.read()
        if frame is None:
            break
        if ret:
            # 检测每一张图片
            video_img = cv2.cvtColor(frame, cv2.IMREAD_UNCHANGED)
            cv2.imshow('video', video_img)
            if cv2.waitKey(10) & 0xFF == 27:
                break
    video.release()
    cv2.destroyAllWindows()


def get_some_img(img):
    # open_video_demo()
    some = img[0:100, 0:200]  # y1:y2,x1:x2  图片截取的坐标 , 0,0 是从左上角开始
    cv2.imshow('1', some)
    cv2.waitKey(1111)


def img_split():
    # 图片通道  b g r
    print(img.shape)
    b, g, r = cv2.split(img)
    print(b)
    print(g)
    print(r)
    # cv2.merge(b,g,r) # 合并通道  超过255 进行取余
    print(img)
    img2 = cv2.resize(img, (200, 200))  # 重新修改图片的大小，就是像素数

    cv2.imshow('1', img2)
    cv2.waitKey(1111)


def get_screenshot():
    # 桌面截屏
    import pyautogui
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save('save.png')


def get_screen2():
    import pyscreenshot as ImageGrab
    # 全屏截取
    im = ImageGrab.grab()
    im.save('grab-full.png')
    im.show()
    # 截取部分
    im = ImageGrab.grab(bbox=(10, 10, 500, 500))
    im.show()
    # 保存文件
    im.save('grab.png')


if __name__ == '__main__':
    # read()
    print('1')
    img = cv2.imread('axis.png', cv2.IMREAD_UNCHANGED)
    # get_some_img(img)

    # img_split()

    # 阈值函数 ，threshold   当像素值 超过时进行的操作  二值化
    # cv2.threshold()

    # get_screenshot()

    # get_screen2()





class Test_cv(unittest.TestCase):


    def test_open_img(self):
        img =cv2.imread('./axis.png',cv2.IMREAD_UNCHANGED)
        cv2.imshow('1',img)
        cv2.waitKey(1111)


    def test_rectangle(self):
        rectangle = np.zeros((300, 300), dtype="uint8")
        cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
        cv2.imwrite("bitwise_rectangle.png", rectangle)
        cv2.imshow('',rectangle)
        cv2.waitKey(1111)



    def test_info(self):
        src = cv2.imread('./axis.png', cv2.IMREAD_UNCHANGED)
        print(src.size)
        print(src.dtype) #uint8
        print(src.shape) #(763, 865, 3)  高y 宽x 通道
        print("图像宽度: {} pixels".format(src.shape[1]))
        print("图像高度: {} pixels".format(src.shape[0]))
        print("通道: {}".format(src.shape[2]))

        # print(src) # 第一个纬度 height nRows 第二纬度 width  nCol 列数


    def test_roi(self):
        # 图像切割
        img = cv2.imread('./axis.png')
        # cv2.imshow('',img)
        # cv2.waitKey(1111)
        # 图像就是个数组，按数组的方法截取就可以了
        img2 = img[300:500,300:500] # y0:y1 ,x0:x1

        cv2.imshow('',img2)
        cv2.waitKey(1111)

    def test_size(self):
        img =cv2.imread('./axis.png',cv2.IMREAD_GRAYSCALE) # BGR 0-255
        cv2.imshow('',img)
        cv2.waitKey(1111)
        cv2.destroyAllWindows()


        img2 = np.copy(img) # 图片拷贝

        # opencv 的图像都是数组


    def test_pixel(self):
        v = cv2.version.opencv_version
        print(v)
        src = cv2.imread('./axis.png')
        print(src.shape)#(763, 865, 3)






    def test_color(self):
        """
        颜色空间RGB
            加法色彩空间，通过红、绿、蓝的线性组合获得

        Lab 颜色空间 COLOR_BGR2LAB
            L 亮度
            a 从绿色到洋红的颜色分量
            b 从蓝色到黄色的颜色分量

        YCrCb 颜色空间源自 RGB 颜色空间，具有以下三个分量。
            Y – 伽马校正后从 RGB 中获得的亮度或亮度分量。
            Cr = R – Y（红色分量距离 Luma 有多远）。
            Cb = B – Y（蓝色分量距离 Luma 有多远）。

            该色彩空间具有以下属性。
            将亮度和色度分量分离到不同的通道中。
            主要用于电视传输的压缩（Cr 和 Cb 分量）。
            取决于设备。

        HSV色彩空间有以下三个分量
            H – 色调（主波长）。
            S – 饱和度（纯度/颜色深浅）。
            V——价值（强度）。

            让我们列举一下它的一些属性。
            最好的是它只使用一个通道来描述颜色（H），使得指定颜色非常直观。
            取决于设备。

        灰度化处理就是将一幅彩色图像转化为灰度图像的过程，简化数据量
        """

        src =  cv2.imread('axis.png')
        print('')
        cv2.imshow('',src)
        cv2.waitKey(1111)
        # 图像截取成500 500
        src2 = cv2.resize(src,(500,500),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('1',src2)
        cv2.waitKey(0)



    def test_get_some(self):
        src =  cv2.imread('axis.png')
        img2 = src[60:560,63:575] # y0:y1 ,x0:x1
        cv2.imshow('3',img2)
        cv2.waitKey(0)
        cv2.imwrite('self.png',img2)


    def test_line(self):
        src = cv2.imread('self.png')
        # pt1 x1,y1 pt2 x2,y2
        cv2.line(src,pt1=(100,100),pt2=(100,200),color=(0,255,0),thickness=3)
        cv2.line(src,pt1=(100,500),pt2=(200,300),color=COLOR_MAP.get('blue'),thickness=3)
        cv2.imshow('3', src)
        cv2.waitKey(0)

    def test_write_font(self):
        # src = cv2.imread('self.png')
        src = np.ones((400, 600, 3), dtype="uint8") # y ,x      高y 宽x 通道
        print(src.shape)
        src *= 255 # 背景色白色，默认黑色
        cv2.putText(img=src,text='hello word',org=(50,55),color=COLOR_MAP.get('red'),thickness=3,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,lineType=cv2.LINE_AA)

        # for x  in range (0,src.shape[1],20):
        #     for y in range (0,src.shape[0],20):
        #         cv2.rectangle(img=src,pt1=(x,y),pt2=(x+20,y,+20),color=COLOR_MAP.get('blue'))


        cv2.imshow('3', src)
        cv2.waitKey(0)

    def test_add(self):
        #图片叠加
        img1 = cv2.imread('img1.png')
        img2 =cv2.imread('img3.png')
        dst = cv2.addWeighted(img2,0.8,img1,0.5,0)
        cv2.imshow('',dst)
        cv2.waitKey(0)

    def test_warp_affine(self):
        # 声明变换矩阵 向右平移10个像素， 向下平移30个像素
        M = np.float32([[1, 0, 10], [0, 1, 30]])
        print(M)
        img = cv2.imread('img1.png')
        img2 =cv2.warpAffine(img,M,(200,200))
        cv2.imshow('',img2)
        cv2.waitKey(0)

    def test_rotation(self):
        # 生成旋转矩阵   (center, angle, scale)  中心点, 角度 逆时针，缩放
        rotateMatrix = cv2.getRotationMatrix2D((100, 100), 45, 1.0)
        print(rotateMatrix)
        np.set_printoptions(precision=2, suppress=True)
        print(rotateMatrix)
        img = cv2.imread('img1.png')
        img2 = cv2.warpAffine(img, rotateMatrix, (200, 200))
        cv2.imshow('', img2)
        cv2.waitKey(0)


    # 图片翻转
    def test_flip(self):
        src = cv2.imread('self.png')
        src2 = cv2.flip(src,1) # 1 水平 2 垂直
        cv2.imshow('',src2)
        cv2.waitKey(1111)

        src3 = cv2.addWeighted(src,0.5,src2,0.5,0)
        cv2.imshow('',src3)
        cv2.waitKey(0)


    def test_window(self):
        cv2.namedWindow('image_win', flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        # cv2.waitKey(0)
        key_num = cv2.waitKey(0)

        if chr(key_num) == 'k':
            print("k pressed...")


    def test_one(self):
        a = ord('a')
        print(a)

        img = cv2.imread('img1.png')
        for row in  range(img.shape[0]):
            for col in range(img.shape[1]):
                b,g,r= img[row,col]
                img[row, col] = (255-b,255-g,255-r) # 像素取反
                img[row, col] = (0,0,255-r) # 像素取反
        # pt1 x1,y1 pt2 x2,y2          pt1  左上角      pt2 左下角
        # thickness=-1    -1 表示填充
        cv2.rectangle(img=img, pt1=(1, 1), pt2=(100,200), color=COLOR_MAP.get('blue'),thickness=-1)
        cv2.imshow('',img)
        cv2.waitKey(1111)


    def test_gray(self):
        # 腐蚀
        img = cv2.imread('self.png',cv2.IMREAD_GRAYSCALE)

        kernel = np.ones((5,5),img.dtype) # 腐蚀内核，就是最小单位
        erorsion_img = cv2.erode(img, kernel, iterations=1) # 腐蚀

        cv2.imshow('',img)
        cv2.imshow('',erorsion_img)
        cv2.waitKey(0)

    def test_float(selfs):
        # 整数转换为浮点数
        img = cv2.imread('self.png',cv2.IMREAD_GRAYSCALE)
        # img2 = np.float(img)
        cv2.imshow('',img/255.0)
        # cv2.imshow('',img.astype(np.float))
        cv2.waitKey(1111)


    def test_erode(self):
        kernel = np.ones((5,5),np.uint8)
        src = cv2.imread('img_1.png')
        img_bin = cv2.inRange(src, lowerb=(9, 16, 84), upperb=(255, 251, 255))


        #  腐蚀
        img_bin = cv2.erode(img_bin, kernel, iterations=1)
        # 膨胀
        # img_bin = cv2.dilate(img_bin, kernel, iterations=2)

        cv2.imshow('',img_bin)
        cv2.waitKey(1111)


    def test_findContours(self):
        # 提取轮廓，选择想要的目标范围

        img = cv2.imread('img_2.png')
        cv2.imshow('',img)
        cv2.waitKey(1111)
        '''
        输入的图像是二值化图像，检测的对象为白色,寻找轮廓就像从黑色背景中寻找白色物体。
        cv2.cvtColor()将 RGB 图像转换为灰度图像
        cv2.threshold()二值化 
                • cv2.THRESH_BINARY（黑白二值）
                • cv2.THRESH_BINARY_INV（黑白二值反转）
                • cv2.THRESH_TRUNC （得到的图像为多像素值）
                • cv2.THRESH_TOZERO
                • cv2.THRESH_TOZERO_INV
        
        CHAIN_APPROX_NONE 记录所有的点，
        CHAIN_APPROX_SIMPLE 记录必要的的点
        '''
        # 二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret 得到的阈值
        ret, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        cv2.imshow('',bin_img)
        cv2.waitKey(1111)

        # cv2.RETR_EXTERNAL：只提取最外层的轮廓
        # cv2.RETR_LIST：提取所有轮廓但不创建层次结构
        # cv2.RETR_CCOMP：提取所有轮廓并创建两层层次结构
        # cv2.RETR_TREE：提取所有轮廓并在树中创建层次结构
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))

        cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)
        cv2.imshow('', img)
        cv2.waitKey(1111)

    #     contours是轮廓列表。每个元素都是一个 numpy 数组，其形状表示轮廓点列表(点の数, 1, 2)。

        for i, cnt in enumerate(contours):
            print(f"contours[{i}].shape: {cnt.shape}")


    def test_create_2(self):
        rectangle = np.zeros((300, 300), dtype="uint8")
        cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
        cv2.imwrite("bitwise_rectangle.png", rectangle)

        circle = np.zeros((300, 300), dtype="uint8")
        cv2.circle(circle, (150, 150), 150, 255, -1)
        cv2.imwrite("bitwise_circle.png", circle)


    def test_not(self):
        # 二值化逻辑运算  not 取反
        circle = np.zeros((300, 300), dtype="uint8")
        cv2.circle(circle, (150, 150), 150, 255, -1)
        cv2.imshow('',circle)
        cv2.waitKey(1111)
        bitwiseNOT = cv2.bitwise_not(circle)
        cv2.imshow('',bitwiseNOT)
        cv2.waitKey(1111)


    def test_and(self):
        # 逻辑与经常被用于遮盖层(MASK), 即去除背景, 选取自己感兴趣的区域.
        # 0 1 = 0  ,1 0 =1 , 1 1 = 1
        rectangle = np.zeros((300, 300), dtype="uint8")
        cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
        cv2.imshow('', rectangle)
        cv2.waitKey(1111)

        circle = np.zeros((300, 300), dtype="uint8")
        cv2.circle(circle, (150, 150), 150, 255, -1)
        cv2.imshow('', circle)
        cv2.waitKey(1111)


        bitwiseAnd = cv2.bitwise_and(rectangle, circle)
        cv2.imshow('', bitwiseAnd)
        cv2.waitKey(1111)

