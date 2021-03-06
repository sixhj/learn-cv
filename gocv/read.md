

# learn opencv
- https://github.com/hybridgroup/gocv
- http://www.1zlab.com/wiki/python-opencv-tutorial/opencv-translation/





# 相关的库
numpy - https://github.com/gonum/gonum 
matplotlib - https://github.com/gonum/plot 
pandas - https://github.com/kniren/gota (API not finalized)
 

# 卷积

什么是卷积   
- https://www.zhihu.com/question/22298352/answer/228543288  
- https://mlnotebook.github.io/post/CNN1/


## 图像二值化

https://zhuanlan.zhihu.com/p/186357948


# 形态学操作

图像形态学主要分为：
- 膨胀（dilate）、
- 腐蚀（erode）、
- 开运算（open）、
- 闭运算（colse）、
- 黑帽（blackHat）、
- 顶帽（topHat）、
- 形态学梯度（gradient）、
- 击中击不中（hit and miss）

## 膨胀

膨胀（dilate）通俗易懂地讲，我们可以把它看成最大值滤波，即用结构元素内的最大值去替换锚点的像素值。
这里所用到的结构元素，和对图像进行卷积操作时所使用的卷积核有些类似，也可以把它理解成形态学操作中的卷积核，只不过在形态学中被称为结构元素，而且结构元素的形状选择会更加的丰富。

对图像进行膨胀操作后，由于相当于对图像进行了一个最大值滤波，
所以对于灰度图像和彩色图像而言，图像的整体亮度会变亮；
对于二值图像而言就是它的白色区域会变大，黑色区域会减少。





## 腐蚀

膨胀和腐蚀是相对而言的，既然膨胀可以被看作是最大值滤波，那么同样的，腐蚀就可以看成是最小值滤波，即用结构元素内的最小值替换锚点像素值。

对图像进行腐蚀操作后，
对于灰度图像和彩色图像而言其整体亮度变暗，
而对于二值图像而言就是黑色区域变大、白色区域减少了。


腐蚀（erode）操作可以消除二值图像中微小的连通域干扰，
因为它能将一些比较小的连通组件给腐蚀掉，使它也变成黑色背景；
或者是用来分离连通域，
如果有一些原本并不属于同一物体的像素点经过二值化后被划分为同一个连通域，
就可以通过腐蚀操作来断开两个不同物体之间的连接，使其分为两个不同的连通域



## 开运算

先腐蚀，再膨胀

腐蚀消除部分亮像素点
膨胀将整体的亮像素点恢复到原来区域范围，整体的灰度值会降低



开操作对于二值图像来说可以消除掉一些小的干扰连通域，而不影响整体的大的连通域，
解决图像二值化之后噪点过多的问题。
还可以用来在纤细点处分离物体，将细小的连接处腐蚀掉再恢复原来的大的连通域。
还可以消除连通域的毛刺，在平滑较大连通域边界的同时并不明显改变其面积。


## 