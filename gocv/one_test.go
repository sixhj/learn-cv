package main

import (
	"fmt"
	"image"
	"image/color"
	"testing"
	"time"

	"github.com/go-vgo/robotgo"
	"github.com/vcaesar/imgo"
	"gocv.io/x/gocv"
)

func TestOpenImg(t *testing.T) {
	t.Log("hello")
	img := gocv.IMRead("./axis.png", gocv.IMReadUnchanged)
	window := gocv.NewWindow("Hello")
	window.IMShow(img)
	window.WaitKey(1111)
	window.Close()
}

// 展示图片
func ShowImg(img gocv.Mat) {
	window := gocv.NewWindow("Hello")
	window.IMShow(img)
	window.WaitKey(0)
	window.Close()
}

func Test2(t *testing.T) {
	img2 := gocv.NewMat()
	m := gocv.IMRead("./axis.png", gocv.IMReadUnchanged)
	//  转换为灰度图
	gocv.CvtColor(m, &img2, gocv.ColorBGRToGray)
	ShowImg(img2)
	gocv.IMWrite("./gray.png", img2) // 保存图片
}

func Test2Chan(t *testing.T) {
	// 图像通道、rows 高  行   cols 宽  列  size 大小

	m := gocv.IMRead("./axis.png", -1)
	defer m.Close()
	fmt.Printf("m.Rows(): %v\n", m.Rows())

	fmt.Printf("m.Cols(): %v\n", m.Cols())

	fmt.Printf("m.Channels(): %v\n", m.Channels())

	fmt.Printf("m.Size(): %v\n", m.Size())

	fmt.Printf("m.Total(): %v\n", m.Total())

}

func TestCreateMat(t *testing.T) {
	//初始化一个通道
	m2 := gocv.Zeros(8, 10, gocv.MatTypeCV8SC1) // MatTypeCV8SC1  1 个通道

	i, err := m2.DataPtrInt16()

	if err != nil {
		t.Error(err)
		return
	}

	fmt.Printf("%+v\n", i)

	fmt.Printf("m2.Size(): %v\n", m2.Size())
}

func TestResize(t *testing.T) {
	// 调整图像大小

	img := gocv.IMRead("./axis.png", -1)
	dst := gocv.NewMat()
	// 缩放
	gocv.Resize(img, &dst, image.Point{img.Cols() / 2, img.Rows() / 2}, 0, 0, gocv.InterpolationLinear)
	ShowImg(dst)
}

func TestSplit(t *testing.T) {
	m := gocv.IMRead("./axis.png", gocv.IMReadColor)
	// 切分三个通道 显示
	m2 := gocv.Split(m)
	for _, v := range m2 {
		ShowImg(v)
		time.Sleep(1112)
	}
}

var colorMap = map[string]color.RGBA{
	"red":    {255, 0, 0, 1},
	"blue":   {0, 0, 255, 1},
	"yellow": {255, 255, 0, 1},
	"green":  {0, 255, 0, 1},
}

func TestCanvas(t *testing.T) {
	m := gocv.Ones(600, 800, gocv.MatTypeCV8SC3)
	fmt.Printf("m.Channels(): %v\n", m.Channels())

	// 直线
	gocv.Line(&m, image.Point{100, 100}, image.Point{300, 20}, colorMap["red"], 1)

	// 矩形
	gocv.Rectangle(&m, image.Rect(100, 200, 200, 300), color.RGBA{255, 255, 0, 200}, 1)

	// 圆
	gocv.Circle(&m, image.Point{500, 200}, 100, color.RGBA{255, 11, 255, 200}, 1)

	// 文字
	gocv.PutText(&m, "hello", image.Point{300, 300}, gocv.FontHersheySimplex, 1, color.RGBA{255, 255, 255, 0}, 1)

	ShowImg(m)

	gocv.IMWrite("canvas.png", m)
}

func TestTrans(t *testing.T) {
	// 图像旋转
	m := gocv.IMRead("./axis.png", gocv.IMReadUnchanged)

	dst := gocv.NewMat()
	// 方向：逆时针
	rotaionMat := gocv.GetRotationMatrix2D(image.Point{m.Cols() / 2, m.Rows() / 2}, 45, 0.5) //旋转中心 角度 缩放比例

	gocv.WarpAffine(m, &dst, rotaionMat, image.Point{m.Cols(), m.Rows()})
	ShowImg(dst)

}

func TestThreshold(t *testing.T) {
	// 阈值
	// src := gocv.IMRead("./axis.png", gocv.IMReadGrayScale)
	src := gocv.IMRead("./axis.png", gocv.IMReadUnchanged)
	ShowImg(src)
	dst := gocv.NewMat()
	gocv.Threshold(src, &dst, 0, 255, gocv.ThresholdBinary) // 0 255 阈值区域
	ShowImg(dst)
}

func TestBinary(t *testing.T) {
	rectangle := gocv.Zeros(300, 300, gocv.MatTypeCV8SC3)

	gocv.Rectangle(&rectangle, image.Rectangle{image.Point{50, 50}, image.Point{100, 200}}, color.RGBA{255, 0, 0, 0}, 3)
	ShowImg(rectangle)
	color.Black.RGBA()

}

func TestPixel(t *testing.T) {
	img := gocv.IMRead("./axis.png", gocv.IMReadUnchanged)

	fmt.Printf("img.Cols(): %v\n", img.Cols())
	fmt.Printf("img.Rows(): %v\n", img.Rows())
	fmt.Printf("img.Type(): %v\n", img.Type())

	// ShowImg(img)
	// split image channels
	bgr := gocv.Split(img)
	fmt.Printf("%+v\n", len(bgr))
	// pixel values for each channel - we know this is a BGR image  获取一点的像素值
	fmt.Printf("Pixel B: %d\n", bgr[0].GetUCharAt(100, 100))
	fmt.Printf("Pixel G: %d\n", bgr[1].GetUCharAt(100, 100))
	fmt.Printf("Pixel R: %d\n", bgr[2].GetUCharAt(100, 100))

	// gocv.Merge(bgr[2], bgr[1], bgr[0], &img)

}

func TestChannels(t *testing.T) {
	m := gocv.IMRead("./axis.png", gocv.IMReadGrayScale)
	fmt.Printf("m.Channels(): %v\n", m.Channels())
	fmt.Printf("m.Type(): %v\n", m.Type())
	fmt.Printf("m.Type().String(): %v\n", m.Type().String())
	fmt.Printf("m.GetUCharAt3(100, 100, 0): %v\n", m.GetUCharAt3(100, 100, 1)) // 数据要与图片的类型保持一致

	mats := gocv.Split(m)
	fmt.Printf("len(mats): %v\n", len(mats))
	fmt.Printf("mats[0].GetUCharAt(100, 100): %v\n", mats[0].GetUCharAt(100, 100))
	// gocv.Line(&m, image.Point{100, 100}, image.Point{300, 20}, colorMap["red"], 1)
	// gocv.IMWrite("self.png", m)

}

func TestKey(t *testing.T) {
	robotgo.KeyTap(`command`, `shift`, `3`)
	x, y := robotgo.GetMousePos()
	fmt.Println("pos: ", x, y)

	bit := robotgo.CaptureScreen(10, 10, 30, 30)
	defer robotgo.FreeBitmap(bit)

	img := robotgo.ToImage(bit)
	imgo.Save("test.png", img)
}



func TestErode(t *testing.T) {
	// 腐蚀 何为腐蚀, 腐蚀在于消除一些孤立点, 消除一些边界点. 使边界向内收缩. 我们可以借用腐蚀来消除无意义的小点.

	m := gocv.IMRead("./axis.png", gocv.IMReadGrayScale)
	defer m.Close()
	dst := gocv.NewMat()
	defer dst.Close()

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3)) // 返回指定大小和形状的结构元素，用于形态学操作。
	gocv.Erode(m, &dst, kernel)
	// gocv.ErodeWithParams(m, &dst, kernel, image.Point{-1, -1}, 3, 1)  // 腐蚀多次
	ShowImg(dst)
}

func TestDilate(t *testing.T) {
	//  膨胀 膨胀是将与物体接触的所有背景点合并到该物体中，使边界向外部扩张的过程。可以用来填补物体中的空洞。

	m := gocv.IMRead("./self.png", gocv.IMReadGrayScale)
	defer m.Close()
	dst := gocv.NewMat()
	defer dst.Close()

	kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(5, 5)) // 返回指定大小和形状的结构元素，用于形态学操作。
	gocv.Dilate(m, &dst, kernel)
	ShowImg(dst)
}

func Test_orphologyEx(t *testing.T) {
	// 开运算 开运算(opening) 等于对图像先进行腐蚀(erode) 然后进行膨胀(dilate).
	//开运算其主要作用与腐蚀相似，与腐蚀操作相比，具有可以基本保持目标原有大小不变的优点。
	// 开运算去除背景噪点的功效.

}

func Test_morphologyEx(t *testing.T) {
	// 闭运算  闭运算(closing) 是先对图像进行膨胀, 然后进行腐蚀操作.
	// 闭运算用来填充物体内细小空洞、连接邻近物体、平滑其边界的同时并不明显改变其面积。

	// 关于消除内部细小空洞的部分
}

func Test_inRange(t *testing.T) {
	// 二值化 将彩图转换成二值化图像
	// inRange 函数判断图像的像素点是否在阈值范围内.

	// 如果在阈值范围内该点的值就为逻辑1, 在灰度图中用值255表示. 如果在范围之外, 就为逻辑0, 用值0表示.

}

// 数学形态学梯度 = 图像膨胀 - 图像腐蚀 从而获取到图像的边缘.
