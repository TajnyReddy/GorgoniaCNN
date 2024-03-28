package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/nfnt/resize"
	"gopkg.in/cheggaaa/pb.v1"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var classNameMap = make(map[string]int)
var dt tensor.Dtype = tensor.Float64
var err error

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

var (
	epochs    = flag.Int("epochs", 5, "Number of epochs to train for")
	batchsize = flag.Int("batchsize", 10, "Batch size")
)

func main() {
	dataDir := "/Users/Tomek/Desktop/Sekrecki_Tomasz_190752/flower_photos"
	trainImages, trainLabels, testImages, testLabels, err := CreateTensors(dataDir, 0.8)
	if err != nil {
		log.Fatal("Error creating slices:", err)
	}

	size := trainImages.Shape()
	fmt.Println("Train Set Size:", size)
	size = testImages.Shape()
	fmt.Println("Test Set Size:", size)
	train(trainImages, trainLabels, testImages, testLabels)
}

func train(inputs, targets, test1, test2 tensor.Tensor) {
	g := gorgonia.NewGraph()
	m := newConvNet(g)
	numExamples := inputs.Shape()[0]
	numTest := test1.Shape()[0]
	bs := *batchsize

	x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 5), gorgonia.WithName("y"))
	x1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 5), gorgonia.WithName("y"))

	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses := gorgonia.Must(gorgonia.BinaryXent(m.out, y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	// cost = gorgonia.Must(gorgonia.Neg(cost))

	vallosses := gorgonia.Must(gorgonia.BinaryXent(m.out, y1))
	valcost := gorgonia.Must(gorgonia.Mean(vallosses))
	// valcost = gorgonia.Must(gorgonia.Neg(valcost))

	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	var lossesVal gorgonia.Value
	gorgonia.Read(losses, &lossesVal)

	var valcostVal gorgonia.Value
	gorgonia.Read(valcost, &valcostVal)

	if _, err := gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)), gorgonia.WithLearnRate(0.00001))

	defer vm.Close()

	batches := numExamples / bs
	log.Printf("\n Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Millisecond * 100)
	bar.SetMaxWidth(70)
	fmt.Println("\n Training:")
	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(1)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}

			if err := xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)

			if err := vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch %d: %v", i, err)
			}

			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			bar.Increment()
			vm.Reset()

		}
		fmt.Printf("\t")
		log.Printf("Epoch %d | Cost %v", i, costVal)
	}
	bar.Finish()

	if err := m.fwd(x1); err != nil {
		log.Fatalf("%+v", err)
	}

	if _, err := gorgonia.Grad(valcost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	valbatches := numTest / bs
	log.Printf("\n Batches %d", valbatches)
	bar1 := pb.New(valbatches)
	bar1.SetRefreshRate(time.Millisecond * 100)
	bar1.SetMaxWidth(70)
	fmt.Println("\n Validating:")
	for i := 0; i < *epochs; i++ {
		bar1.Prefix(fmt.Sprintf("Epoch %d", i))
		bar1.Set(0)
		bar1.Start()
		for b := 0; b < valbatches; b++ {
			start := b * bs
			end := start + bs
			if start >= numTest {
				break
			}
			if end > numTest {
				end = numTest
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = test1.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = test2.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}

			if err := xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x1, xVal)
			gorgonia.Let(y1, yVal)

			if err := vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch %d: %v", i, err)
			}

			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar1.Increment()
		}
		fmt.Printf("\t")
		log.Printf("Epoch %d | Cost %v", i, valcostVal)
	}

	bar1.Finish()

	imagePath := "C:/Users/Tomek/Desktop/Sekrecki_Tomasz_190752/predict.jpg"
	imageTensor, err := loadImageAndPreprocess(imagePath)

	if err != nil {
		log.Fatal("Error loading and preprocessing image:", err)
	}

	x2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))

	bar2 := pb.New(1)
	bar2.SetRefreshRate(time.Millisecond * 100)
	bar2.SetMaxWidth(70)
	fmt.Println("Predicting:")
	bar2.Set(1)
	bar2.Start()
	gorgonia.Let(x2, imageTensor)

	if err := vm.RunAll(); err != nil {
		log.Fatalf("Failed at epoch %d: %v", 1, err)
	}

	if m.out.Value() == nil {
		log.Fatal("Output tensor is nil. Check the forward pass implementation.")
	}

	outputTensor := m.out.Value().Data().([]float64)
	predictedClass := getPredictedClass(outputTensor)

	fmt.Printf("\n Image: %s \n Predicted Class: %s\n", imagePath, predictedClass)
}

func CreateTensors(dataDir string, splitRatio float64) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor, error) {
	var images []tensor.Tensor
	var labels []tensor.Tensor

	var totalImages int

	err := filepath.Walk(dataDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Println("Error walking directory:", err)
			return err
		}

		if info.IsDir() {
			return nil
		}

		if strings.HasSuffix(strings.ToLower(info.Name()), ".jpg") {
			imagePath := path
			label := filepath.Base(filepath.Dir(path))

			labelTensor, err := stringToOneHot(label, 5)
			if err != nil {
				fmt.Println(err)
				return nil
			}

			imageTensor, err := resizeImage(imagePath)
			if err != nil {
				fmt.Println("Error loading image:", err)
				return nil
			}

			images = append(images, imageTensor)
			labels = append(labels, labelTensor)

			totalImages++
		}

		return nil
	})

	if err != nil {
		fmt.Println("Error walking directory:", err)
		return nil, nil, nil, nil, err
	}

	splitIndex := int(float64(len(images)) * splitRatio)

	trainImagesTensor, err := stackTensors(images[:splitIndex])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("Error stacking train images: %v", err)
	}

	trainLabelsTensor, err := stackTensors(labels[:splitIndex])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("Error stacking train labels: %v", err)
	}

	testImagesTensor, err := stackTensors(images[splitIndex:])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("Error stacking test images: %v", err)
	}

	testLabelsTensor, err := stackTensors(labels[splitIndex:])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("Error stacking test labels: %v", err)
	}

	return trainImagesTensor, trainLabelsTensor, testImagesTensor, testLabelsTensor, nil
}

func stackTensors(tensors []tensor.Tensor) (tensor.Tensor, error) {
	if len(tensors) == 0 {
		return nil, nil
	}

	result := tensors[0]
	for i := 1; i < len(tensors); i++ {
		var err error
		result, err = tensor.Concat(0, result, tensors[i])
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

func stringToOneHot(s string, numClasses int) (*tensor.Dense, error) {
	index := stringToIndex(s)
	if index == -1 {
		return nil, fmt.Errorf("unknown class name: %s", s)
	}

	oneHot := make([]float64, numClasses)
	oneHot[index] = 1.0

	tensorShape := []int{1, numClasses}
	labelTensor := tensor.New(tensor.WithShape(tensorShape...), tensor.Of(tensor.Float64), tensor.WithBacking(oneHot))

	return labelTensor, nil
}

func stringToIndex(s string) int {
	index, ok := classNameMap[s]
	if !ok {
		index = len(classNameMap)
		classNameMap[s] = index

	}
	return index
}

func loadImage(imagePath string) (image.Image, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		fmt.Println("Error opening image file:", err)
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Println("Error decoding image:", err)
		return nil, err
	}

	return img, nil
}

func resizeImage(imagePath string) (*tensor.Dense, error) {
	img, err := loadImage(imagePath)
	if err != nil {
		return nil, err
	}

	resizedImg := resize.Resize(28, 28, img, resize.Lanczos3)
	tensor := imageToTensor(resizedImg)

	return tensor, nil
}

func imageToTensor(img image.Image) *tensor.Dense {
	bounds := img.Bounds()
	rows, cols := bounds.Max.Y, bounds.Max.X

	data := make([]float64, rows*cols)
	idx := 0

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			grayValue := 0.299*float64(r)/65535.0 + 0.587*float64(g)/65535.0 + 0.114*float64(b)/65535.0
			data[idx] = grayValue
			idx++
		}
	}

	tensorShape := []int{1, 1, rows, cols}
	tensor := tensor.New(tensor.WithShape(tensorShape...), tensor.Of(tensor.Float32), tensor.WithBacking(data))
	return tensor
}

type convnet struct {
	g                  *gorgonia.ExprGraph
	w0, w1, w2, w3, w4 *gorgonia.Node // weights
	d0, d1, d2, d3     float64        //dropout

	out *gorgonia.Node
}

func newConvNet(g *gorgonia.ExprGraph) *convnet {
	w0 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(32, 1, 3, 3), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.HeN(1.0))) //relu
	w1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.HeN(1.0)))
	w2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(128, 64, 3, 3), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.HeN(1.0)))
	w3 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128*3*3, 625), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.HeN(1.0)))
	w4 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(625, 5), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.HeN(1.0)))
	return &convnet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,

		d0: 0.1,
		d1: 0.3,
		d2: 0.3,
		d3: 0.55,
	}
}

func (m *convnet) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3, m.w4}
}

func (m *convnet) fwd(x *gorgonia.Node) (err error) {
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node

	// LAYER 0
	c0, err = gorgonia.Conv2d(x, m.w0, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return err
	}
	a0, err = gorgonia.Rectify(c0)
	if err != nil {
		return err
	}
	p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return err
	}
	l0, err = gorgonia.Dropout(p0, m.d0)
	if err != nil {
		return err
	}

	// Layer 1
	c1, err = gorgonia.Conv2d(l0, m.w1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return err
	}
	a1, err = gorgonia.Rectify(c1)
	if err != nil {
		return err
	}
	p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return err
	}
	l1, err = gorgonia.Dropout(p1, m.d1)
	if err != nil {
		return err
	}

	// Layer 2
	c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return err
	}
	a2, err = gorgonia.Rectify(c2)
	if err != nil {
		return err
	}
	p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return err
	}

	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w})
	if err != nil {
		return err
	}
	l2, err = gorgonia.Dropout(r2, m.d2)
	if err != nil {
		return err
	}

	// Layer 3
	fc, err = gorgonia.Mul(l2, m.w3)
	if err != nil {
		return err
	}
	a3, err = gorgonia.Rectify(fc)
	if err != nil {
		return err
	}
	l3, err = gorgonia.Dropout(a3, m.d3)
	if err != nil {
		return err
	}

	// output decode
	var out *gorgonia.Node
	out, err = gorgonia.Mul(l3, m.w4)
	if err != nil {
		return err
	}

	m.out, err = gorgonia.Sigmoid(out)
	if err != nil {
		return err
	}

	return nil
}
func loadImageAndPreprocess(imagePath string) (*gorgonia.Node, error) {
	img, err := loadImage(imagePath)
	if err != nil {
		return nil, err
	}

	resizedImg := resize.Resize(28, 28, img, resize.Lanczos3)
	tensor := imageToTensor(resizedImg)

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))
	gorgonia.Let(x, tensor)

	return x, nil
}

func getPredictedClass(outputTensor []float64) string {

	maxIndex := argmax(outputTensor)
	predictedClass := reverseClassIndexLookup(maxIndex)
	return predictedClass
}

func argmax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}
	return maxIndex
}

func reverseClassIndexLookup(index int) string {
	for className, classIndex := range classNameMap {
		if classIndex == index {
			return className
		}
	}
	return "Unknown"
}
