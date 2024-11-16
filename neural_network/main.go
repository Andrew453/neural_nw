package main

import (
	"encoding/csv"
	"fmt"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func main() {
	// Chihuahua - 1.0, Beagle - 2.0, Russian Hound - 0.0
	// features - height,weight,nose_length,ear_shape,breed
	// ear_shape: 2 - висячие, 1 - стоячие.
	data, err := LoadDataSetFromDOGSCSV("./dogs.csv")
	if err != nil {
		fmt.Println(err)
		return
	}
	data.Shuffle()

	n := deep.NewNeural(&deep.Config{
		Inputs: 4,
		Layout: []int{16, 64, 64, 3},
		// Задание функции активации
		Activation: deep.ActivationReLU,
		// Мульти классовый режим работы - то есть режим, в котором несколько классов
		Mode: deep.ModeMultiClass,
		// Задание изначальных весов
		Weight: deep.NewNormal(1.0, 0.0),
		// Задание функции потерь. В данном случае использую кросс энтропия - одна из самых популярных
		Loss: deep.LossCrossEntropy,
		// Задание использования смещения.
		Bias: true,
	})

	// Настройка алгоритма оптимизации обучения
	// params: learning rate, momentum, alpha decay, nesterov
	optimizer := training.NewAdam(0.01, 0, 0, 0)

	// params: optimizer, verbosity (print stats at every 50th iteration)
	trainer := training.NewTrainer(optimizer, 50)

	training, heldout := data.Split(0.75)
	trainer.Train(n, training, heldout, 100) // training, validation, iteration
	err = ResultToFile(n, data, "./result_dogs")
	if err != nil {
		fmt.Println(err)
		return
	}

	IrisClassification()
}

func IrisClassification() {
	data, err := LoadDataSetFromCSV("./iris.csv")
	if err != nil {
		fmt.Println(err)
		return
	}
	data.Shuffle()

	n := deep.NewNeural(&deep.Config{
		Inputs:     4,
		Layout:     []int{3, 3},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(1.0, 0.0),
		Loss:       deep.LossCrossEntropy,
		Bias:       true,
	})

	// params: learning rate, momentum, alpha decay, nesterov
	//optimizer := training.NewSGD(0.05, 0.1, 1e-6, true)
	optimizer := training.NewAdam(0.01, 0, 0, 0)
	// params: optimizer, verbosity (print stats at every 50th iteration)
	trainer := training.NewTrainer(optimizer, 50)

	training, heldout := data.Split(0.75)
	trainer.Train(n, training, heldout, 100) // training, validation, iteration
	err = ResultToFile(n, data, "./result")
	if err != nil {
		fmt.Println(err)
		return
	}
}

// LoadDataSetFromDOGSCSV Функция получения данных из CSV
func LoadDataSetFromDOGSCSV(path string) (data training.Examples, err error) {
	// Получение абсолютного пути
	absp, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	// Открытие файла
	file, err := os.Open(absp)
	if err != nil {
		return nil, err
	}
	// Создание нового ридера для CSV
	reader := csv.NewReader(file)

	for {
		var features []float64
		var label float64
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		for i, s := range record {
			// Запись породы
			if i == len(record)-1 {
				//	label
				label, err = strconv.ParseFloat(s, 64)
				if err != nil {
					break
				}
				continue
			}
			// запись параметров features
			f, err := strconv.ParseFloat(s, 64)
			if err != nil {
				break
			}
			features = append(features, f)
		}
		if err != nil {
			fmt.Println(err)
			continue
		}
		data = append(data, training.Example{
			Input:    features,
			Response: onehot(3, label),
		})
	}
	return data, err
}

func LoadDataSetFromCSV(path string) (data training.Examples, err error) {
	absp, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(absp)
	if err != nil {
		return nil, err
	}
	reader := csv.NewReader(file)

	for {
		var features []float64
		var label float64
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		for i, s := range record {
			if i == 0 {
				continue
			}
			if i == len(record)-1 {
				//	label
				label, err = strconv.ParseFloat(s, 64)
				if err != nil {
					break
				}
				continue
			}
			// features
			f, err := strconv.ParseFloat(s, 64)
			if err != nil {
				break
			}
			features = append(features, f)
		}
		if err != nil {
			fmt.Println(err)
			continue
		}
		data = append(data, training.Example{
			Input:    features,
			Response: onehot(3, label),
		})
	}
	return data, err
}

// Представление числа val в бинарном виде, записанном в массиве float. 2 при 3 классах = 001, 1 - 010, 0 - 100
func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)] = 1
	return res
}

func ArgMax(d []float64) (id int) {
	var m float64 = 0
	for i, f := range d {
		if f > m {
			id = i
			m = f
		}
	}
	return id
}

func toString(ex training.Example, result int) (str string) {
	var sb strings.Builder
	for _, i := range ex.Input {
		sb.WriteString(fmt.Sprintf("%f", i))
		sb.WriteString(", ")
	}
	sb.WriteString(fmt.Sprintf("%d", result))
	sb.WriteRune('\n')
	return sb.String()
}

func ResultToFile(n *deep.Neural, dataset training.Examples, name string) (err error) {
	f, err := os.OpenFile(name, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, example := range dataset {
		//fmt.Println(toString(example, ArgMax(n.Predict(example.Input))))
		//fmt.Println(example.Input, "=>", ArgMax(n.Predict(example.Input)))
		_, err = f.WriteString(toString(example, ArgMax(n.Predict(example.Input))))
		if err != nil {
			return err
		}
	}
	return nil
}
