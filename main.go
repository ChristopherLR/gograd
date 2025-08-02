package main

import (
	"fmt"

	"github.com/ChristopherLR/gograd/grad"
)

var rng = grad.NewRNG(42)

//func main() {
//	tr, _, _ := grad.GenDataYinYang(rng, 1000, 0.1, 0.5)
//	fv := tr[0]
//	vals := make([]*grad.Value, 2)
//	vals[0] = grad.NewVal(fv.X)
//	vals[1] = grad.NewVal(fv.Y)
//
//	n := grad.NewNeuron(2, true)
//	out := n.Forward(vals)
//
//	fmt.Println(fv)
//	fmt.Println(n)
//
//	grad.PrintTree(out, 0)
//}

func main() {
	tr, _, _ := grad.GenDataYinYang(rng, 1000, 0.1, 0.5)
	fv := tr[0]
	vals := make([]*grad.Value, 2)
	vals[0] = grad.NewVal(fv.X)
	vals[1] = grad.NewVal(fv.Y)

	model := grad.NewMLP(2, []int{8, 3})
	fmt.Println(model)
	out := model.Forward(vals)

	fmt.Println(out)
}
