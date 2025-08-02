package grad

import (
	"fmt"
	"math"
)

type Module interface {
	ZeroGrad()
	Parameters() []*Value
}

type Neuron struct {
	W []*Value
	B *Value
	Nonlin bool
}

var random = NewRNG(42)

func MakeNeuron(nin int, nonlin bool) *Neuron {
	weights := make([]*Value, nin)
	for i := range len(weights) {
		fmt.Println(i)
		weights[i] = NewVal(1/math.Sqrt(float64(nin)) * float64(random.Uniform(-1, 1)))
	}

	n := Neuron{
		W: weights,
		B: NewVal(0),
		Nonlin: nonlin,
	}
	return &n
}

func (n *Neuron) Parameters() []*Value {
	return append([]*Value{n.B}, n.W...)
}

func (n *Neuron) ZeroGrad() {
	for _, v := range n.Parameters() {
		v.Grad = 0.0
	}
}

