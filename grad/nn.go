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
	W      []*Value
	B      *Value
	Nonlin bool
}

func NewNeuron(nin int, nonlin bool) *Neuron {
	weights := make([]*Value, nin)
	for i := range len(weights) {
		weights[i] = NewVal((1 / math.Sqrt(float64(nin))) * float64(Random.Uniform(-1, 1)))
	}

	n := Neuron{
		W:      weights,
		B:      NewVal(0),
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

func (n *Neuron) Forward(xs []*Value) *Value {
	if len(xs) != len(n.W) {
		panic(fmt.Sprintf("Foward: Expected %d inputs, got %d", len(n.W), len(xs)))
	}

	out := NewVal(0)

	for i := range len(n.W) {
		out = out.Add(n.W[i].Mul(xs[i]))
	}
	out = out.Add(n.B)

	if n.Nonlin {
		out = out.Tanh()
	}

	return out
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin int, nout int, nonlin bool) *Layer {
	neurons := make([]*Neuron, nout)
	for i := range nout {
		neurons[i] = NewNeuron(nin, nonlin)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Parameters() []*Value {
	total := 0
	for _, n := range l.Neurons {
		total += len(n.W) + 1 // bias
	}

	params := make([]*Value, 0, total)
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}

	return params
}

func (l *Layer) Forward(xs []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(xs)
	}
	return out
}

func (l *Layer) String() string {
	out := "Layer["
	for _, n := range l.Neurons {
		out += fmt.Sprintf("[%d, %v]", len(n.W), n.Nonlin)
	}
	out += "]"
	return out
}

type MLP struct {
	Layers []*Layer
}

// NewNLP(2, [8, 3])
func NewMLP(nin int, nouts []int) *MLP {
	layers := make([]*Layer, len(nouts))

	layers[0] = NewLayer(nin, nouts[0], true) // input
	for i := 1; i < len(nouts)-1; i++ {
		layers[i] = NewLayer(nouts[i-1], nouts[i], true)
	}
	layers[len(nouts)-1] = NewLayer(nouts[len(nouts)-2], nouts[len(nouts)-1], false) // output

	return &MLP{Layers: layers}
}

func (m *MLP) Forward(xs []*Value) []*Value {
	for _, l := range m.Layers {
		xs = l.Forward(xs)
	}
	return xs
}

func (m *MLP) Parameters() []*Value {
	params := []*Value{}
	for _, l := range m.Layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

func (m *MLP) String() string {
	out := "MLP[\n"
	for _, l := range m.Layers {
		out += fmt.Sprint("  ", l, "\n")
	}
	out += "]"
	return out
}
