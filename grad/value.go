package grad

import (
	"fmt"
	"math"
	"strings"
)

type Value struct {
	Data     float64
	Grad     float64
	prev     []*Value
	backward func()
	op       string
}

func NewVal(x float64) *Value { return &Value{Data: x} }

func (v *Value) String() string {
	if len(v.op) > 0 {
		return fmt.Sprintf("(d=%v, g=%v, op=%v)", v.Data, v.Grad, v.op)
	}
	return fmt.Sprintf("(d=%v, g=%v)", v.Data, v.Grad)
}

func (v *Value) Add(other *Value) *Value {
	out := Value{Data: v.Data + other.Data, prev: []*Value{v, other}, op: "+"}
	out.backward = func() {
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return &out
}

func (v *Value) Mul(other *Value) *Value {
	out := Value{Data: v.Data * other.Data, prev: []*Value{v, other}, op: "*"}
	out.backward = func() {
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return &out
}

func (v *Value) ReLU() *Value {
	out := Value{Data: v.Data, prev: []*Value{v}, op: "ReLU"}
	if v.Data < 0 {
		out.Data = 0
	}
	out.backward = func() {
		if out.Data > 0 {
			v.Grad += out.Grad
		}
	}
	return &out
}

func (v *Value) Tanh() *Value {
	out := Value{Data: math.Tanh(v.Data), prev: []*Value{v}, op: "tanh"}
	out.backward = func() {
		v.Grad += (1 - out.Data*out.Data) * out.Grad
	}
	return &out
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := map[*Value]bool{}

	var build_topo func(*Value)
	build_topo = func(val *Value) {
		if visited[val] {
			return
		}
		visited[val] = true
		for _, elem := range val.prev {
			build_topo(elem)
		}
		topo = append(topo, val)
	}

	build_topo(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward == nil {
			continue
		}
		topo[i].backward()
	}
}

func PrintTree(root *Value, depth int) {
	spaces := strings.Repeat(" ", depth)

	fmt.Println(spaces, root)
	for _, c := range root.prev {
		PrintTree(c, depth+2)
	}
}
