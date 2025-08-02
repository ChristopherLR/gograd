package grad

import "math"

type adamMoment struct {
	m, v float64
}

type AdamW struct {
	Parameters   []*Value
	Moments      map[*Value]*adamMoment
	LR           float64
	Beta1, Beta2 float64
	Eps          float64
	WeightDecay  float64
	T            int
}

func NewAdamW(parameters []*Value, lr float64) *AdamW {
	moments := make(map[*Value]*adamMoment, len(parameters))
	for _, p := range parameters {
		moments[p] = &adamMoment{m: 0, v: 0}
	}
	return &AdamW{
		Parameters:  parameters,
		Moments:     moments,
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.95,
		Eps:         1e-8,
		WeightDecay: 0.0,
		T:           0,
	}
}

func (a *AdamW) Step() {
	a.T++
	for _, p := range a.Parameters {
		moment := a.Moments[p]
		moment.m = a.Beta1*moment.m + (1-a.Beta1)*p.Grad
		moment.v = a.Beta2*moment.v + (1-a.Beta2)*p.Grad*p.Grad
		t := float64(a.T)
		mHat := moment.m / (1 - math.Pow(a.Beta1, t))
		vHat := moment.v / (1 - math.Pow(a.Beta2, t))
		p.Data -= a.LR * (mHat/(math.Sqrt(vHat)+a.Eps) + a.WeightDecay*p.Data)
	}
}

func (a *AdamW) ZeroGrad() {
	for _, p := range a.Parameters {
		p.Grad = 0
	}
}
