package main

import (
	"fmt"

	"github.com/ChristopherLR/gograd/grad"
)

func crossEntropy(logits []*grad.Value, target uint8) *grad.Value {
	ex := make([]*grad.Value, len(logits))
	for i, v := range logits {
		ex[i] = v.Exp()
	}

	denom := grad.NewVal(0)
	for _, e := range ex {
		denom = denom.Add(e)
	}

	probs := make([]*grad.Value, len(logits))
	for i, e := range ex {
		probs[i] = e.Div(denom)
	}

	logp := probs[target].Log().Neg()
	return logp
}

func lossFun(model *grad.MLP, split []grad.Sample) *grad.Value {
	total := grad.NewVal(0)
	for _, s := range split {
		logits := model.Forward(s.X)
		loss := crossEntropy(logits, s.Y)
		total = total.Add(loss)
	}
	invN := 1.0 / float64(len(split))
	return total.Mul(grad.NewVal(invN))
}

var (
	trainSplit, valSplit, _ = grad.GenDataYinYang(grad.Random, 100, 0.1, 0.5)
	model                   = grad.NewMLP(2, []int{8, 3})
	optimizer               = grad.NewAdamW(model.Parameters(), 1e-1 /*LR*/, 1e-4 /*WeightDecay*/)
	steps                   = 100
)

func main() {
	for i := range steps {
		if i%10 == 0 {
			valLoss := lossFun(model, valSplit)
			fmt.Printf("step %d/%d loss %f\n", i, steps, valLoss.Data)
		}

		loss := lossFun(model, trainSplit)
		loss.Backward()

		optimizer.Step()
		optimizer.ZeroGrad()

		fmt.Printf("step %d/%d loss %f\n", i, steps, loss.Data)
	}
}
