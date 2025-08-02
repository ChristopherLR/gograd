package main

import (
	"github.com/ChristopherLR/gograd/grad"
	"fmt"
)

func main() {

	n := grad.MakeNeuron(4, true)

	fmt.Println(n)
}
