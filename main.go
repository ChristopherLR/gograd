package main

import "github.com/ChristopherLR/micrograd_go/grad"

func main() {
	a := grad.New(1.4)
	b := grad.New(1.3)

	c := a.Add(b)
	d := c.Add(a)
	d.Backward()

	grad.PrintTree(d, 0)
}
