package grad

type Point struct {
	X float64
	Y float64
	C uint8
}

func GenDataYinYang(random RNG, n int, rSmall float64, rBig float64) (train, val, test []Point) {
	pts := make([]Point, n)

	rBigSq, rSmallSq := rBig*rBig, rSmall*rSmall

	distSq := func(x, y, cx, cy float64) float64 {
		dx, dy := x-cx, y-cy
		return dx*dx + dy*dy
	}

	whichClass := func(x, y float64) uint8 {
		dRightSq := distSq(x, y, 1.5*rBig, rBig)
		dLeftSq := distSq(x, y, 0.5*rBig, rBig)

		isCircle := dRightSq < rSmallSq || dLeftSq < rSmallSq
		isYin := dRightSq <= rSmallSq ||
			(dLeftSq > rSmallSq && dLeftSq <= 0.25*rBigSq) ||
			(y > rBig && dRightSq > 0.25*rBigSq)

		switch {
		case isCircle:
			return 2
		case isYin:
			return 0
		default:
			return 1
		}
	}

	getSample := func(goalClass uint8) Point {
		for {
			x := random.Uniform(0, 2*rBig)
			y := random.Uniform(0, 2*rBig)

			if distSq(x, y, rBig, rBig) > rBigSq {
				continue
			}

			if c := whichClass(x, y); c == goalClass {
				return Point{
					X: (x/rBig - 1) * 2,
					Y: (y/rBig - 1) * 2,
					C: c,
				}
			}
		}
	}

	for i := range n {
		pts[i] = getSample(uint8(i % 3))
	}

	trainEnd := int(0.8 * float64(n))
	valEnd := int(0.9 * float64(n))

	train = pts[:trainEnd]
	val = pts[trainEnd:valEnd]
	test = pts[valEnd:]

	return
}
