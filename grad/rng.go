package grad

type RNG struct {
	State uint64
}

func NewRNG(state uint64) RNG {
	return RNG{state}
}

func (r *RNG) RandomU32() uint64 {
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	// doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
	// doing & 0xFFFFFFFF is the same as cast to uint32 in C
	r.State ^= (r.State >> 12) & 0xFFFFFFFFFFFFFFFF
	r.State ^= (r.State << 25) & 0xFFFFFFFFFFFFFFFF
	r.State ^= (r.State >> 27) & 0xFFFFFFFFFFFFFFFF
	return ((r.State * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF
}

func (r *RNG) Random() float64 {
	return float64(r.RandomU32()>>8) / 16777216.0
}

func (r *RNG) Uniform(a, b float64) float64 {
	return a + (b-a)*r.Random()
}

var Random = NewRNG(42)
