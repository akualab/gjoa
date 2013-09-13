package floatx

import ()

type Error string

func (err Error) Error() string { return string(err) }

const (
	ErrIndexOutOfRange = Error("floatx: index out of range")
	ErrZeroLength      = Error("floatx: zero length in slice definition")
	ErrLength          = Error("floatx: length mismatch")
)

func MakeFloat3D(n1, n2, n3 int) [][][]float64 {

	s := make([][][]float64, n1)
	for i := 0; i < n1; i++ {
		s[i] = make([][]float64, n2)
		for j := 0; j < n2; j++ {
			s[i][j] = make([]float64, n3)
		}
	}

	return s
}

func MakeFloat2D(n1, n2 int) [][]float64 {

	s := make([][]float64, n1)
	for i := 0; i < n1; i++ {
		s[i] = make([]float64, n2)
	}

	return s
}

func Check2D(s [][]float64) (n1, n2 int) {

	n1 = len(s)
	if n1 == 0 {
		panic(ErrZeroLength)
	}

	n2 = len(s[0])
	if n2 == 0 {
		panic(ErrZeroLength)
	}

	return n1, n2
}

func Check3D(s [][][]float64) (n1, n2, n3 int) {

	n1 = len(s)
	if n1 == 0 {
		panic(ErrZeroLength)
	}

	n2 = len(s[0])
	if n2 == 0 {
		panic(ErrZeroLength)
	}

	n3 = len(s[0])
	if n3 == 0 {
		panic(ErrZeroLength)
	}

	return n1, n2, n3
}

type ApplyFunc func(n int, v float64) float64
type ApplyFunc2D func(n1, n2 int, v float64) float64
type ApplyFunc3D func(n1, n2, n3 int, v float64) float64

// Apply function to 1D slice. If out slice is empty, the function is applied in place.
func Apply(fn ApplyFunc, in, out []float64) []float64 {

	n := len(in)
	if n == 0 {
		panic(ErrZeroLength)
	}
	if len(out) == 0 {
		out = in
	}
	for i := 0; i < n; i++ {
		out[i] = fn(i, in[i])
	}

	return out
}

func SubSlice2D(s [][]float64, c int) []float64 {

	n1, n2 := Check2D(s)
	if c < 0 || c >= n2 {
		panic(ErrIndexOutOfRange)
	}
	out := make([]float64, n1)
	for i := 0; i < n1; i++ {
		out[i] = s[i][c]
	}
	return out
}

// Apply function to 2D slice. If out slice is empty, the function is applied in place.
func Apply2D(fn ApplyFunc2D, in, out [][]float64) [][]float64 {

	n1, n2 := Check2D(in)
	if len(out) == 0 {
		out = in
	}
	for i := 0; i < n1; i++ {
		for j := 0; j < n2; j++ {
			out[i][j] = fn(i, j, in[i][j])
		}
	}

	return out
}

// Apply function to 3D slice. If out slice is empty, the function is applied in place.
func Apply3D(fn ApplyFunc3D, in, out [][][]float64) [][][]float64 {

	n1, n2, n3 := Check3D(in)
	if len(out) == 0 {
		out = in
	}
	for i := 0; i < n1; i++ {
		for j := 0; j < n2; j++ {
			for k := 0; k < n3; k++ {
				out[i][j][k] = fn(i, j, k, in[i][j][k])
			}
		}
	}

	return out
}

func Flatten2D(s [][]float64) []float64 {

	n1, n2 := Check2D(s)
	out := make([]float64, n1*n2)

	p := 0
	for _, c := range s {
		copy(out[p:], c)
		p += len(c)
	}
	return out
}
