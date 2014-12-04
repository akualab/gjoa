package floatx

import (
	"math"
)

type Error string

func (err Error) Error() string { return string(err) }

const (
	ErrIndexOutOfRange = Error("floatx: index out of range")
	ErrZeroLength      = Error("floatx: zero length in slice definition")
	ErrLength          = Error("floatx: length mismatch")
)

// Replace Inf and -Inf to +/- MaxFloat64 value.
func ConvertInf(value float64) float64 {
	if math.IsInf(value, 1) {
		return math.MaxFloat64
	}
	if math.IsInf(value, -1) {
		return -math.MaxFloat64
	}
	return value
}

// Replace Inf and -Inf values in slice to +/- MaxFloat64 value.
func ConvertInfSlice(values []float64) []float64 {

	for k, v := range values {
		values[k] = ConvertInf(v)
	}
	return values
}

// Replace Inf and -Inf values in slice of slices to +/- MaxFloat64 value.
func ConvertInfSlice2D(values [][]float64) [][]float64 {

	for k, v := range values {
		values[k] = ConvertInfSlice(v)
	}
	return values
}

var Inv = func(r int, v float64) float64 { return 1.0 / v }

func AddScalarFunc(f float64) ApplyFunc {
	return func(r int, v float64) float64 { return v + f }
}
func ScaleFunc(f float64) ApplyFunc {
	return func(r int, v float64) float64 { return v * f }
}
func SetValueFunc(f float64) ApplyFunc {
	return func(r int, v float64) float64 { return f }
}

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

func CopyFloat2D(s [][]float64) [][]float64 {

	n1, n2 := Check2D(s)
	out := MakeFloat2D(n1, n2)
	for i := 0; i < n1; i++ {
		copy(out[i], s[i])
	}

	return out
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

// Set all values to zero.
func Clear(s []float64) {

	Apply(SetValueFunc(0), s, nil)
}

// Set all values to zero.
func Clear2D(s [][]float64) {

	for _, slice := range s {
		Clear(slice)
	}
}

// Set all values to zero.
func Clear3D(s [][][]float64) {

	for _, slice := range s {
		Clear2D(slice)
	}
}

// A simple []float64 slice pool object.
// Use it to avoid allocating unecessary resources in
// concurrent code.
type Pool struct {
	n   int
	buf chan []float64
}

func NewPool(n int) *Pool {

	return &Pool{n, make(chan []float64, 1)}
}

func (pool *Pool) Get() []float64 {
	select {
	case b := <-pool.buf:
		return b
	default:
	}
	return make([]float64, pool.n)
}

func (pool *Pool) Put(p []float64) {
	select {
	case pool.buf <- p:
	default:
	}
}

// Log returns the natural logarithm, element-wise, of the elements of s, and stores in dst.
// Panics if the lengths of dst and s do not match.
func Log(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	for i, val := range s {
		dst[i] = math.Log(val)
	}
}

// Exp returns the exponential base-e, element-wise, of the elements of s, and stores in dst.
// Panics if the lengths of dst and s do not match.
func Exp(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	for i, val := range s {
		dst[i] = math.Exp(val)
	}
}

// Sq returns the square, element-wise, of the elements of s, and stores in dst.
// Panics if the lengths of dst and s do not match.
func Sq(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	for i, val := range s {
		dst[i] = val * val
	}
}

// Sqrt returns the square root, element-wise, of the elements of s, and stores in dst.
// Panics if the lengths of dst and s do not match.
func Sqrt(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	for i, val := range s {
		dst[i] = math.Sqrt(val)
	}
}

// Abs returns the absolute value, element-wise, of the elements of s, and stores in dst.
// Panics if the lengths of dst and s do not match.
func Abs(dst, s []float64) {
	if len(dst) != len(s) {
		panic("floats: length of the slices do not match")
	}
	for i, val := range s {
		dst[i] = math.Abs(val)
	}
}
