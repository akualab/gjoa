package cache

import (
	"testing"
)

func TestCacheInitF(t *testing.T) {
	cache := NewCache(5)
	sz, c, _ := cache.Stats()
	if sz != 0 {
		t.Errorf("size = %v, want 0", sz)
	}
	if c != 5 {
		t.Errorf("capacity = %v, want 5", c)
	}
}

func TestSetInsertsValueF(t *testing.T) {
	cache := NewCache(100)
	data := []float64{1.1, 2.2, 3.3}
	var key uint64 = 33
	cache.Set(key, data)

	v, ok := cache.Get(key)
	if !ok {
		t.Errorf("Cache returned not ok")
	}

	for i, f := range []float64(v) {
		if f != data[i] {
			t.Errorf("Cache has incorrect value: %f != %f", data[i], []float64(v)[i])
		}
	}

}
