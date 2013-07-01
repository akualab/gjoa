package cache

// LRU Cache with key of type uint64 and value of type []*float64.
// For now this is a front end that uses the LRU cache from the Vitess project.
//  https://code.google.com/p/vitess
//
// We will replace this implementation with a circular buffer of []float64

import (
	"strconv"
	"time"
)

type Cache struct {
	lruCache *LRUCache
}

type FValue struct{ v []float64 }

func (fv *FValue) Size() int {
	return 1
}

func NewCache(cap uint64) *Cache {

	return &Cache{
		lruCache: NewLRUCache(cap),
	}
}

func (c *Cache) Stats() (size, capacity uint64, oldest time.Time) {

	_, size, capacity, oldest = c.lruCache.Stats()

	return
}

func (c *Cache) Set(n uint64, v []float64) {

	cv := &FValue{v}
	key := strconv.FormatUint(n, 10)
	c.lruCache.Set(key, cv)
}

func (c *Cache) SetIfAbsent(n uint64, v []float64) {

	cv := &FValue{v}
	key := strconv.FormatUint(n, 10)
	c.lruCache.SetIfAbsent(key, cv)
}

func (c *Cache) Get(n uint64) (v []float64, ok bool) {

	key := strconv.FormatUint(n, 10)
	cv, ok := c.lruCache.Get(key)
	if ok {
		fv := cv.(*FValue)
		v = fv.v
	}
	return
}

func (c *Cache) Delete(n uint64) bool {

	key := strconv.FormatUint(n, 10)
	return c.lruCache.Delete(key)
}

func (c *Cache) Clear() {
	c.lruCache.Clear()
}
