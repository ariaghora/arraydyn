package tests

import ar "../arraydyn"

import "core:testing"
@(test)
test_add :: proc(t: ^testing.T) {
	a := ar.new_with_init(i32, []i32{1, 2, 3}, {3})
	defer ar.array_free(a)
	b := ar.new_with_init(i32, []i32{4, 5, 6}, {3})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	testing.expect_value(t, c.data[0], 5)
	testing.expect_value(t, c.data[1], 7)
	testing.expect_value(t, c.data[2], 9)
}

@(test)
test_sub :: proc(t: ^testing.T) {
	a := ar.new_with_init(i32, []i32{4, 5, 6}, {3})
	defer ar.array_free(a)
	b := ar.new_with_init(i32, []i32{1, 2, 3}, {3})
	defer ar.array_free(b)
	c := ar.sub(a, b)
	defer ar.array_free(c)
	testing.expect_value(t, c.data[0], 3)
	testing.expect_value(t, c.data[1], 3)
	testing.expect_value(t, c.data[2], 3)
}
