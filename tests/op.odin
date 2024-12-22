package tests

import ar "../arraydyn"

import "core:slice"
import "core:testing"

@(test)
test_add :: proc(t: ^testing.T) {
	a := ar.new_with_init(i32, []i32{1, 2, 3}, {3})
	defer ar.array_free(a)
	b := ar.new_with_init(i32, []i32{4, 5, 6}, {3})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	testing.expect(t, slice.equal(c.data, []i32{5, 7, 9}))

}

@(test)
test_add_broadacst :: proc(t: ^testing.T) {
	// (3,2) and (3,1)
	a := ar.new_with_init(i32, []i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(a)
	b := ar.new_with_init(i32, []i32{10, 20, 30}, {3, 1})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	res := ar._get_strided_data(c)
	defer delete(res)
	testing.expect(t, slice.equal(res, []i32{11, 12, 23, 24, 35, 36}))

	// (3,2) and (2)
	a1 := ar.new_with_init(i32, []i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(a1)
	b1 := ar.new_with_init(i32, []i32{10, 20}, {3})
	defer ar.array_free(b1)
	c1 := ar.add(a1, b1)
	defer ar.array_free(c1)
	res1 := ar._get_strided_data(c1)
	defer delete(res1)
	testing.expect(t, slice.equal(res1, []i32{11, 22, 13, 24, 15, 26}))
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
