package tests

import ar "../arraydyn"

import "core:math"
import "core:slice"
import "core:testing"

@(test)
test_add :: proc(t: ^testing.T) {
	a := ar.new_with_init([]i32{1, 2, 3}, {3})
	defer ar.array_free(a)
	b := ar.new_with_init([]i32{4, 5, 6}, {3})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	testing.expect(t, slice.equal(c.data, []i32{5, 7, 9}))

}

@(test)
test_add_broadacst :: proc(t: ^testing.T) {
	// (3,2) and (3,1)
	a := ar.new_with_init([]i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(a)
	b := ar.new_with_init([]i32{10, 20, 30}, {3, 1})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	res := ar._get_strided_data(c)
	defer delete(res)
	testing.expect(t, slice.equal(res, []i32{11, 12, 23, 24, 35, 36}))

	// (3,2) and (2)
	a1 := ar.new_with_init([]i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(a1)
	b1 := ar.new_with_init([]i32{10, 20}, {2})
	defer ar.array_free(b1)
	c1 := ar.add(a1, b1)
	defer ar.array_free(c1)
	res1 := ar._get_strided_data(c1)
	defer delete(res1)
	testing.expect(t, slice.equal(res1, []i32{11, 22, 13, 24, 15, 26}))

	// Test broadcasting between (1,3) and (3,1)
	a3 := ar.new_with_init([]i32{1, 2, 3}, {1, 3}) // [1 2 3]
	defer ar.array_free(a3)
	b3 := ar.new_with_init([]i32{10, 20, 30}, {3, 1}) // [10; 20; 30]
	defer ar.array_free(b3)
	c3 := ar.add(a3, b3)
	defer ar.array_free(c3)
	res3 := ar._get_strided_data(c3)
	defer delete(res3)

	// Broadcasting result should be:
	//      [1 2 3]
	// +[10]       --> [11 12 13]
	// +[20]       --> [21 22 23]
	// +[30]       --> [31 32 33]
	//
	// Result in row-major order: [11,12,13, 21,22,23, 31,32,33]
	expected := []i32{11, 12, 13, 21, 22, 23, 31, 32, 33}
	testing.expect(t, slice.equal(res3, expected))

	// Test broadcasting between (2,3,2) and (3,2)
	// [[[ 1  2]      [[10 20]
	//   [ 3  4]   +   [30 40]
	//   [ 5  6]]      [50 60]]
	//  [[ 7  8]         ...
	//   [ 9 10]         ...
	//   [11 12]]]       ...
	a4 := ar.new_with_init([]i32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2})
	defer ar.array_free(a4)
	b4 := ar.new_with_init([]i32{10, 20, 30, 40, 50, 60}, {3, 2})
	defer ar.array_free(b4)
	c4 := ar.add(a4, b4)
	defer ar.array_free(c4)
	res4 := ar._get_strided_data(c4)
	defer delete(res4)

	expected4 := []i32{11, 22, 33, 44, 55, 66, 17, 28, 39, 50, 61, 72}
	testing.expect(t, slice.equal(res4, expected4))

}

@(test)
test_sub :: proc(t: ^testing.T) {
	a := ar.new_with_init([]i32{4, 5, 6}, {3})
	defer ar.array_free(a)
	b := ar.new_with_init([]i32{1, 2, 3}, {3})
	defer ar.array_free(b)
	c := ar.sub(a, b)
	defer ar.array_free(c)
	testing.expect_value(t, c.data[0], 3)
	testing.expect_value(t, c.data[1], 3)
	testing.expect_value(t, c.data[2], 3)
}

@(test)
test_exp :: proc(t: ^testing.T) {
	a := ar.new_with_init([]f32{0, 1, 2}, {3})
	defer ar.array_free(a)
	c := ar.exp(a)
	defer ar.array_free(c)
	testing.expect(t, slice.equal(c.data, []f32{1.0, 2.718281828, 7.389056099}))

	a2 := ar.new_with_init([]f32{0, 1, 2, 3}, {2, 2})
	defer ar.array_free(a2)
	c2 := ar.exp(a2)
	defer ar.array_free(c2)
	res := ar._get_strided_data(c2)
	defer delete(res)
	testing.expect(t, slice.equal(res, []f32{1.0, 2.718281828, 7.389056099, 20.085536923}))
}
