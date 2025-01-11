package tests

import ar "../arraydyn"

import "core:fmt"
import "core:math"
import "core:slice"
import "core:testing"

@(test)
test_add :: proc(t: ^testing.T) {
	a := ar._new_with_init([]i32{1, 2, 3}, {3})
	defer ar.array_free(a)
	b := ar._new_with_init([]i32{4, 5, 6}, {3})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	testing.expect(t, slice.equal(c.data, []i32{5, 7, 9}))

}

@(test)
test_add_broadacst :: proc(t: ^testing.T) {
	// (3,2) and (3,1)
	a := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(a)
	b := ar._new_with_init([]i32{10, 20, 30}, {3, 1})
	defer ar.array_free(b)
	c := ar.add(a, b)
	defer ar.array_free(c)
	res := ar._get_strided_data(c)
	defer delete(res)
	testing.expect(t, slice.equal(res, []i32{11, 12, 23, 24, 35, 36}))

	// (3,2) and (2)
	a1 := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(a1)
	b1 := ar._new_with_init([]i32{10, 20}, {2})
	defer ar.array_free(b1)
	c1 := ar.add(a1, b1)
	defer ar.array_free(c1)
	res1 := ar._get_strided_data(c1)
	defer delete(res1)
	testing.expect(t, slice.equal(res1, []i32{11, 22, 13, 24, 15, 26}))

	// Test broadcasting between (1,3) and (3,1)
	a3 := ar._new_with_init([]i32{1, 2, 3}, {1, 3}) // [1 2 3]
	defer ar.array_free(a3)
	b3 := ar._new_with_init([]i32{10, 20, 30}, {3, 1}) // [10; 20; 30]
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
	a4 := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2})
	defer ar.array_free(a4)
	b4 := ar._new_with_init([]i32{10, 20, 30, 40, 50, 60}, {3, 2})
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
	a := ar._new_with_init([]i32{4, 5, 6}, {3})
	defer ar.array_free(a)
	b := ar._new_with_init([]i32{1, 2, 3}, {3})
	defer ar.array_free(b)
	c := ar.sub(a, b)
	defer ar.array_free(c)
	testing.expect_value(t, c.data[0], 3)
	testing.expect_value(t, c.data[1], 3)
	testing.expect_value(t, c.data[2], 3)
}

@(test)
test_matmul :: proc(t: ^testing.T) {
	// (2,3) and (3,2) -> (2,2)
	a := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, {2, 3})
	defer ar.array_free(a)
	b := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, {3, 2})
	defer ar.array_free(b)
	c := ar.matmul(a, b)
	defer ar.array_free(c)
	res := ar._get_strided_data(c)
	defer delete(res)
	//
	// [1 2 3] [1 2]   [22 28]
	// [4 5 6] [3 4]   [49 64]
	//         [5 6]
	testing.expect(t, slice.equal(res, []i32{22, 28, 49, 64}))
}

@(test)
test_exp :: proc(t: ^testing.T) {
	a := ar._new_with_init([]f32{0, 1, 2}, {3})
	defer ar.array_free(a)
	c := ar.exp(a)
	defer ar.array_free(c)
	testing.expect(t, slice.equal(c.data, []f32{1.0, 2.718281828, 7.389056099}))

	a2 := ar._new_with_init([]f32{0, 1, 2, 3}, {2, 2})
	defer ar.array_free(a2)
	c2 := ar.exp(a2)
	defer ar.array_free(c2)
	res := ar._get_strided_data(c2)
	defer delete(res)
	testing.expect(t, slice.equal(res, []f32{1.0, 2.718281828, 7.389056099, 20.085536923}))
}

@(test)
test_softmax :: proc(t: ^testing.T) {
	// Test 1D softmax
	{
		a := ar._new_with_init([]f32{1.0, 2.0, 3.0}, {3})
		defer ar.array_free(a)
		s := ar.softmax(a, 0)
		defer ar.array_free(s)

		// sum(exp(x_i)) = e^1 + e^2 + e^3
		sum_exp := math.exp_f32(1.0) + math.exp_f32(2.0) + math.exp_f32(3.0)
		expected := []f32 {
			f32(math.exp_f32(1.0) / sum_exp),
			f32(math.exp_f32(2.0) / sum_exp),
			f32(math.exp_f32(3.0) / sum_exp),
		}
		for i in 0 ..< len(expected) {
			testing.expect(
				t,
				abs(s.data[i] - expected[i]) < 1e-6,
				fmt.tprintf("Expected %v, got %v", expected[i], s.data[i]),
			)
		}
	}

	// Test 2D softmax (along last dimension)
	{
		b := ar._new_with_init([]f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3})
		defer ar.array_free(b)
		s2 := ar.softmax(b, 1)
		defer ar.array_free(s2)

		// First row: sum(exp(x_i)) = e^1 + e^2 + e^3
		sum_exp1 := math.exp_f32(1.0) + math.exp_f32(2.0) + math.exp_f32(3.0)
		// Second row: sum(exp(x_i)) = e^4 + e^5 + e^6
		sum_exp2 := math.exp_f32(4.0) + math.exp_f32(5.0) + math.exp_f32(6.0)

		expected2 := []f32 {
			f32(math.exp_f32(1.0) / sum_exp1),
			f32(math.exp_f32(2.0) / sum_exp1),
			f32(math.exp_f32(3.0) / sum_exp1),
			f32(math.exp_f32(4.0) / sum_exp2),
			f32(math.exp_f32(5.0) / sum_exp2),
			f32(math.exp_f32(6.0) / sum_exp2),
		}

		res := ar._get_strided_data(s2)
		defer delete(res)
		for i in 0 ..< len(expected2) {
			testing.expect(
				t,
				abs(res[i] - expected2[i]) < 1e-6,
				fmt.tprintf("Expected %v, got %v", expected2[i], res[i]),
			)
		}
	}

	// Test 3D softmax (along last dimension)
	{
		c := ar._new_with_init(
			[]f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0},
			{2, 2, 3},
		)
		defer ar.array_free(c)
		s3 := ar.softmax(c, 2)
		defer ar.array_free(s3)

		// Row sums for each 3-element slice
		sum_exp1 := math.exp_f32(1.0) + math.exp_f32(2.0) + math.exp_f32(3.0)
		sum_exp2 := math.exp_f32(4.0) + math.exp_f32(5.0) + math.exp_f32(6.0)
		sum_exp3 := math.exp_f32(7.0) + math.exp_f32(8.0) + math.exp_f32(9.0)
		sum_exp4 := math.exp_f32(10.0) + math.exp_f32(11.0) + math.exp_f32(12.0)

		expected3 := []f32 {
			f32(math.exp_f32(1.0) / sum_exp1),
			f32(math.exp_f32(2.0) / sum_exp1),
			f32(math.exp_f32(3.0) / sum_exp1),
			f32(math.exp_f32(4.0) / sum_exp2),
			f32(math.exp_f32(5.0) / sum_exp2),
			f32(math.exp_f32(6.0) / sum_exp2),
			f32(math.exp_f32(7.0) / sum_exp3),
			f32(math.exp_f32(8.0) / sum_exp3),
			f32(math.exp_f32(9.0) / sum_exp3),
			f32(math.exp_f32(10.0) / sum_exp4),
			f32(math.exp_f32(11.0) / sum_exp4),
			f32(math.exp_f32(12.0) / sum_exp4),
		}

		res3 := ar._get_strided_data(s3)
		defer delete(res3)
		for i in 0 ..< len(expected3) {
			testing.expect(
				t,
				abs(res3[i] - expected3[i]) < 1e-6,
				fmt.tprintf("Expected %v, got %v", expected3[i], res3[i]),
			)
		}
	}
}
