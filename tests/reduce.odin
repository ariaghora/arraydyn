package tests

import ar "../arraydyn"
import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_sum :: proc(t: ^testing.T) {
	// Test 1: Simple 2x3 array, sum along axis 0 (columns)
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)

		sum0 := ar.sum(arr, 0) // Should be [5,7,9]
		defer ar.array_free(sum0)

		testing.expect_value(t, len(sum0.shape), 1)
		testing.expect_value(t, sum0.shape[0], 3)
		res := ar._get_strided_data(sum0)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{5, 7, 9}))
	}

	// Test 2: Same array, sum along axis 1 (rows)
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)

		sum1 := ar.sum(arr, 1) // Should be [6,15]
		defer ar.array_free(sum1)

		testing.expect_value(t, len(sum1.shape), 1)
		testing.expect_value(t, sum1.shape[0], 2)
		res := ar._get_strided_data(sum1)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{6, 15}))
	}

	// Test 3: With keepdims=true
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)

		sum0 := ar.sum(arr, 0, keepdims = true) // Should be [[5,7,9]]
		defer ar.array_free(sum0)

		testing.expect_value(t, len(sum0.shape), 2)
		testing.expect_value(t, sum0.shape[0], 1)
		testing.expect_value(t, sum0.shape[1], 3)
		res := ar._get_strided_data(sum0)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{5, 7, 9}))
	}

	// Test 4: 3D array test
	{
		// Create 2x2x2 array
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6, 7, 8}, []uint{2, 2, 2})
		defer ar.array_free(arr)

		sum1 := ar.sum(arr, 1) // Sum middle dimension
		defer ar.array_free(sum1)

		testing.expect_value(t, len(sum1.shape), 2)
		testing.expect_value(t, sum1.shape[0], 2)
		testing.expect_value(t, sum1.shape[1], 2)
		res := ar._get_strided_data(sum1)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{4, 6, 12, 14}))
	}

	// Test 5: Non-contiguous (transposed) array test
	{
		// Create 2x3 array but transpose it by swapping strides
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)
		arr.contiguous = false
		arr.shape[0], arr.shape[1] = arr.shape[1], arr.shape[0]
		arr.strides[0], arr.strides[1] = arr.strides[1], arr.strides[0]

		// Original array in memory: [1,2,3,4,5,6]
		// Viewed as:
		// 1 2 3  ->  1 4
		// 4 5 6      2 5
		//            3 6

		// Sum along axis 0 (now summing pairs of numbers in columns)
		sum0 := ar.sum(arr, 0) // Should be [6, 15]
		defer ar.array_free(sum0)

		testing.expect_value(t, len(sum0.shape), 1)
		testing.expect_value(t, sum0.shape[0], 2)
		res := ar._get_strided_data(sum0)

		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{6, 15})) // 1+4=5, 2+5=7, 3+6=9

		// Sum along axis 1
		sum1 := ar.sum(arr, 1) // Should be [3,7,11]
		defer ar.array_free(sum1)

		testing.expect_value(t, len(sum1.shape), 1)
		testing.expect_value(t, sum1.shape[0], 3)
		res1 := ar._get_strided_data(sum1)
		defer delete(res1)
		testing.expect(t, slice.equal(res1, []i32{5, 7, 9})) // 1+2+3=6, 4+5+6=15
	}
}

@(test)
test_max_min :: proc(t: ^testing.T) {
	// Test 1: Simple 2x3 array, max along axis 0
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)

		max0 := ar.max(arr, 0) // Should be [4,5,6]
		defer ar.array_free(max0)

		testing.expect_value(t, len(max0.shape), 1)
		testing.expect_value(t, max0.shape[0], 3)
		res := ar._get_strided_data(max0)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{4, 5, 6}))
	}

	// Test 2: Same array, min along axis 1
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)

		min1 := ar.min(arr, 1) // Should be [1,4]
		defer ar.array_free(min1)

		testing.expect_value(t, len(min1.shape), 1)
		testing.expect_value(t, min1.shape[0], 2)
		res := ar._get_strided_data(min1)
		expected := []i32{1, 4}
		defer delete(res)
		testing.expect(t, slice.equal(res, expected), fmt.tprintf("%v vs %v", res, expected))
	}

	// Test 3: With keepdims=true
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)

		max0 := ar.max(arr, 0, keepdims = true) // Should be [[4,5,6]]
		defer ar.array_free(max0)

		testing.expect_value(t, len(max0.shape), 2)
		testing.expect_value(t, max0.shape[0], 1)
		testing.expect_value(t, max0.shape[1], 3)
		res := ar._get_strided_data(max0)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{4, 5, 6}))
	}

	// Test 4: 3D array test
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6, 7, 8}, []uint{2, 2, 2})
		defer ar.array_free(arr)

		min1 := ar.min(arr, 1) // Min middle dimension
		defer ar.array_free(min1)

		testing.expect_value(t, len(min1.shape), 2)
		testing.expect_value(t, min1.shape[0], 2)
		testing.expect_value(t, min1.shape[1], 2)
		res := ar._get_strided_data(min1)
		defer delete(res)
		expected := []i32{1, 2, 5, 6}
		testing.expect(t, slice.equal(res, expected), fmt.tprintf("%v vs %v", res, expected))
	}

	// Test 5: Non-contiguous array
	{
		arr := ar._new_with_init([]i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		defer ar.array_free(arr)
		arr.contiguous = false
		arr.shape[0], arr.shape[1] = arr.shape[1], arr.shape[0]
		arr.strides[0], arr.strides[1] = arr.strides[1], arr.strides[0]

		max1 := ar.max(arr, 1)
		defer ar.array_free(max1)

		testing.expect_value(t, len(max1.shape), 1)
		testing.expect_value(t, max1.shape[0], 3)
		res := ar._get_strided_data(max1)
		defer delete(res)
		testing.expect(t, slice.equal(res, []i32{4, 5, 6}))
	}
}

@(test)
test_sum_autograd :: proc(t: ^testing.T) {
	// Test basic sum gradient
	{
		x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		ar.set_requires_grad(x, true)
		defer ar.tensor_release(x)

		y := ar.sum(x, 1) // Sum rows: [6, 15]
		defer ar.tensor_release(y)

		ar.backward(y)

		expected_grad := ar._new_with_init([]f32{1, 1, 1, 1, 1, 1}, []uint{2, 3})
		defer ar.array_free(expected_grad)

		testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
	}

	// Test sum gradient with keepdims
	{
		x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		ar.set_requires_grad(x, true)
		defer ar.tensor_release(x)

		y := ar.sum(x, 0, keepdims = true) // Sum columns with keepdims: [[5, 7, 9]]
		defer ar.tensor_release(y)

		ar.backward(y)

		expected_grad := ar._new_with_init([]f32{1, 1, 1, 1, 1, 1}, []uint{2, 3})
		defer ar.array_free(expected_grad)

		testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
	}

	// Test sum gradient without axis (global sum)
	{
		x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		ar.set_requires_grad(x, true)
		defer ar.tensor_release(x)

		y := ar.sum(x) // Global sum: 21
		testing.expect_value(t, y.data[0], 21) // Verify sum is 21
		defer ar.tensor_release(y)

		ar.backward(y)

		expected_grad := ar._new_with_init([]f32{1, 1, 1, 1, 1, 1}, []uint{2, 3})
		defer ar.array_free(expected_grad)

		testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
	}
}

@(test)
test_mean_autograd :: proc(t: ^testing.T) {
	// Test basic mean gradient
	{
		x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		ar.set_requires_grad(x, true)
		defer ar.tensor_release(x)

		y := ar.mean(x, 1) // Mean of rows: [2, 5]
		defer ar.tensor_release(y)

		ar.backward(y)

		// Gradient should be 1/3 since mean divides by 3 (axis size)
		expected_grad := ar._new_with_init(
			[]f32{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
			[]uint{2, 3},
		)
		defer ar.array_free(expected_grad)

		testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
	}

	// Test mean gradient with keepdims
	{
		x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		ar.set_requires_grad(x, true)
		defer ar.tensor_release(x)

		y := ar.mean(x, 0, keepdims = true) // Mean of columns with keepdims: [[2.5, 3.5, 4.5]]
		defer ar.tensor_release(y)

		ar.backward(y)

		// Gradient should be 1/2 since mean divides by 2 (axis size)
		expected_grad := ar._new_with_init([]f32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, []uint{2, 3})
		defer ar.array_free(expected_grad)

		testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
	}

	// Test mean without axis (global mean)
	{
		x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
		ar.set_requires_grad(x, true)
		defer ar.tensor_release(x)

		y := ar.mean(x) // Global mean: 3.5
		defer ar.tensor_release(y)

		ar.backward(y)

		// Gradient should be 1/6 since mean divides by total size (2*3=6)
		expected_grad := ar._new_with_init(
			[]f32{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0},
			[]uint{2, 3},
		)
		defer ar.array_free(expected_grad)

		testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
	}
}

@(test)
test_argmax :: proc(t: ^testing.T) {
	// Test case 1: Simple 2D array, axis 1
	{
		data := []f32{1, 5, 3, 2, 8, 4, 7, 3, 6}
		arr := ar._new_with_init(data, []uint{3, 3})
		indices := ar.argmax_axis_a(arr, axis = 1, keepdims = false)
		defer ar.array_free(arr, indices)

		expected := []f32{1, 1, 0} // Expected indices of max values
		testing.expect_value(t, indices.shape[0], uint(3))
		for i := 0; i < len(expected); i += 1 {
			testing.expect_value(t, indices.data[i], expected[i])
		}
	}

	// Test case 2: 2D array with obvious max positions, axis 0
	{
		data := []f32{1, 9, 3, 4, 2, 8}
		arr := ar._new_with_init(data, []uint{2, 3})
		indices := ar.argmax_axis_a(arr, axis = 0, keepdims = false)
		defer ar.array_free(arr, indices)

		expected := []f32{1, 0, 1} // Max along columns
		testing.expect_value(t, indices.shape[0], uint(3))
		for i := 0; i < len(expected); i += 1 {
			testing.expect_value(t, indices.data[i], expected[i])
		}
	}

	// Test case 3: 1D array
	{
		data := []f32{3, 1, 4, 1, 5, 9, 2}
		arr := ar._new_with_init(data, []uint{7})
		indices := ar.argmax_axis_a(arr, axis = 0, keepdims = false)
		defer ar.array_free(arr, indices)

		expected := []f32{5} // Max at index 5 (value 9)
		testing.expect_value(t, indices.data[0], expected[0])
	}

	// Test case 4: Test keepdims=true
	{
		data := []f32{1, 5, 3}
		arr := ar._new_with_init(data, []uint{1, 3})
		indices := ar.argmax_axis_a(arr, axis = 1, keepdims = true)
		defer ar.array_free(arr, indices)

		// Shape should be [1, 1] when keepdims=true
		testing.expect_value(t, len(indices.shape), 2)
		testing.expect_value(t, indices.shape[0], uint(1))
		testing.expect_value(t, indices.shape[1], uint(1))
		testing.expect_value(t, indices.data[0], f32(1)) // Max at index 1 (value 5)
	}

	// Test case 5: 3D array in CHW format (3 channels, 2x2 image)
	{
		// Shape: [3, 2, 2] (C=3, H=2, W=2)
		// Channel 0: [[1, 2],
		//            [3, 4]]
		// Channel 1: [[6, 5],
		//            [4, 3]]
		// Channel 2: [[2, 1],
		//            [5, 2]]
		data := []f32 {
			// Channel 0
			1,
			2,
			3,
			4,
			// Channel 1
			6,
			5,
			4,
			3,
			// Channel 2
			2,
			1,
			5,
			2,
		}
		arr := ar._new_with_init(data, []uint{3, 2, 2})
		indices := ar.argmax_axis_a(arr, axis = 0, keepdims = false) // axis 0 is channel
		defer ar.array_free(arr, indices)

		// Expected result shape: [2, 2] (H, W)
		testing.expect_value(t, len(indices.shape), 2)
		testing.expect_value(t, indices.shape[0], uint(2)) // Height
		testing.expect_value(t, indices.shape[1], uint(2)) // Width

		// Expected max channel indices:
		// [[1, 1], -- Channel 1 has max (6,5)
		//  [2, 0]] -- Channel 2 has max (5), Channel 0 has max (4)
		expected := []f32 {
			1,
			1, // top row
			2,
			0, // bottom row
		}

		for i := 0; i < len(expected); i += 1 {
			testing.expect_value(t, indices.data[i], expected[i])
		}
	}
}
