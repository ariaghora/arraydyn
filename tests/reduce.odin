package tests

import ar "../arraydyn"
import "core:slice"
import "core:testing"

@(test)
test_sum :: proc(t: ^testing.T) {
	// Test 1: Simple 2x3 array, sum along axis 0 (columns)
	{
		arr := ar.new_with_init(i32, []i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
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
		arr := ar.new_with_init(i32, []i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
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
		arr := ar.new_with_init(i32, []i32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
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
		arr := ar.new_with_init(i32, []i32{1, 2, 3, 4, 5, 6, 7, 8}, []uint{2, 2, 2})
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
}
