package tests

import ar "../arraydyn"
import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_init :: proc(t: ^testing.T) {
	empty := ar._array_alloc(i32, {2, 3})
	defer ar.array_free(empty)
	m_ones := ar._ones(f16, {2, 3})
	defer ar.array_free(m_ones)

	empty_t := ar.new_with_init([]i32{1, 2, 3}, {3})
	ar.tensor_release(empty_t)

	t_ones := ar._tensor_from_array(ar._ones(f16, {2, 3}))
	defer ar.tensor_release(t_ones)
	testing.expect(t, slice.equal(t_ones.shape, []uint{2, 3}))
}

@(test)
test_strided_data_extract :: proc(t: ^testing.T) {
	arr := ar._new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.array_free(arr)
	arr.shape[0], arr.shape[1] = arr.shape[1], arr.shape[0]
	arr.strides[0], arr.strides[1] = arr.strides[1], arr.strides[0]
	arr.contiguous = false
	transposed := ar._get_strided_data(arr)
	defer delete(transposed)
	testing.expect_value(t, transposed[0], 1)
	testing.expect_value(t, transposed[1], 3)
	testing.expect_value(t, transposed[2], 2)
	testing.expect_value(t, transposed[3], 4)
}

@(test)
test_reshape :: proc(t: ^testing.T) {
	// Test basic reshape
	x := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, {2, 3})
	defer ar.tensor_release(x)
	ar.set_requires_grad(x, true)

	// Reshape to different dimensions
	y := ar.reshape(x, {3, 2})
	defer ar.tensor_release(y)

	// Verify shape changed but data preserved
	testing.expect(t, slice.equal(y.shape, []uint{3, 2}))
	expected := []f32{1, 2, 3, 4, 5, 6}
	testing.expect(t, slice.equal(y.data, expected))

	// Test gradient flow
	z := ar.sum(y)
	defer ar.tensor_release(z)
	ar.backward(z)

	// Gradient should maintain original shape
	testing.expect(t, slice.equal(x.grad.shape, []uint{2, 3}))
	expected_grad := []f32{1, 1, 1, 1, 1, 1}
	testing.expect(t, slice.equal(x.grad.data, expected_grad))
}

R :: ar.Range

@(test)
test_slice_1d :: proc(t: ^testing.T) {
	// Test 1D array slicing
	arr := ar.new_with_init([]f32{1, 2, 3, 4, 5}, {5})
	defer ar.tensor_release(arr)

	// Slice [1:3] -> [2, 3]
	s1 := ar.slice(arr, R{1, 3})
	defer ar.tensor_release(s1)
	testing.expect(
		t,
		slice.equal(s1.data[:2], []f32{2, 3}),
		fmt.tprintf("Expected [2, 3], got %v", s1.data[:2]),
	)
	testing.expect(t, slice.equal(s1.shape, []uint{2}))

	// Slice [2:5] -> [3, 4, 5]
	s2 := ar.slice(arr, R{2, 5})
	defer ar.tensor_release(s2)
	testing.expect(
		t,
		slice.equal(s2.data[:3], []f32{3, 4, 5}),
		fmt.tprintf("Expected [3, 4, 5], got %v", s2.data[:3]),
	)
}

@(test)
test_slice_2d :: proc(t: ^testing.T) {
	// Test 2D array slicing
	arr := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3})
	defer ar.tensor_release(arr)

	// Slice rows [0:2] -> [[1, 2, 3], [4, 5, 6]]
	s1 := ar.slice(arr, {0, 2})
	defer ar.tensor_release(s1)
	testing.expect(t, slice.equal(s1.shape, []uint{2, 3}))
	testing.expect(
		t,
		ar.array_get(s1.arrdata, 1, 1) == 5,
		fmt.tprintf("Expected s1[1,1] = 5, got %v", ar.array_get(s1.arrdata, 1, 1)),
	)

	// Slice both dims [1:3, 1:3] -> [[5, 6], [8, 9]]
	s2 := ar.slice(arr, R{1, 3}, R{1, 3})
	defer ar.tensor_release(s2)
	testing.expect(t, slice.equal(s2.shape, []uint{2, 2}))
	testing.expect(
		t,
		ar.array_get(s2.arrdata, 0, 0) == 5,
		fmt.tprintf("Expected s2[0,0] = 5, got %v", ar.array_get(s2.arrdata, 0, 0)),
	)
}

@(test)
test_slice_3d :: proc(t: ^testing.T) {
	// Test 3D array slicing
	arr := ar.new_with_init(
		[]f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		{3, 2, 2}, // 3 planes, 2 rows each, 2 cols each
	)
	defer ar.tensor_release(arr)

	// Slice first dimension [1:3] -> planes 2 and 3
	s1 := ar.slice(arr, {1, 3})
	defer ar.tensor_release(s1)
	testing.expect(t, slice.equal(s1.shape, []uint{2, 2, 2}))
	testing.expect(
		t,
		ar.array_get(s1.arrdata, 0, 0, 0) == 5,
		fmt.tprintf("Expected s1[0,0,0] = 5, got %v", ar.array_get(s1.arrdata, 0, 0, 0)),
	)

	// Slice all dimensions [1:3, 0:1, 1:2] -> [[[6], [8]], [[10], [12]]]
	s2 := ar.slice(arr, R{1, 3}, R{0, 1}, R{1, 2})
	defer ar.tensor_release(s2)
	testing.expect(t, slice.equal(s2.shape, []uint{2, 1, 1}))
	testing.expect(
		t,
		ar.array_get(s2.arrdata, 0, 0, 0) == 6,
		fmt.tprintf("Expected s2[0,0,0] = 6, got %v", ar.array_get(s2.arrdata, 0, 0, 0)),
	)
}
