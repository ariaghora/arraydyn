package tests

import ar "../arraydyn"
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
