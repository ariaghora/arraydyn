package tests

import ar "../arraydyn"
import "core:slice"
import "core:testing"

@(test)
test_init :: proc(t: ^testing.T) {
	empty := ar._array_alloc(i32, {2, 3})
	defer ar.array_free(empty)
	m_ones := ar.ones(f16, {2, 3})
	defer ar.array_free(m_ones)
}

@(test)
test_strided_data_extract :: proc(t: ^testing.T) {
	arr := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
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
test_require_grads :: proc(t: ^testing.T) {
	a := ar.new_with_init([]i32{1, 2}, {2})
	ar.set_requires_grad(a, true)
	b := ar.new_with_init([]i32{1, 2}, {2})
	c := ar.add(a, b)

	testing.expect(t, c.requires_grad)
	testing.expect(t, a.requires_grad)
	testing.expect(t, slice.equal(c.grad.shape, c.shape))

	c_val := ar._get_strided_data(c)
	defer delete(c_val)
	testing.expect(t, slice.equal(c_val, []i32{2, 4}))

	// disable gradients
	ar.set_requires_grad(a, false)
	testing.expect(t, !a.requires_grad)

	ar.array_free(a)
	ar.array_free(b)
	ar.array_free(c)
}

@(test)
test_require_grads_self :: proc(t: ^testing.T) {
	a := ar.new_with_init([]i32{1, 2}, {2})
	ar.set_requires_grad(a, true)
	b := ar.add(a, a)

	ar.array_free(a)
	ar.array_free(b)
}
