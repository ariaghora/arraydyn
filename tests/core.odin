package tests

import ar "../arraydyn"
import "core:testing"

@(test)
test_init :: proc(t: ^testing.T) {
	empty := ar._array_alloc(i32, {2, 3})
	defer ar.array_free(empty)
	m_ones := ar.ones(f16, {2, 3})
	defer ar.array_free(m_ones)
}

import "core:fmt"
@(test)
test_strided_data_extract :: proc(t: ^testing.T) {
	arr := ar.new_with_init(i32, []i32{1, 2, 3, 4}, {2, 2})
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
