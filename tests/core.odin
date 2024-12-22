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

@(test)
test_transpose :: proc(t: ^testing.T) {
	arr := ar.new_with_init(f32, []f32{1, 2, 3, 4, 5, 6}, []uint{2, 3})
	defer ar.array_free(arr)

	transposed := ar.tranpose(arr)
	defer ar.array_free(transposed)

	expected := []f32{1, 4, 2, 5, 3, 6}

	testing.expect(t, len(transposed.data) == len(expected))
	for item, i in soa_zip(x = transposed.data, y = expected) {
		testing.expect_value(t, item.x, item.y)
	}

}
