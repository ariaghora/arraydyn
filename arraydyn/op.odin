package arraydyn


array_get :: proc {
	_array_get_2d,
	_array_get_3d,
	_array_get_4d,
	_array_get_nd,
}

_array_get_1d :: #force_inline proc(arr: ^Array_Dyn($T), i0: uint) -> T {
	s0 := arr.strides[0]
	return arr.data[i0 * s0]
}

_array_get_2d :: #force_inline proc(arr: ^Array_Dyn($T), row, col: uint) -> T {
	s0, s1 := arr.strides[0], arr.strides[1]
	return arr.data[row * s0 + col * s1]
}

_array_get_3d :: #force_inline proc(arr: ^Array_Dyn($T), i0, i1, i2: uint) -> T {
	s0, s1, s2 := arr.strides[0], arr.strides[1], arr.strides[2]
	return arr.data[i0 * s0 + i1 * s1 + i2 * s2]
}

_array_get_4d :: #force_inline proc(arr: ^Array_Dyn($T), i0, i1, i2, i3: uint) -> T {
	s0, s1, s2, s3 := arr.strides[0], arr.strides[1], arr.strides[2], arr.strides[3]
	return arr.data[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]
}

_array_get_nd :: #force_inline proc(arr: ^Array_Dyn($T), coord: []uint) -> T {
	index: uint = 0
	for i := 0; i < len(coord); i += 1 {
		index += coord[i] * arr.strides[i]
	}
	return arr.data[index]
}


_compute_strided_index :: #force_inline proc(arr: ^Array_Dyn($T), idx: uint) -> uint {
	switch len(arr.shape) {
	case 1:
		return idx * arr.strides[0]
	case 2:
		i1 := idx % arr.shape[1]
		i0 := idx / arr.shape[1]
		return i0 * arr.strides[0] + i1 * arr.strides[1]
	case 3:
		i2 := idx % arr.shape[2]
		tmp := idx / arr.shape[2]
		i1 := tmp % arr.shape[1]
		i0 := tmp / arr.shape[1]
		return i0 * arr.strides[0] + i1 * arr.strides[1] + i2 * arr.strides[2]
	case:
		// N-dimensional case
		coord := make([]uint, len(arr.shape))
		defer delete(coord)

		remaining := idx
		for i := len(arr.shape) - 1; i > 0; i -= 1 {
			coord[i] = remaining % arr.shape[i]
			remaining /= arr.shape[i]
		}
		coord[0] = remaining

		offset: uint = 0
		for i := 0; i < len(coord); i += 1 {
			offset += coord[i] * arr.strides[i]
		}
		return offset
	}
}

_array_binary_op :: #force_inline proc(
	a, b: ^Array_Dyn($T),
	fn: proc(_: T, _: T) -> T,
) -> (
	res: ^Array_Dyn(T),
) {
	res = _array_alloc(T, a.shape)
	size := _shape_to_size(a.shape)

	switch {
	case a.contiguous && b.contiguous:
		for i := uint(0); i < size; i += 1 {
			res.data[i] = fn(a.data[i], b.data[i])
		}
	case a.contiguous:
		for i := uint(0); i < size; i += 1 {
			idx_b := _compute_strided_index(b, i)
			res.data[i] = fn(a.data[i], b.data[idx_b])
		}
	case b.contiguous:
		for i := uint(0); i < size; i += 1 {
			idx_a := _compute_strided_index(a, i)
			res.data[i] = fn(a.data[idx_a], b.data[i])
		}
	case:
		for i := uint(0); i < size; i += 1 {
			idx_a := _compute_strided_index(a, i)
			idx_b := _compute_strided_index(b, i)
			res.data[i] = fn(a.data[idx_a], b.data[idx_b])
		}
	}
	return res
}

add :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x + y})
}
sub :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x - y})
}
