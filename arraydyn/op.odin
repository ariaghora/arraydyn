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
	for i in 0 ..< len(coord) {
		index += coord[i] * arr.strides[i]
	}
	return arr.data[index]
}


_compute_strided_index :: #force_inline proc(shape, strides: []uint, idx: uint) -> uint {
	switch len(shape) {
	case 1:
		return idx * strides[0]
	case 2:
		i1 := idx % shape[1]
		i0 := idx / shape[1]
		return i0 * strides[0] + i1 * strides[1]
	case 3:
		i2 := idx % shape[2]
		tmp := idx / shape[2]
		i1 := tmp % shape[1]
		i0 := tmp / shape[1]
		return i0 * strides[0] + i1 * strides[1] + i2 * strides[2]
	case:
		// N-dimensional case
		coord := make([]uint, len(shape))
		defer delete(coord)

		remaining := idx
		for i := len(shape) - 1; i > 0; i -= 1 {
			coord[i] = remaining % shape[i]
			remaining /= shape[i]
		}
		coord[0] = remaining

		offset: uint = 0
		for i in 0 ..< len(coord) {
			offset += coord[i] * strides[i]
		}
		return offset
	}
}

_is_broadcastable :: proc(shape_a, shape_b: []uint) -> bool {
	rank_a := len(shape_a)
	rank_b := len(shape_b)
	max_rank := max(rank_a, rank_b)

	for i in 0 ..< max_rank {
		a_idx := rank_a - 1 - i
		b_idx := rank_b - 1 - i

		dim_a := uint(1)
		if a_idx >= 0 {
			dim_a = shape_a[a_idx]
		}

		dim_b := uint(1)
		if b_idx >= 0 {
			dim_b = shape_b[b_idx]
		}

		if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
			return false
		}
	}
	return true
}

_get_broadcast_shape_and_strides :: proc(
	shape_a, shape_b, strides_a_in, strides_b_in: []uint,
) -> (
	shape: []uint,
	strides_a, strides_b: []uint,
) {
	rank_a := len(shape_a)
	rank_b := len(shape_b)
	max_rank := max(rank_a, rank_b)

	shape = make([]uint, max_rank)
	strides_a = make([]uint, max_rank)
	strides_b = make([]uint, max_rank)

	// Calculate broadcast dimensions and strides
	for i in 0 ..< max_rank {
		dim_a := uint(1)
		stride_a := uint(0)
		if i < rank_a {
			dim_a = shape_a[rank_a - 1 - i]
			stride_a = strides_a_in[rank_a - 1 - i]
		}

		dim_b := uint(1)
		stride_b := uint(0)
		if i < rank_b {
			dim_b = shape_b[rank_b - 1 - i]
			stride_b = strides_b_in[rank_b - 1 - i]
		}

		shape[max_rank - 1 - i] = max(dim_a, dim_b)

		// Keep stride if dimension matches, set to 0 if dimension is being broadcast
		if dim_a == 1 {
			strides_a[max_rank - 1 - i] = 0
		} else {
			strides_a[max_rank - 1 - i] = stride_a
		}

		if dim_b == 1 {
			strides_b[max_rank - 1 - i] = 0
		} else {
			strides_b[max_rank - 1 - i] = stride_b
		}
	}

	return shape, strides_a, strides_b
}

_array_binary_op :: #force_inline proc(
	a, b: ^Array_Dyn($T),
	fn: proc(_: T, _: T) -> T,
) -> (
	res: ^Array_Dyn(T),
) {
	if !_is_broadcastable(a.shape, b.shape) {
		panic("Arrays cannot be broadcast together")
	}
	broadcast_shape, strides_a, strides_b := _get_broadcast_shape_and_strides(
		a.shape,
		b.shape,
		a.strides,
		b.strides,
	)
	defer delete(broadcast_shape)
	defer delete(strides_a)
	defer delete(strides_b)

	res = _array_alloc(T, broadcast_shape)
	size := _shape_to_size(broadcast_shape)

	// TODO(Aria): eliminate necessity to clone data
	arr_a := _get_strided_data(a, broadcast_shape, strides_a)
	defer delete(arr_a)
	arr_b := _get_strided_data(b, broadcast_shape, strides_b)
	defer delete(arr_b)

	for i in 0 ..< size {
		res.data[i] = fn(arr_a[i], arr_b[i])
	}
	return res
}

add :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x + y})
}

sub :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x - y})
}

mul :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x * y})
}

div :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x / y})
}
