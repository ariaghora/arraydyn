package arraydyn

_reduction_shape :: proc(shape: []uint, axis: int, keepdims: bool) -> []uint {
	if axis < 0 || axis >= len(shape) {
		panic("Axis out of bounds")
	}

	if keepdims {
		// Create new shape with axis dimension set to 1
		new_shape := make([]uint, len(shape))
		copy(new_shape, shape)
		new_shape[axis] = 1
		return new_shape
	} else {
		// Original behavior: remove the axis
		new_shape := make([]uint, len(shape) - 1)
		j := 0
		for i := 0; i < len(shape); i += 1 {
			if i != axis {
				new_shape[j] = shape[i]
				j += 1
			}
		}
		return new_shape
	}
}

// Modified reduction function with keepdims parameter
_reduce :: proc(
	arr: ^Array_Dyn($T),
	reducer: proc(_: T, _: T) -> T,
	initial: T,
	axis: int,
	keepdims := false,
) -> ^Array_Dyn(T) {
	out_shape := _reduction_shape(arr.shape, axis, keepdims)
	defer delete(out_shape)
	result := _array_alloc(T, out_shape)

	// Initialize result
	for i := 0; i < len(result.data); i += 1 {
		result.data[i] = initial
	}

	// Rest of the reduction logic remains the same
	outer_size: uint = 1
	for i := 0; i < axis; i += 1 {
		outer_size *= arr.shape[i]
	}

	inner_size: uint = 1
	for i := axis + 1; i < len(arr.shape); i += 1 {
		inner_size *= arr.shape[i]
	}

	axis_size := arr.shape[axis]

	// Perform reduction
	for outer := uint(0); outer < outer_size; outer += 1 {
		for inner := uint(0); inner < inner_size; inner += 1 {
			out_idx := outer * inner_size + inner
			for a := uint(0); a < axis_size; a += 1 {
				idx := outer * (axis_size * inner_size) + a * inner_size + inner
				result.data[out_idx] = reducer(result.data[out_idx], arr.data[idx])
			}
		}
	}

	return result
}

// Updated convenience functions
sum :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return _reduce(arr, proc(x, y: T) -> T {return x + y}, T(0), axis, keepdims)
}

mean :: proc(
	arr: ^Array_Dyn($T),
	axis: int,
	keepdims := false,
) -> ^Array_Dyn(T) where intrinsics.type_is_float(T) {
	s := sum(arr, axis, keepdims)
	n := T(arr.shape[axis])
	for i := 0; i < len(s.data); i += 1 {
		s.data[i] /= n
	}
	return s
}
