package arraydyn

import "core:math"
import "core:slice"

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
	// Get ranks of input arrays to handle arrays of different dimensions.
	// This allows broadcasting between e.g. 2D and 3D arrays.
	rank_a := len(shape_a)
	rank_b := len(shape_b)
	max_rank := max(rank_a, rank_b)

	// Allocate output arrays with size of largest rank. We need these to be
	// the same size to properly align dimensions for broadcasting.
	shape = make([]uint, max_rank)
	strides_a = make([]uint, max_rank)
	strides_b = make([]uint, max_rank)

	// Process dimensions right-to-left to match NumPy broadcasting rules.
	// This allows smaller arrays to be broadcast against larger ones by
	// aligning their rightmost dimensions.
	for i in 0 ..< max_rank {
		// Default to dim=1, stride=0 for missing dimensions to enable broadcasting.
		// This lets us transparently handle arrays of different ranks.
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

		// Output shape takes the larger dimension, allowing broadcasting of
		// size-1 dimensions against larger ones.
		shape[max_rank - 1 - i] = max(dim_a, dim_b)

		// Set stride to 0 for broadcasted (size-1) dimensions to replicate values.
		// Keep original stride otherwise to preserve data layout.
		if dim_a == 1 {
			strides_a[max_rank - 1 - i] = 0 // Broadcasting - replicate the single value
		} else {
			strides_a[max_rank - 1 - i] = stride_a // Normal indexing
		}

		if dim_b == 1 {
			strides_b[max_rank - 1 - i] = 0 // Broadcasting - replicate the single value
		} else {
			strides_b[max_rank - 1 - i] = stride_b // Normal indexing
		}
	}

	return shape, strides_a, strides_b
}

// Element-wise function operation between arrays while handling broadcasting.
// Broadcasting allows operating on arrays of different shapes by replicating
// values across dimensions following NumPy-style rules.
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

	// TODO(Aria): eliminate necessity to clone data.
	// Thing we can consider: check if a/b requires broadcasting. If it is,
	// then just take reference to its data without `defer delete`
	arr_a := _get_strided_data(a, broadcast_shape, strides_a)
	defer delete(arr_a)
	arr_b := _get_strided_data(b, broadcast_shape, strides_b)
	defer delete(arr_b)

	for i in 0 ..< size {
		res.data[i] = fn(arr_a[i], arr_b[i])
	}

	// Handle dependencies
	res.requires_grad = a.requires_grad || b.requires_grad
	append(&res.deps, a, b)

	return res
}

_array_unary_op :: #force_inline proc(
	arr: ^Array_Dyn($T),
	fn: proc(_: T) -> T,
) -> (
	res: ^Array_Dyn(T),
) {
	res = _array_alloc(T, arr.shape)
	size := _shape_to_size(arr.shape)

	arr_data := _get_strided_data(arr)
	defer delete(arr_data)

	for i in 0 ..< size {
		res.data[i] = fn(arr_data[i])
	}
	return res
}

/******************************************************************************
 Basic arithmetic operations
 *****************************************************************************/

// Binary operators

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

// Unary operators

exp :: proc(arr: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_unary_op(arr, #force_inline proc(x: T) -> T {return math.exp(x)})
}
