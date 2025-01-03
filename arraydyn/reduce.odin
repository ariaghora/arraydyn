package arraydyn

import "core:math"

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

// Perform a reduction operation along a specific axis. The reducer function is applied
// pairwise to all elements along the given axis, with the specified initial value as
// starting point. The operation preserves the array dimensions if keepdims is true.
_reduce :: proc(
	arr: ^Array_Dyn($T),
	reducer: proc(_: T, _: T) -> T, // Function that combines two elements into one
	initial: T, // Starting value for reduction (e.g., 0 for sum, 1 for product)
	axis: int, // Which dimension to reduce along
	keepdims := false, // Whether to preserve reduced dimension as size 1
) -> ^Array_Dyn(T) {
	// Get shape for result array, handling keepdims case.
	// Deferred delete needed since _array_alloc copies it
	out_shape := _reduction_shape(arr.shape, axis, keepdims)
	defer delete(out_shape)
	result := _array_alloc(T, out_shape)

	// Initialize all elements to initial value since reducer will use these as
	// starting points
	for i := 0; i < len(result.data); i += 1 {
		result.data[i] = initial
	}

	// Calculate the total size of all dimensions before the reduction axis.
	// For example, if we have array shape [2,3,4] and axis=1:
	// - Dimensions before axis are [2], so outer_size = 2
	// - This determines how many complete "slices" we need to process
	outer_size: uint = 1
	for i := 0; i < axis; i += 1 {
		outer_size *= arr.shape[i]
	}

	// Calculate the total size of all dimensions after the reduction axis.
	// For example, if we have array shape [2,3,4] and axis=1:
	// - Dimensions after axis are [4], so inner_size = 4
	// - This determines how many elements we need to process within each "slice"
	// - With outer_size=2 and axis_size=3, we process:
	//   * For outer=0: process 4 elements across each of the 3 positions in axis
	//   * For outer=1: repeat for the second slice
	inner_size: uint = 1
	for i := axis + 1; i < len(arr.shape); i += 1 {
		inner_size *= arr.shape[i]
	}

	// Size of dimension being reduced - we'll iterate through this in innermost loop
	axis_size := arr.shape[axis]

	// Split into two paths for performance: contiguous arrays can use direct indexing,
	// while non-contiguous need stride calculations. This optimization matters because
	// most arrays in practice are contiguous.
	if arr.contiguous {
		// Fast path: direct indexing for contiguous memory layout. The nested loops
		// process: outer dims -> inner dims -> reduction axis.
		for outer := uint(0); outer < outer_size; outer += 1 {
			for inner := uint(0); inner < inner_size; inner += 1 {
				// Output index maps to reduced shape by skipping reduction axis.
				// Each out_idx accumulates values from axis_size elements.
				out_idx := outer * inner_size + inner
				for a := uint(0); a < axis_size; a += 1 {
					// Input index formula expands position to full dimensionality.
					// The term (a * inner_size) strides through reduction axis while
					// keeping inner/outer coords fixed.
					idx := outer * (axis_size * inner_size) + a * inner_size + inner
					result.data[out_idx] = reducer(result.data[out_idx], arr.data[idx])
				}
			}
		}
	} else {
		// Slow path: handle non-contiguous arrays (e.g., views, transposed).
		// Same loop structure but must compute true memory position using strides.
		// This is slower but necessary for correctness with arbitrary layouts.
		for outer := uint(0); outer < outer_size; outer += 1 {
			for inner := uint(0); inner < inner_size; inner += 1 {
				out_idx := outer * inner_size + inner
				for a := uint(0); a < axis_size; a += 1 {
					// First compute logical flat index, then map to actual memory
					// position using strides. This two-step process handles any
					// possible memory layout.
					flat_idx := outer * (axis_size * inner_size) + a * inner_size + inner
					src_idx := _compute_strided_index(arr.shape, arr.strides, flat_idx)
					result.data[out_idx] = reducer(result.data[out_idx], arr.data[src_idx])
				}
			}
		}
	}

	return result
}

sum :: proc {
	sum_a,
	sum_t,
}

sum_a :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return _reduce(arr, proc(x, y: T) -> T {return x + y}, T(0), axis, keepdims)
}

sum_t :: proc(t: ^Tensor($T), axis: int, keepdims := false) -> ^Tensor(T) {
	// Create tensors to store axis and keepdims values
	axis_tensor := new_with_init([]T{T(axis)}, {1})
	keepdims_tensor := new_with_init([]T{T(keepdims ? 1 : 0)}, {1})
	// Defer tensor release because these tensors are added to deps in autograd_make_op
	// which increments their ref_count. We need to release our local reference to them
	// after autograd_make_op returns but before this function returns.
	defer tensor_release(axis_tensor, keepdims_tensor)

	return autograd_make_op(
		deps = []^Tensor(T){t, axis_tensor, keepdims_tensor},
		new_arrdata = sum_a(t.arrdata, axis, keepdims),
		backward_fn = proc(tensor: ^Tensor(T), grad_output: ^Array_Dyn(T)) {
			input_tensor := tensor.deps[0]
			// Extract cached values from deps
			axis := int(tensor.deps[1].data[0])
			keepdims := tensor.deps[2].data[0] > 0

			// Create output shape for the gradient based on keepdims
			out_shape := _reduction_shape(input_tensor.shape, axis, keepdims)
			defer delete(out_shape)

			// Expand gradient back to input shape
			expanded_grad := broadcast_grad_to_shape(grad_output, input_tensor.shape)

			// Add to existing gradient
			old_grad := input_tensor.grad
			input_tensor.grad = add(old_grad, expanded_grad)
			array_free(old_grad, expanded_grad)
		},
		backward_fn_name = "sum_backward",
	)
}


// NOTE(Aria): We chose name `maximum` and `minimum` to avoid collision with Odin's builtin
maximum :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return _reduce(arr, proc(x, y: T) -> T {return max(x, y)}, T(0), axis, keepdims)
}

minimum :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return _reduce(arr, proc(x, y: T) -> T {return min(x, y)}, T(0), axis, keepdims)
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
