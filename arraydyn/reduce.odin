package arraydyn

import "base:builtin"
import "core:math"

@(private = "package")
reduction_shape :: proc(shape: []uint, axis: int, keepdims: bool) -> []uint {
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
		for i in 0 ..< len(shape) {
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
@(private = "package")
reduce_along_axis :: proc(
	arr: ^Array_Dyn($T),
	reducer: proc(_: T, _: T) -> T, // Function that combines two elements into one
	initial: T, // Starting value for reduction (e.g., 0 for sum, 1 for product)
	axis: int, // Which dimension to reduce along
	keepdims := false, // Whether to preserve reduced dimension as size 1
) -> ^Array_Dyn(T) {
	// Get shape for result array, handling keepdims case.
	// Deferred delete needed since _array_alloc copies it
	out_shape := reduction_shape(arr.shape, axis, keepdims)
	defer delete(out_shape)
	result := _array_alloc(T, out_shape)

	// Initialize all elements to initial value since reducer will use these as
	// starting points
	for _, i in result.data {
		result.data[i] = initial
	}

	// Calculate the total size of all dimensions before the reduction axis.
	// For example, if we have array shape [2,3,4] and axis=1:
	// - Dimensions before axis are [2], so outer_size = 2
	// - This determines how many complete "slices" we need to process
	outer_size: uint = 1
	for i in 0 ..< axis {
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
	for i in (axis + 1) ..< len(arr.shape) {
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
		for outer in 0 ..< outer_size {
			for inner in 0 ..< inner_size {
				// Output index maps to reduced shape by skipping reduction axis.
				// Each out_idx accumulates values from axis_size elements.
				out_idx := outer * inner_size + inner
				for a in 0 ..< axis_size {
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
		for outer in 0 ..< outer_size {
			for inner in 0 ..< inner_size {
				out_idx := outer * inner_size + inner
				for a in 0 ..< axis_size {
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
	sum_axis_a,
	sum_axis_t,
}

sum_a :: proc(arr: ^Array_Dyn($T), keepdims := false) -> ^Array_Dyn(T) {
	out_shape := keepdims ? make([]uint, len(arr.shape)) : []uint{1}
	defer if keepdims do delete(out_shape)

	// Create result array with appropriate shape
	result := _array_alloc(T, out_shape)

	// Sum all elements directly
	sum: T = 0
	if arr.contiguous {
		// Fast path for contiguous data
		for val in arr.data {
			sum += val
		}
	} else {
		// Handle non-contiguous case
		size := _shape_to_size(arr.shape)
		for i: uint = 0; i < size; i += 1 {
			idx := _compute_strided_index(arr.shape, arr.strides, i)
			sum += arr.data[idx]
		}
	}

	result.data[0] = sum
	return result
}

sum_t :: proc(t: ^Tensor($T), keepdims := false) -> ^Tensor(T) {
	keepdims_tensor := new_with_init([]T{T(keepdims ? 1 : 0)}, {1})
	defer tensor_release(keepdims_tensor)

	return autograd_make_op(
		deps = []^Tensor(T){t, keepdims_tensor},
		new_arrdata = sum_a(t.arrdata, keepdims),
		backward_fn = proc(tensor: ^Tensor(T), grad_output: ^Array_Dyn(T)) {
			input_tensor := tensor.deps[0]
			keepdims := tensor.deps[1].data[0] > 0

			// grad is necessarily a 0-tensor, so each input element
			// contribute that much
			ones_arr := _ones(T, input_tensor.shape)
			defer array_free(ones_arr)
			local_grad := mul(ones_arr, grad_output)
			defer array_free(local_grad)

			// Add to existing gradient
			old_grad := input_tensor.grad
			input_tensor.grad = add(old_grad, local_grad)
			array_free(old_grad)
		},
		backward_fn_name = "sum_backward",
	)
}

sum_axis_a :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return reduce_along_axis(arr, proc(x, y: T) -> T {return x + y}, T(0), axis, keepdims)
}

sum_axis_t :: proc(t: ^Tensor($T), axis: int, keepdims := false) -> ^Tensor(T) {
	// Create tensors to store axis and keepdims values
	axis_tensor := new_with_init([]T{T(axis)}, {1})
	keepdims_tensor := new_with_init([]T{T(keepdims ? 1 : 0)}, {1})
	defer tensor_release(axis_tensor, keepdims_tensor)

	return autograd_make_op(
		deps = []^Tensor(T){t, axis_tensor, keepdims_tensor},
		new_arrdata = sum_axis_a(t.arrdata, axis, keepdims),
		backward_fn = proc(tensor: ^Tensor(T), grad_output: ^Array_Dyn(T)) {
			input_tensor := tensor.deps[0]
			// Extract cached values from deps
			axis := int(tensor.deps[1].data[0])
			keepdims := tensor.deps[2].data[0] > 0

			// Create output shape for the gradient based on keepdims
			out_shape := reduction_shape(input_tensor.shape, axis, keepdims)
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


max :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return reduce_along_axis(
		arr,
		proc(x, y: T) -> T {return builtin.max(x, y)},
		builtin.min(T),
		axis,
		keepdims,
	)
}

min :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	return reduce_along_axis(
		arr,
		proc(x, y: T) -> T {return builtin.min(x, y)},
		builtin.max(T),
		axis,
		keepdims,
	)
}

mean :: proc {
	mean_a,
	mean_t,
	mean_axis_a,
	mean_axis_t,
}

mean_a :: proc(arr: ^Array_Dyn($T), keepdims := false) -> ^Array_Dyn(T) {
	out_shape := keepdims ? make([]uint, len(arr.shape)) : []uint{1}
	defer if keepdims do delete(out_shape)

	// Create result array with appropriate shape
	result := _array_alloc(T, out_shape)

	// Calculate mean of all elements
	sum: T = 0
	if arr.contiguous {
		// Fast path for contiguous data
		for val in arr.data {
			sum += val
		}
	} else {
		// Handle non-contiguous case
		size := _shape_to_size(arr.shape)
		for i: uint = 0; i < size; i += 1 {
			idx := _compute_strided_index(arr.shape, arr.strides, i)
			sum += arr.data[idx]
		}
	}

	n := T(data_len(arr))
	result.data[0] = sum / n
	return result
}

mean_t :: proc(t: ^Tensor($T), keepdims := false) -> ^Tensor(T) {
	keepdims_tensor := new_with_init([]T{T(keepdims ? 1 : 0)}, {1})
	defer tensor_release(keepdims_tensor)

	return autograd_make_op(
		deps = []^Tensor(T){t, keepdims_tensor},
		new_arrdata = mean_a(t.arrdata, keepdims),
		backward_fn = proc(tensor: ^Tensor(T), grad_output: ^Array_Dyn(T)) {
			input_tensor := tensor.deps[0]
			if !input_tensor.requires_grad {return}

			keepdims := tensor.deps[1].data[0] > 0
			size := uint(1)
			for s in input_tensor.shape {
				size *= s
			}
			n := T(size)

			// Scale gradient by 1/n since d(mean)/dx = 1/n for each input x
			scaled_grad := clone(grad_output)
			for i := 0; i < len(scaled_grad.data); i += 1 {
				scaled_grad.data[i] /= n
			}

			// Expand gradient back to input shape
			expanded_grad := broadcast_grad_to_shape(scaled_grad, input_tensor.shape)
			array_free(scaled_grad)

			// Add to existing gradient
			old_grad := input_tensor.grad
			input_tensor.grad = add(old_grad, expanded_grad)
			array_free(old_grad, expanded_grad)
		},
		backward_fn_name = "mean_backward",
	)
}
mean_axis_a :: proc(arr: ^Array_Dyn($T), axis: int, keepdims := false) -> ^Array_Dyn(T) {
	s := sum_axis_a(arr, axis, keepdims)
	n := T(arr.shape[axis])
	for i in 0 ..< len(s.data) {
		s.data[i] /= n
	}
	return s
}

mean_axis_t :: proc(t: ^Tensor($T), axis: int, keepdims := false) -> ^Tensor(T) {
	// Create tensors to store axis and keepdims values
	axis_tensor := new_with_init([]T{T(axis)}, {1})
	keepdims_tensor := new_with_init([]T{T(keepdims ? 1 : 0)}, {1})
	defer tensor_release(axis_tensor, keepdims_tensor)

	return autograd_make_op(
		deps = []^Tensor(T){t, axis_tensor, keepdims_tensor},
		new_arrdata = mean_axis_a(t.arrdata, axis, keepdims),
		backward_fn = proc(tensor: ^Tensor(T), grad_output: ^Array_Dyn(T)) {
			input_tensor := tensor.deps[0]
			if !input_tensor.requires_grad {return}

			axis := int(tensor.deps[1].data[0])
			keepdims := tensor.deps[2].data[0] > 0
			n := T(input_tensor.shape[axis])

			// Scale gradient by 1/n since d(mean)/dx = 1/n for each input x
			scaled_grad := clone(grad_output)
			for i in 0 ..< len(scaled_grad.data) {
				scaled_grad.data[i] /= n
			}

			// Expand gradient back to input shape
			expanded_grad := broadcast_grad_to_shape(scaled_grad, input_tensor.shape)
			array_free(scaled_grad)

			// Add to existing gradient
			old_grad := input_tensor.grad
			input_tensor.grad = add(old_grad, expanded_grad)
			array_free(old_grad, expanded_grad)
		},
		backward_fn_name = "mean_backward",
	)
}

argmax :: proc {
	argmax_axis_a,
	argmax_axis_t,
}

// Returns the indices of maximum values along a specified axis
argmax_axis_a :: proc(arr: ^Array_Dyn($T), axis: int, keepdims: bool = false) -> ^Array_Dyn(T) {
	if axis < 0 || axis >= len(arr.shape) {
		panic("Axis out of bounds")
	}

	// Get shape for result array, handling keepdims case
	out_shape := reduction_shape(arr.shape, axis, keepdims)
	defer delete(out_shape)
	result := _array_alloc(T, out_shape)

	// Calculate sizes for iteration
	outer_size: uint = 1
	for i in 0 ..< axis {
		outer_size *= arr.shape[i]
	}

	inner_size: uint = 1
	for i in (axis + 1) ..< len(arr.shape) {
		inner_size *= arr.shape[i]
	}

	axis_size := arr.shape[axis]

	// For each position, find index of maximum value
	if arr.contiguous {
		// Fast path for contiguous arrays
		for outer in 0 ..< outer_size {
			for inner in 0 ..< inner_size {
				out_idx := outer * inner_size + inner
				max_val := builtin.min(T)
				max_idx := T(0)

				// Find maximum value and its index along the axis
				for a in 0 ..< axis_size {
					idx := outer * (axis_size * inner_size) + a * inner_size + inner
					val := arr.data[idx]
					if val > max_val {
						max_val = val
						max_idx = T(a)
					}
				}
				result.data[out_idx] = max_idx
			}
		}
	} else {
		// Handle non-contiguous arrays
		for outer in 0 ..< outer_size {
			for inner in 0 ..< inner_size {
				out_idx := outer * inner_size + inner
				max_val := builtin.min(T)
				max_idx := T(0)

				// Find maximum value and its index along the axis
				for a in 0 ..< axis_size {
					flat_idx := outer * (axis_size * inner_size) + a * inner_size + inner
					src_idx := _compute_strided_index(arr.shape, arr.strides, flat_idx)
					val := arr.data[src_idx]
					if val > max_val {
						max_val = val
						max_idx = T(a)
					}
				}
				result.data[out_idx] = max_idx
			}
		}
	}

	return result
}

argmax_axis_t :: proc(t: ^Tensor($T), axis: int, keepdims: bool = false) -> ^Tensor(T) {
	// Create tensors to store axis and keepdims values
	axis_tensor := new_with_init([]T{T(axis)}, {1})
	keepdims_tensor := new_with_init([]T{T(keepdims ? 1 : 0)}, {1})
	defer tensor_release(axis_tensor, keepdims_tensor)

	return autograd_make_op(
		deps = []^Tensor(T){t, axis_tensor, keepdims_tensor},
		new_arrdata = argmax_axis_a(t.arrdata, axis, keepdims),
		backward_fn = nil,
		backward_fn_name = "argmax_backward",
	)
}
