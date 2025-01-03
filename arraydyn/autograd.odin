package arraydyn

import "core:slice"


autograd_make_op :: proc(
	deps: []^Tensor($T),
	new_arrdata: ^Array_Dyn(T),
	backward_fn: proc(tensor: ^Tensor(T), upstream_grad: ^Array_Dyn(T)),
	backward_fn_name: string,
) -> ^Tensor(T) {
	res := _tensor_from_array(new_arrdata)

	// Track if gradients are needed by checking if any dependency requires gradients
	// This propagates gradient requirements up through the computational graph
	requres_grad := false
	for dep in deps {
		// Increment reference count to prevent premature cleanup since this tensor
		// now depends on the dependency tensor
		dep.ref_count += 1
		// Use OR to propagate gradient requirement - if any dependency needs gradients,
		// this tensor will need them too for backprop
		requres_grad |= dep.requires_grad
	}

	// Store dependencies for backpropagation - we need these to recursively compute
	// gradients during backward pass
	append(&res.deps, ..deps)

	// Configure gradient tracking based on dependencies' requirements
	// This ensures we only allocate gradients where needed
	set_requires_grad(res, requres_grad)

	// Store the backward function and its name for gradient computation
	// The name is useful for debugging and visualization of the computation graph
	res.backward_fn = backward_fn
	res.backward_fn_name = backward_fn_name

	return res
}

backward :: proc {
	backward_with_grad,
	backward_no_grad,
}

backward_with_grad :: proc(t: ^Tensor($T), grad: ^Array_Dyn(T)) {
	// If there's no backward function or no dependencies, nothing more to do
	if t.backward_fn == nil || len(t.deps) == 0 {
		return
	}

	// Call backward function to compute gradients for dependencies
	t.backward_fn(t, grad)

	// Recursively propagate gradients through dependencies
	for dep in t.deps {
		if dep.requires_grad {
			backward_with_grad(dep, grad)
		}
	}
}

backward_no_grad :: proc(t: ^Tensor($T)) {
	grad := _ones(T, t.shape)

	// Set initial gradient
	if t.requires_grad {
		array_free(t.grad)
		t.grad = grad
	}

	// If there's no backward function or no dependencies, nothing more to do
	if t.backward_fn == nil || len(t.deps) == 0 {
		return
	}

	// Call backward function to compute gradients for dependencies
	t.backward_fn(t, grad)

	// Recursively propagate gradients through dependencies
	for dep in t.deps {
		if dep.requires_grad {
			backward_with_grad(dep, grad)
		}
	}
}


// broadcast_grad_to_shape handles gradient computation for broadcasted operations by:
// 1. Reducing summed dimensions where broadcasting occurred
// 2. Expanding the gradient back to match the target shape
broadcast_grad_to_shape :: proc(grad_arr: ^Array_Dyn($T), target_shape: []uint) -> ^Array_Dyn(T) {
	// Find max rank needed because broadcasting may have increased dimensions
	// from either input, and we need to handle all cases
	grad_shape := grad_arr.shape
	max_dims := max(len(grad_shape), len(target_shape))

	// Left-pad gradient shape with ones to match max rank. This is necessary
	// because numpy-style broadcasting aligns arrays from the right, treating
	// missing left dimensions as size 1
	grad_shape_padded := make([dynamic]uint, max_dims - len(grad_shape))
	defer delete(grad_shape_padded)
	slice.fill(grad_shape_padded[:], 1)
	append(&grad_shape_padded, ..grad_shape)

	// Similarly pad target shape from left to match ranks. This creates
	// a common frame of reference to compare corresponding dimensions
	target_shape_padded := make([dynamic]uint, max_dims - len(target_shape))
	defer delete(target_shape_padded)
	slice.fill(target_shape_padded[:], 1)
	append(&target_shape_padded, ..target_shape)

	// Find dimensions that were broadcast during forward pass. We need these
	// because gradients must be summed across broadcast dimensions to preserve
	// the chain rule when backpropagating through broadcast operations
	sum_axes := make([dynamic]int)
	defer delete(sum_axes)
	for i in 0 ..< max_dims {
		if grad_shape_padded[i] != target_shape_padded[i] && target_shape_padded[i] == 1 {
			append(&sum_axes, i)
		}
	}

	// Clone gradient since we'll modify it. We can't modify the input
	// directly as it may be used elsewhere in the computation graph
	grad := clone(grad_arr)

	// Sum across broadcast dimensions to collapse expanded axes. This is required
	// because broadcasting artificially expanded some dimensions in forward pass,
	// so we need to aggregate gradients across those dimensions in backward pass
	if len(sum_axes) > 0 {
		result := grad
		for axis in sum_axes {
			old_result := result
			// keepdims preserves axis for later broadcasting
			result = sum(old_result, axis, keepdims = true)
			array_free(old_result)
		}
		grad = result
	}

	// Allocate output array matching target shape. We need a new array because
	// the gradient shape after summing may still not match the target
	result := _array_alloc(T, target_shape)

	// Broadcast gradient values to fill target shape. We use modulo to handle
	// the repeating pattern created by broadcasting - this effectively reverses
	// the expansion that happened in the forward pass
	size := _shape_to_size(target_shape)
	grad_size := _shape_to_size(grad.shape)
	for i: uint = 0; i < size; i += 1 {
		// TODO(Aria): This will be slow
		idx := _compute_strided_index(grad.shape, grad.strides, i % grad_size)
		result.data[i] = grad.data[idx]
	}
	array_free(grad)

	return result
}
