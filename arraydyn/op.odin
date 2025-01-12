package arraydyn

import "base:builtin"
import "core:fmt"
import "core:math"
import "core:slice"

_is_broadcastable :: proc(shape_a, shape_b: []uint) -> bool {
	rank_a := len(shape_a)
	rank_b := len(shape_b)
	max_rank := builtin.max(rank_a, rank_b)

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
	max_rank := builtin.max(rank_a, rank_b)

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
		shape[max_rank - 1 - i] = builtin.max(dim_a, dim_b)

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

add :: proc {
	add_a,
	add_t,
}

add_a :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x + y})
}

add_t :: proc(a, b: ^Tensor($T)) -> ^Tensor(T) {
	res := autograd_make_op(
		[]^Tensor(T){a, b},
		new_arrdata = add_a(a.arrdata, b.arrdata),
		backward_fn_name = "add_backward",
		backward_fn = proc(tensor: ^Tensor(T), upstream_grad: ^Array_Dyn(T)) {
			a, b := tensor.deps[0], tensor.deps[1]

			// Propagate gradient to a if needed
			if a.requires_grad {
				old_grad := a.grad
				grad_a := broadcast_grad_to_shape(upstream_grad, a.arrdata.shape)
				a.grad = add(old_grad, grad_a)
				array_free(old_grad, grad_a)
			}

			// Propagate gradient to b if needed
			if b.requires_grad {
				old_grad := b.grad
				grad_b := broadcast_grad_to_shape(upstream_grad, b.arrdata.shape)
				b.grad = add(old_grad, grad_b)
				array_free(old_grad, grad_b)
			}
		},
	)

	return res
}


sub :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x - y})
}

mul :: proc {
	mul_a,
	mul_t,
}

mul_a :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x * y})
}

mul_t :: proc(a, b: ^Tensor($T)) -> ^Tensor(T) {
	res := autograd_make_op(
		[]^Tensor(T){a, b},
		new_arrdata = mul_a(a.arrdata, b.arrdata),
		backward_fn_name = "mul_backward",
		backward_fn = proc(tensor: ^Tensor(T), upstream_grad: ^Array_Dyn(T)) {
			a, b := tensor.deps[0], tensor.deps[1]

			// Propagate gradient to a if needed
			if a.requires_grad {
				old_grad := a.grad
				new_grad := b.arrdata
				local_grad := mul(new_grad, upstream_grad)
				grad_a := broadcast_grad_to_shape(local_grad, a.arrdata.shape)
				defer array_free(local_grad)
				defer array_free(grad_a)
				a.grad = add(old_grad, grad_a)
				array_free(old_grad)
			}

			// Propagate gradient to b if needed
			if b.requires_grad {
				old_grad := b.grad
				new_grad := a.arrdata
				local_grad := mul(new_grad, upstream_grad)
				grad_b := broadcast_grad_to_shape(local_grad, b.arrdata.shape)
				defer array_free(local_grad)
				defer array_free(grad_b)
				b.grad = add(old_grad, grad_b)
				array_free(old_grad)
			}
		},
	)

	return res
}
matmul :: proc {
	matmul_a,
	matmul_t,
}

transpose :: proc(a: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	if len(a.shape) != 2 {
		panic("can only transpose matrix")
	}

	m, n := a.shape[0], a.shape[1]
	result := _array_alloc(T, []uint{n, m})

	for i: uint = 0; i < m; i += 1 {
		for j: uint = 0; j < n; j += 1 {
			result.data[j * result.strides[0] + i * result.strides[1]] = array_get(a, i, j)
		}
	}

	return result
}

matmul_a :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	if len(a.shape) != 2 {
		panic("matmul is only for tensor with 2 dimensions (matrix)")
	}

	if !a.contiguous {
		panic("for now matmul is only for contiguous matrices")
	}

	m, k, n := a.shape[0], a.shape[1], b.shape[1]
	if b.shape[0] != k {
		fmt.panicf(
			"matmul shape mismatch: a = %v, b = %v. dim 1 of a must match dim 0 of b",
			a.shape,
			b.shape,
		)
	}
	// Result will be m x n
	result := _array_alloc(T, []uint{m, n})

	// For each element in result matrix
	for i: uint = 0; i < m; i += 1 {
		for j: uint = 0; j < n; j += 1 {
			sum: T = 0
			// Compute dot product of row i from a and column j from b
			for l: uint = 0; l < k; l += 1 {
				a_val := array_get(a, i, l)
				b_val := array_get(b, l, j)
				sum += a_val * b_val
			}
			// Store result at (i,j)
			result.data[i * result.strides[0] + j * result.strides[1]] = sum
		}
	}

	return result
}

matmul_t :: proc(a, b: ^Tensor($T)) -> ^Tensor(T) {
	res := autograd_make_op(
		[]^Tensor(T){a, b},
		new_arrdata = matmul_a(a.arrdata, b.arrdata),
		backward_fn_name = "matmul_backward",
		backward_fn = proc(tensor: ^Tensor(T), upstream_grad: ^Array_Dyn(T)) {
			a, b := tensor.deps[0], tensor.deps[1]

			// Propagate gradient to a if needed
			if a.requires_grad {
				old_grad := a.grad
				// dL/dA = dL/dC * B^T
				b_transpose := transpose(b.arrdata)
				local_grad := matmul_a(upstream_grad, b_transpose)
				defer array_free(local_grad)
				defer array_free(b_transpose)
				a.grad = add(old_grad, local_grad)
				array_free(old_grad)
			}

			// Propagate gradient to b if needed
			if b.requires_grad {
				old_grad := b.grad
				// dL/dB = A^T * dL/dC
				a_transpose := transpose(a.arrdata)
				local_grad := matmul_a(a_transpose, upstream_grad)
				defer array_free(local_grad)
				defer array_free(a_transpose)
				b.grad = add(old_grad, local_grad)
				array_free(old_grad)
			}
		},
	)

	return res
}


div :: proc(a, b: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_binary_op(a, b, #force_inline proc(x, y: T) -> T {return x / y})
}

// Unary operators

exp :: proc(arr: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_unary_op(arr, #force_inline proc(x: T) -> T {return math.exp(x)})
}

log :: proc(arr: ^Array_Dyn($T)) -> ^Array_Dyn(T) {
	return _array_unary_op(arr, #force_inline proc(x: T) -> T {return math.ln(x)})
}

softmax :: proc {
	softmax_a,
	softmax_t,
}

softmax_a :: proc(arr: ^Array_Dyn($T), axis: uint) -> ^Array_Dyn(T) {
	maxima := max(arr, axis, true)
	arr_sub := sub(arr, maxima)
	numerator := exp(arr_sub)
	denominator := sum(numerator, axis, true)
	defer array_free(maxima, arr_sub, numerator, denominator)
	return div(numerator, denominator)
}

softmax_t :: proc(t: ^Tensor($T), axis: uint) -> ^Tensor(T) {
	axis_tensor := new_with_init([]T{T(axis)}, {1})
	defer tensor_release(axis_tensor)

	return autograd_make_op(
		[]^Tensor(T){t, axis_tensor},
		new_arrdata = softmax_a(t.arrdata, axis),
		backward_fn_name = "softmax_backward",
		backward_fn = proc(tensor: ^Tensor(T), upstream_grad: ^Array_Dyn(T)) {
			input := tensor.deps[0]
			if !input.requires_grad {
				return
			}

			// Get stored axis from deps
			axis := uint(tensor.deps[1].data[0])

			// Get softmax output which we need for gradient
			s := softmax_a(input.arrdata, axis)
			defer array_free(s)

			// grad = s * (upstream_grad - sum(s * upstream_grad))
			s_times_upstream := mul(s, upstream_grad)
			defer array_free(s_times_upstream)

			sum_term := sum(s_times_upstream, axis, keepdims = true)
			defer array_free(sum_term)

			diff := sub(upstream_grad, sum_term)
			defer array_free(diff)

			local_grad := mul(s, diff)
			defer array_free(local_grad)

			old_grad := input.grad
			input.grad = add(old_grad, local_grad)
			array_free(old_grad)
		},
	)
}
