package tests

import ar "../arraydyn"
import "core:slice"
import "core:testing"


@(test)
test_broadcast_grad_to_shape :: proc(t: ^testing.T) {
	grad := ar.ones(f32, {10})
	grad_bcast := ar.broadcast_grad_to_shape(grad.arrdata, {32, 10})
	defer ar.tensor_release(grad)
	defer ar.array_free(grad_bcast)

	testing.expect_value(t, len(grad_bcast.shape), 2)
	testing.expect_value(t, grad_bcast.shape[0], 32)
	testing.expect_value(t, grad_bcast.shape[1], 10)

	// Check that all values are 1 since we broadcast from ones
	data := ar._get_strided_data(grad_bcast)
	defer delete(data)
	for val in data {
		testing.expect_value(t, val, 1)
	}
}

@(test)
test_broadcast_grad_to_shape_mlp :: proc(t: ^testing.T) {
	x := ar.ones(f32, {32, 64}) // batch_size x input_dim
	w := ar.ones(f32, {64, 128}) // input_dim x output_dim
	b := ar.ones(f32, {128}) // output_dim
	ar.set_requires_grad(x, true)
	ar.set_requires_grad(w, true)
	ar.set_requires_grad(b, true)
	defer ar.tensor_release(x, w, b)

	// Forward pass
	wx := ar.matmul(x, w)
	out := ar.add(wx, b)
	defer ar.tensor_release(wx, out)

	// Backward pass
	ar.backward(out)

	// Test gradient shapes and values
	testing.expect_value(t, len(b.grad.shape), 1)
	testing.expect_value(t, b.grad.shape[0], 128)

	data := ar._get_strided_data(b.grad)
	defer delete(data)
	for val in data {
		testing.expect_value(t, val, 32) // sum over batch dimension
	}
}
