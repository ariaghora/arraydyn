package tests_nn

import ar "../../arraydyn"
import nn "../../arraydyn/nn"
import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_layer_relu :: proc(t: ^testing.T) {
	// Test with mixed positive and negative values
	x := ar.new_with_init([]f32{-2, -1, 0, 1, 2}, {5})
	res := nn.relu(x)
	defer ar.tensor_release(x, res)

	// Expected: [0, 0, 0, 1, 2]
	expected := []f32{0, 0, 0, 1, 2}
	for i := 0; i < len(expected); i += 1 {
		testing.expect(
			t,
			res.data[i] == expected[i],
			fmt.tprintf("Expected %f, got %f at index %d", expected[i], res.data[i], i),
		)
	}
}

@(test)
test_layer_relu_backward :: proc(t: ^testing.T) {
	// Test gradient computation
	// Input tensor with requires_grad
	x := ar.new_with_init([]f32{-2, -1, 0, 1, 2}, {5})
	ar.set_requires_grad(x, true)

	// Forward pass
	out := nn.relu(x)

	// Backward pass
	ar.backward(out)
	defer ar.tensor_release(x, out)

	// Expected gradients: [0, 0, 0, 1, 1]
	// ReLU derivative is 1 for positive inputs, 0 for negative
	expected_grad := []f32{0, 0, 0, 1, 1}
	for i := 0; i < len(expected_grad); i += 1 {
		testing.expect(
			t,
			x.grad.data[i] == expected_grad[i],
			fmt.tprintf(
				"Expected gradient %f, got %f at index %d",
				expected_grad[i],
				x.grad.data[i],
				i,
			),
		)
	}
}

@(test)
test_layer_relu_zero :: proc(t: ^testing.T) {
	// Test with all zeros
	x := ar.new_with_init([]f32{0, 0, 0}, {3})
	res := nn.relu(x)
	defer ar.tensor_release(x, res)

	// Expected: all zeros remain zeros
	for i := 0; i < len(res.data); i += 1 {
		testing.expect(
			t,
			res.data[i] == 0,
			fmt.tprintf("Expected 0, got %f at index %d", res.data[i], i),
		)
	}
}

@(test)
test_layer_relu_positive :: proc(t: ^testing.T) {
	// Test with all positive values
	x := ar.new_with_init([]f32{1, 2, 3}, {3})
	res := nn.relu(x)
	defer ar.tensor_release(x, res)

	// Expected: positive values should remain unchanged
	expected := []f32{1, 2, 3}
	for i := 0; i < len(expected); i += 1 {
		testing.expect(
			t,
			res.data[i] == expected[i],
			fmt.tprintf("Expected %f, got %f at index %d", expected[i], res.data[i], i),
		)
	}
}
