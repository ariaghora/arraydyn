package tests

import ar "../arraydyn"

import "core:math"
import "core:slice"
import "core:testing"

@(test)
test_add_autograd :: proc(t: ^testing.T) {
	// Test x + y
	a := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(a)
	ar.set_requires_grad(a, true)
	b := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	ar.set_requires_grad(b, true)
	defer ar.tensor_release(b)

	c := ar.add(a, b)
	ar.backward(c)
	defer ar.tensor_release(c)

	// After backward, a.grad and b.grad should each contain ones
	// since derivative of addition is 1 for both inputs
	expected_grad := ar._ones(i32, {2, 2})
	defer ar.array_free(expected_grad)

	testing.expect(t, slice.equal(a.grad.data, expected_grad.data))
	testing.expect(t, slice.equal(b.grad.data, expected_grad.data))

	// Test x + x
	x := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(x)
	ar.set_requires_grad(x, true)

	y := ar.add(x, x)
	defer ar.tensor_release(y)
	ar.backward(y)

	// For x + x, gradient should be 2 since each x contributes 1
	expected_grad_2 := ar.new_with_init([]i32{2, 2, 2, 2}, {2, 2})
	defer ar.tensor_release(expected_grad_2)
	testing.expect(t, slice.equal(x.grad.data, expected_grad_2.data))

	// // Test x + x + x
	xx := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(xx)
	ar.set_requires_grad(xx, true)

	yy := ar.add(xx, xx)
	defer ar.tensor_release(yy)
	yyy := ar.add(yy, xx)
	defer ar.tensor_release(yyy)
	ar.backward(yyy)

	// For x + x + x, gradient should be 3
	expected_grad_3 := ar.new_with_init([]i32{3, 3, 3, 3}, {2, 2})
	defer ar.tensor_release(expected_grad_3)
	testing.expect(t, slice.equal(xx.grad.data, expected_grad_3.data))
}

@(test)
test_leak_from_complex_ops_inside_fn :: proc(t: ^testing.T) {
	// Create input tensors
	a := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(a)
	ar.set_requires_grad(a, true)
	b := ar.new_with_init([]i32{5, 6, 7, 8}, {2, 2})
	defer ar.tensor_release(b)
	ar.set_requires_grad(b, true)

	// Complex operation in a function
	complex_add :: proc(x, y: ^ar.Tensor(i32)) -> ^ar.Tensor(i32) {
		temp1 := ar.add(x, y)
		defer ar.tensor_release(temp1)
		temp2 := ar.add(temp1, x)
		defer ar.tensor_release(temp2)
		temp3 := ar.add(temp2, temp2)
		return temp3
	}

	// Test the complex operation
	result := complex_add(a, b)
	defer ar.tensor_release(result)
	ar.backward(result)

	// For this operation ((x + y) + x) + ((x + y) + x)
	// Gradient for x should be 4 (contributes twice in each temp2)
	// Gradient for y should be 2 (contributes once in each temp2)
	expected_grad_x := ar.new_with_init([]i32{4, 4, 4, 4}, {2, 2})
	defer ar.tensor_release(expected_grad_x)
	expected_grad_y := ar.new_with_init([]i32{2, 2, 2, 2}, {2, 2})
	defer ar.tensor_release(expected_grad_y)

	testing.expect(t, slice.equal(a.grad.data, expected_grad_x.data))
	testing.expect(t, slice.equal(b.grad.data, expected_grad_y.data))
}

@(test)
test_mul_autograd :: proc(t: ^testing.T) {
	// Test x * y
	a := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(a)
	ar.set_requires_grad(a, true)
	b := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	ar.set_requires_grad(b, true)
	defer ar.tensor_release(b)

	c := ar.mul(a, b)
	ar.backward(c)
	defer ar.tensor_release(c)

	// For multiplication, gradient for each input should be the other input
	// since d(a*b)/da = b and d(a*b)/db = a
	testing.expect(t, slice.equal(a.grad.data, b.arrdata.data))
	testing.expect(t, slice.equal(b.grad.data, a.arrdata.data))

	// Test x * x
	x := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(x)
	ar.set_requires_grad(x, true)

	y := ar.mul(x, x)
	defer ar.tensor_release(y)
	ar.backward(y)

	// For x * x, gradient should be 2x since d(x^2)/dx = 2x
	expected_grad := ar.new_with_init([]i32{2, 4, 6, 8}, {2, 2})
	defer ar.tensor_release(expected_grad)
	testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
}

@(test)
test_matmul_autograd :: proc(t: ^testing.T) {
	// Test matrix multiplication
	a := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(a)
	ar.set_requires_grad(a, true)
	b := ar.new_with_init([]i32{5, 6, 7, 8}, {2, 2})
	ar.set_requires_grad(b, true)
	defer ar.tensor_release(b)

	c := ar.matmul(a, b)
	ar.backward(c)
	defer ar.tensor_release(c)

	// For matmul C = A @ B:
	// dC/dA = C @ B.T
	// dC/dB = A.T @ C
	expected_grad_a := ar.new_with_init([]i32{11, 15, 11, 15}, {2, 2})
	defer ar.tensor_release(expected_grad_a)
	expected_grad_b := ar.new_with_init([]i32{4, 4, 6, 6}, {2, 2})
	defer ar.tensor_release(expected_grad_b)
	testing.expect(t, slice.equal(a.grad.data, expected_grad_a.data))
	testing.expect(t, slice.equal(b.grad.data, expected_grad_b.data))

	// Test matmul with same matrix: x @ x
	x := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_release(x)
	ar.set_requires_grad(x, true)

	y := ar.matmul(x, x)
	defer ar.tensor_release(y)
	ar.backward(y)

	// For x @ x, gradient combines both dC/dA and dC/dB
	expected_grad := ar.new_with_init([]i32{7, 11, 9, 13}, {2, 2})
	defer ar.tensor_release(expected_grad)
	testing.expect(t, slice.equal(x.grad.data, expected_grad.data))
}
