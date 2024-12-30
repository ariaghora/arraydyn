package tests

import ar "../arraydyn"

import "core:math"
import "core:slice"
import "core:testing"

@(test)
test_add_autograd :: proc(t: ^testing.T) {
	// Test x + y
	a := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_free(a)
	ar.set_requires_grad(a, true)
	b := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	ar.set_requires_grad(b, true)
	defer ar.tensor_free(b)

	c := ar.add(a, b)
	ar.backward(c)
	defer ar.tensor_free(c)

	// After backward, a.grad and b.grad should each contain ones
	// since derivative of addition is 1 for both inputs
	expected_grad := ar.ones(i32, {2, 2})
	defer ar.array_free(expected_grad)

	testing.expect(t, slice.equal(a.grad.data, expected_grad.data))
	testing.expect(t, slice.equal(b.grad.data, expected_grad.data))

	// Test x + x
	x := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_free(x)
	ar.set_requires_grad(x, true)

	y := ar.add(x, x)
	defer ar.tensor_free(y)
	ar.backward(y)

	// For x + x, gradient should be 2 since each x contributes 1
	expected_grad_2 := ar.new_with_init([]i32{2, 2, 2, 2}, {2, 2})
	defer ar.tensor_free(expected_grad_2)
	testing.expect(t, slice.equal(x.grad.data, expected_grad_2.data))

	// // Test x + x + x
	xx := ar.new_with_init([]i32{1, 2, 3, 4}, {2, 2})
	defer ar.tensor_free(xx)
	ar.set_requires_grad(xx, true)

	yy := ar.add(xx, xx)
	defer ar.tensor_free(yy)
	yyy := ar.add(yy, xx)
	defer ar.tensor_free(yyy)
	ar.backward(yyy)

	// For x + x + x, gradient should be 3
	expected_grad_3 := ar.new_with_init([]i32{3, 3, 3, 3}, {2, 2})
	defer ar.tensor_free(expected_grad_3)
	testing.expect(t, slice.equal(xx.grad.data, expected_grad_3.data))
}
