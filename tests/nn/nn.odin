package tests_nn

import ar "../../arraydyn"
import nn "../../arraydyn/nn"
import "core:fmt"
import "core:slice"
import "core:testing"

BaseFloat :: f32

@(test)
test_layer_linear :: proc(t: ^testing.T) {
	l := nn.layer_linear_new(BaseFloat, 3, 2, false)
	defer nn.layer_linear_free(l)

	dummy := ar.new_with_init([]f32{1, 2, 3, 4, 5, 6}, {2, 3})
	res := nn.layer_linear_forward(l, dummy)
	defer ar.tensor_release(dummy, res)

	testing.expect(t, slice.equal(res.shape, []uint{2, 2}))
}

@(test)
test_layer_embedding :: proc(t: ^testing.T) {
	l := nn.layer_embedding_new(f32, 3, 5)
	defer nn.layer_embedding_free(l)

	idx := ar.new_with_init([]f32{0, 1, 2, 1, 0, 0, 0, 1}, {2, 4})
	defer ar.tensor_release(idx)
	res := nn.layer_embedding_forward(l, idx)
	defer ar.tensor_release(res)

	// Check output shape - should be [2, 4, 5]
	// (batch_size=2, seq_len=4, embedding_dim=5)
	testing.expect(t, slice.equal(res.shape, []uint{2, 4, 5}))

	// Check embedding values
	testing.expect(
		t,
		slice.equal(res.data[0:5], l.weight.data[0:5]), // First embedding (idx 0)
		fmt.tprintf("First embedding mismatch: %v vs %v", res.data[0:5], l.weight.data[0:5]),
	)

	testing.expect(
		t,
		slice.equal(res.data[5:10], l.weight.data[5:10]), // Second embedding (idx 1)
		fmt.tprintf("Second embedding mismatch: %v vs %v", res.data[5:10], l.weight.data[5:10]),
	)

	testing.expect(
		t,
		slice.equal(res.data[10:15], l.weight.data[10:15]), // Third embedding (idx 2)
		fmt.tprintf("Third embedding mismatch: %v vs %v", res.data[10:15], l.weight.data[10:15]),
	)

	// Test gradients
	ar.set_requires_grad(l.weight, true)
	total := ar.sum(res)
	ar.backward(total)
	defer ar.tensor_release(total)

	// Each index occurrence should contribute 1 to its embedding's gradient
	// idx 0 appears 4 times, idx 1 appears 3 times, idx 2 appears once
	expected_counts := []f32{4, 3, 1}
	for i := 0; i < 3; i += 1 {
		for j := 0; j < 5; j += 1 {
			testing.expect(
				t,
				l.weight.grad.data[i * 5 + j] == expected_counts[i],
				fmt.tprintf(
					"Gradient mismatch for embedding %v dim %v: expected %v, got %v",
					i,
					j,
					expected_counts[i],
					l.weight.grad.data[i * 5 + j],
				),
			)
		}
	}
}
