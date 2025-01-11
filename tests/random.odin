package tests

import ar "../arraydyn"
import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_multinomial :: proc(t: ^testing.T) {
	// Simple case: equal probabilities
	probs := ar.new_with_init([]f32{0.5, 0.5}, {1, 2})
	defer ar.tensor_release(probs)

	samples := ar.multinomial_a(probs.arrdata, 10)
	defer ar.array_free(samples)

	fmt.println("Samples:", samples.data)

	// Verify output shape
	testing.expect(t, slice.equal(samples.shape, []uint{1, 10}))

	// Verify values are 0 or 1
	for v in samples.data {
		testing.expect(t, v == 0 || v == 1)
	}
}
