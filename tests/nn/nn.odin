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
