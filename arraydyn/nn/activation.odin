package nn

import ar "../"

Layer_ReLU :: struct($T: typeid) {}

layer_relu_new :: proc($T: typeid) -> ^Layer_ReLU(T) {
	res := new(Layer_ReLU(T))
	return res
}

layer_relu_forward :: proc(l: ^Layer_ReLU($T), x: ^ar.Tensor(T)) -> ^ar.Tensor(T) {
	relu := ar.clone(x.arrdata)
	// Forward pass: max(0, x)
	for i in 0 ..< len(relu.data) {
		relu.data[i] = max(0, relu.data[i])
	}

	return ar.autograd_make_op(
		deps = []^ar.Tensor(T){x},
		new_arrdata = relu,
		backward_fn = proc(t: ^ar.Tensor(T), grad_output: ^ar.Array_Dyn(T)) {
			// ReLU derivative: 1 if x > 0, 0 otherwise
			relu_output := t.data // Relu forward pass output
			input_tensor := t.deps[0] // Original input tensor

			for i in 0 ..< len(input_tensor.data) {
				input_tensor.grad.data[i] += grad_output.data[i] * T(relu_output[i] > 0 ? 1 : 0)
			}
		},
		backward_fn_name = "relu_backward",
	)
}

layer_relu_free :: proc(l: ^Layer_ReLU($T)) {
	free(l)
}
