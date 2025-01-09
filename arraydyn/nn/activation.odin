package nn

import ar "../"


relu :: proc(x: ^ar.Tensor($T)) -> ^ar.Tensor(T) {
	arrdata := ar.clone(x.arrdata)
	// Forward pass: max(0, x)
	for i in 0 ..< len(arrdata.data) {
		arrdata.data[i] = max(0, arrdata.data[i])
	}

	return ar.autograd_make_op(
		deps = []^ar.Tensor(T){x},
		new_arrdata = arrdata,
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
