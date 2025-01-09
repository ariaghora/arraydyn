package main

import ar "../arraydyn"
import nn "../arraydyn/nn"
import "core:fmt"

MAX_EPOCH :: 5000

main :: proc() {
	x := ar.new_with_init([]f32{0, 0, 1, 0, 0, 1, 1, 1}, {4, 2})
	y := ar.new_with_init([]f32{0, 1, 1, 0}, {4})

	// inputs -> 10 hidden units -> 2 output units
	l1 := nn.layer_linear_new(f32, 2, 10, use_bias = true)
	defer nn.layer_linear_free(l1)
	l2 := nn.layer_linear_new(f32, 10, 2, use_bias = true)
	defer nn.layer_linear_free(l2)
	relu := nn.layer_relu_new(f32)
	defer nn.layer_relu_free(relu)

	// storing predictions at the end of the training loop
	preds_logits: ^ar.Tensor(f32)

	for i in 1 ..= MAX_EPOCH {
		h1 := nn.layer_linear_forward(l1, x)
		h1_relu := nn.layer_relu_forward(relu, h1)
		h2 := nn.layer_linear_forward(l2, h1_relu)
		loss := nn.loss_crossentropy_with_logit(h2, y)

		ar.backward(loss)

		if i % 500 == 0 {
			fmt.printfln("Loss at %d: %v ", i, loss.data[0])
		}

		sgd_update([]^ar.Tensor(f32){l1.weight, l1.bias, l2.weight, l2.bias})

		// reset gradient
		ar.zero_grad(l1.weight)
		ar.zero_grad(l1.bias)
		ar.zero_grad(l2.weight)
		ar.zero_grad(l2.bias)

		// cleanups
		ar.tensor_release(h1, h1_relu, loss)
		if i < MAX_EPOCH {
			ar.tensor_release(h2)
		} else {
			preds_logits = h2
		}
	}
	fmt.println("================= Finished Training =================\n")

	preds_classes := ar.argmax(preds_logits, 1)
	defer ar.tensor_release(preds_logits, preds_classes)

	fmt.printfln("XOR Inputs:")
	ar.print(x)
	fmt.printfln("XOR Predictions:")
	ar.print(preds_classes)
	fmt.printfln("XOR Actual:")
	ar.print(y)
}

sgd_update :: proc(tensors: []^ar.Tensor(f32), lr: f32 = 0.01) {
	for t in tensors {
		//scale gradients with some learning rate
		for _, i in t.grad.data {
			t.grad.data[i] *= lr
		}

		// update the model parameters
		new_arrdata := ar.sub(t.arrdata, t.grad)
		ar.array_free(t.arrdata) // free the old parameter arrdata
		t.arrdata = new_arrdata
	}
}
