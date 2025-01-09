package main

import ar "../arraydyn"
import nn "../arraydyn/nn"
import "core:fmt"

MAX_EPOCH :: 5000

// SGD update rule, θ = θ - lr * ∇θ
sgd_update :: proc(tensors: ..^ar.Tensor(f32), lr: f32 = 0.01) {
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

main :: proc() {
	// x_1 | x_2 | y
	//-----+-----+---
	//  0  |  0  | 0
	//  1  |  0  | 1
	//  0  |  1  | 1
	//  1  |  1  | 0
	x := ar.new_with_init([]f32{0, 0, 1, 0, 0, 1, 1, 1}, shape = {4, 2})
	y := ar.new_with_init([]f32{0, 1, 1, 0}, shape = {4})

	// Model weights to map 2 input units -> 10 hidden units -> 2 output units
	w1, b1 := ar.ones(f32, {2, 10}), ar.zeros(f32, {10})
	w2, b2 := ar.ones(f32, {10, 2}), ar.zeros(f32, {2})
	defer ar.tensor_release(w1, b1, w2, b2)

	ar.set_requires_grad(w1, true)
	ar.set_requires_grad(b1, true)
	ar.set_requires_grad(w2, true)
	ar.set_requires_grad(b2, true)


	// Storing predictions at the end of the training loop
	preds_logits: ^ar.Tensor(f32)

	// Main training loop
	for i in 1 ..= MAX_EPOCH {
		// Input to hidden
		h1 := ar.matmul(x, w1)
		h1_b := ar.add(h1, b1)
		h1_relu := nn.relu(h1_b)

		// Hidden to output
		h2 := ar.matmul(h1_relu, w2)
		h2_b := ar.add(h2, b2)

		// Compute loss and do backward propagation to compute all gradients
		loss := nn.loss_crossentropy_with_logit(h2_b, y)
		ar.backward(loss)

		defer ar.tensor_release(h1, h1_b, h1_relu, h2, loss)

		// SGD update rule, θ = θ - lr * ∇θ
		sgd_update(w1, b1, w2, b2)

		// reset gradient
		ar.zero_grad(w1, b1, w2, b2)

		// Printing loss
		if i % 500 == 0 || i == 1 {fmt.printfln("Loss at %d: %v ", i, loss.data[0])}

		if i < MAX_EPOCH {
			ar.tensor_release(h2_b)
		} else {
			preds_logits = h2_b
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
