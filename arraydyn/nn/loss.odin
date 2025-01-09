package nn

import ar "../"

loss_crossentropy_with_logit :: proc(logits, targets: ^ar.Tensor($T)) -> ^ar.Tensor(T) {
	if len(logits.shape) != 2 {
		panic("logits must be 2D tensor [batch_size, num_classes]")
	}
	if len(targets.shape) != 1 {
		panic("targets must be 1D tensor [batch_size] containing class indices")
	}
	if targets.shape[0] != logits.shape[0] {
		panic("batch sizes must match")
	}

	return ar.autograd_make_op(
		[]^ar.Tensor(T){logits, targets},
		new_arrdata = ce_forward(logits.arrdata, targets.arrdata),
		backward_fn_name = "crossentropy_backward",
		backward_fn = proc(tensor: ^ar.Tensor(T), upstream_grad: ^ar.Array_Dyn(T)) {
			logits, targets := tensor.deps[0], tensor.deps[1]

			if logits.requires_grad {
				old_grad := logits.grad
				grad_logits := ce_backward(logits.arrdata, targets.arrdata, upstream_grad)
				logits.grad = ar.add(old_grad, grad_logits)
				ar.array_free(old_grad, grad_logits)
			}
		},
	)
}

ce_forward :: proc(logits, targets: ^ar.Array_Dyn($T)) -> ^ar.Array_Dyn(T) {
	// Get max for numerical stability
	max_logits := ar.max(logits, 1, true) // [batch, 1]
	defer ar.array_free(max_logits)

	// Compute log_softmax = logits - max - log(sum(exp(logits - max)))
	shifted_logits := ar.sub(logits, max_logits)
	defer ar.array_free(shifted_logits)

	exp_logits := ar.exp(shifted_logits)
	defer ar.array_free(exp_logits)

	sum_exp := ar.sum(exp_logits, 1, keepdims = true) // [batch, 1]
	defer ar.array_free(sum_exp)

	log_sum := ar.log(sum_exp)
	defer ar.array_free(log_sum)

	log_softmax := ar.sub(shifted_logits, log_sum)
	defer ar.array_free(log_softmax)

	// Convert targets to loss using batched indexing
	// TODO: Need to implement gather op to index using targets
	// For now using loop (this part needs array ops support)
	loss := ar._array_alloc(T, []uint{1})
	batch_size := logits.shape[0]
	for i: uint = 0; i < batch_size; i += 1 {
		target_idx := uint(ar.array_get(targets, i))
		loss.data[0] -= ar.array_get(log_softmax, i, target_idx)
	}

	// Average over batch
	loss.data[0] /= T(batch_size)
	return loss
}

ce_backward :: proc(logits, targets, upstream_grad: ^ar.Array_Dyn($T)) -> ^ar.Array_Dyn(T) {
	// Compute softmax
	max_logits := ar.max(logits, 1, keepdims = true)
	defer ar.array_free(max_logits)

	shifted_logits := ar.sub(logits, max_logits)
	defer ar.array_free(shifted_logits)

	exp_logits := ar.exp(shifted_logits)
	defer ar.array_free(exp_logits)

	sum_exp := ar.sum(exp_logits, 1, keepdims = true)
	defer ar.array_free(sum_exp)

	probs := ar.div(exp_logits, sum_exp)
	defer ar.array_free(probs)

	// TODO: Need gather/scatter ops to handle target indexing with array ops
	// For now using loop for the target part
	grad := ar.clone(probs)
	defer ar.array_free(grad)
	batch_size := logits.shape[0]
	for i: uint = 0; i < batch_size; i += 1 {
		target_idx := uint(ar.array_get(targets, i))
		grad.data[i * grad.strides[0] + target_idx * grad.strides[1]] -= 1
	}

	// Scale by upstream grad and batch size
	nom := ar._ones(T, grad.shape)
	denom := ar._new_with_init([]T{T(batch_size)}, []uint{1})
	scale := ar.div(nom, denom)
	defer ar.array_free(nom, denom, scale)

	return ar.mul(grad, scale)
}
