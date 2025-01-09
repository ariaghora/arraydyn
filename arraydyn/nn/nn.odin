package nn

import ar "../"
import "core:math"
import "core:math/rand"

Layer_Linear :: struct($T: typeid) {
	weight:   ^ar.Tensor(T),
	bias:     ^ar.Tensor(T),
	use_bias: bool,
}

xavier_init :: proc($T: typeid, in_size, out_size: uint) -> ^ar.Tensor(T) {
	scale := math.sqrt(2.0 / f64(in_size + out_size))
	res := ar._tensor_from_array(ar._array_alloc(T, {in_size, out_size}))
	for i in 0 ..< len(res.arrdata.data) {
		res.arrdata.data[i] = T((rand.float32_uniform(0, 1) * 2 - 1) * T(scale))
	}
	return res
}

layer_linear_new :: proc($T: typeid, in_size, out_size: uint, use_bias: bool) -> ^Layer_Linear(T) {
	res := new(Layer_Linear(T))

	res.weight = xavier_init(T, in_size, out_size)
	ar.set_requires_grad(res.weight, true)

	res.use_bias = use_bias
	if use_bias {
		res.bias = ar.zeros(T, {out_size})
		ar.set_requires_grad(res.bias, true)
	}
	return res
}

layer_linear_forward :: proc(l: ^Layer_Linear($T), x: ^ar.Tensor(T)) -> ^ar.Tensor(T) {
	affine := ar.matmul(x, l.weight)
	if l.use_bias {
		affine_plus_bias := ar.add(affine, l.bias)
		ar.tensor_release(affine)
		return affine_plus_bias
	}
	return affine
}

layer_linear_free :: proc(l: ^Layer_Linear($T)) {
	ar.tensor_release(l.weight)
	if l.use_bias {ar.tensor_release(l.bias)}
	free(l)
}

Layer_Embedding :: struct($T: typeid) {
	weight:         ^ar.Tensor(T),
	num_embeddings: uint,
	embedding_dim:  uint,
	counts:         map[T]T,
}

layer_embedding_new :: proc(
	$T: typeid,
	num_embeddings, embedding_dim: uint,
) -> ^Layer_Embedding(T) {
	res := new(Layer_Embedding(T))
	res.weight = xavier_init(T, num_embeddings, embedding_dim)
	ar.set_requires_grad(res.weight, true)
	res.num_embeddings = num_embeddings
	res.embedding_dim = embedding_dim
	return res
}

layer_embedding_forward :: proc(l: ^Layer_Embedding($T), indices: ^ar.Tensor(T)) -> ^ar.Tensor(T) {
	if len(indices.shape) != 2 {
		panic("Indices tensor must be a rank-2 tensor")
	}
	b, t := indices.shape[0], indices.shape[1]
	out_shape := []uint{b, t, l.embedding_dim}
	indices_flatten: []T
	if !indices.contiguous {
		indices_flatten = ar._get_strided_data(indices.arrdata)
		defer delete(indices_flatten)
	} else {
		indices_flatten = indices.data
	}

	data_out := make([]T, ar._shape_to_size(out_shape))
	defer delete(data_out)

	for i := uint(0); i < b * t; i += 1 {
		idx := uint(indices_flatten[i])
		if idx >= l.num_embeddings {
			panic("Index out of bounds")
		}

		// Copy embedding vector for this index
		start := i * l.embedding_dim
		emb_start := idx * l.embedding_dim
		emb_end := emb_start + l.embedding_dim
		copy(data_out[start:start + l.embedding_dim], l.weight.data[emb_start:emb_end])
	}

	result := ar._new_with_init(data_out, out_shape)
	embedding_dim_tensor := ar.new_with_init([]T{T(l.embedding_dim)}, {1})
	defer ar.tensor_release(embedding_dim_tensor)

	return ar.autograd_make_op(
		[]^ar.Tensor(T){l.weight, indices, embedding_dim_tensor},
		new_arrdata = result,
		backward_fn_name = "embedding_backward",
		backward_fn = proc(tensor: ^ar.Tensor(T), upstream_grad: ^ar.Array_Dyn(T)) {
			weight, indices, embedding_dim_tensor := tensor.deps[0], tensor.deps[1], tensor.deps[2]
			embedding_dim := uint(embedding_dim_tensor.data[0])

			if weight.requires_grad {
				old_grad := weight.grad
				grad_weight := ar._zeros(T, weight.shape)

				indices_flatten: []T
				if !indices.contiguous {
					indices_flatten = ar._get_strided_data(indices.arrdata)
					defer delete(indices_flatten)
				} else {
					indices_flatten = indices.data
				}

				// Accumulate gradients for each index
				b, t := indices.shape[0], indices.shape[1]
				for i := uint(0); i < b * t; i += 1 {
					idx := uint(indices_flatten[i])
					start := i * embedding_dim
					emb_start := idx * embedding_dim

					// Add gradient for this embedding vector
					for j := uint(0); j < embedding_dim; j += 1 {
						grad_weight.data[emb_start + j] += upstream_grad.data[start + j]
					}
				}

				weight.grad = ar.add(old_grad, grad_weight)
				ar.array_free(old_grad, grad_weight)
			}
		},
	)
}

layer_embedding_free :: proc(l: ^Layer_Embedding($T)) {
	ar.tensor_release(l.weight)
	delete(l.counts)
	free(l)
}
