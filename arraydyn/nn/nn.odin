package nn

import ar "../"
import "core:math"
import "core:math/rand"

Layer_Linear :: struct($T: typeid) {
	weight:   ^ar.Tensor(T),
	bias:     ^ar.Tensor(T),
	use_bias: bool,
}

layer_linear_new :: proc($T: typeid, in_size, out_size: uint, use_bias: bool) -> ^Layer_Linear(T) {
	res := new(Layer_Linear(T))

	// Xavier
	denom := math.sqrt(2.0 / T(in_size + out_size))
	res.weight = ar._tensor_from_array(ar._array_alloc(T, {in_size, out_size}))
	for i in 0 ..< len(res.weight.arrdata.data) {
		res.weight.arrdata.data[i] = T((rand.float32() * 2 - 1) * denom)
	}
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
	defer ar.tensor_release(affine)
	if l.use_bias {
		affine_plus_bias := ar.add(affine, l.bias)
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
}

layer_embedding_new :: proc(
	$T: typeid,
	num_embeddings, embedding_dim: uint,
) -> ^Layer_Embedding(T) {
	res := new(Layer_Embedding(T))
	res.weight = ar.randn(T, {num_embeddings, embedding_dim}, T(0), T(0.02))
	res.num_embeddings = num_embeddings
	res.embedding_dim = embedding_dim
	return res
}

layer_embedding_forward :: proc(l: ^Layer_Embedding($T), indices: ^ar.Tensor(T)) -> ^ar.Tensor(T) {
	unimplemented("not implemented yet")
}

layer_embedding_free :: proc(l: ^Layer_Embedding($T)) {
	ar.tensor_release(l.weight)
	free(l)
}
