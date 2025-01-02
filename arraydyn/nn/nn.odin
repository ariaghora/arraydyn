package nn

import ar "../"

Layer_Linear :: struct($T: typeid) {
	weight:   ^ar.Tensor(T),
	bias:     ^ar.Tensor(T),
	use_bias: bool,
}

layer_linear_new :: proc($T: typeid, in_size, out_size: uint, use_bias: bool) -> ^Layer_Linear(T) {
	res := new(Layer_Linear(T))
	res.weight = ar.randn(T, {in_size, out_size}, T(0), T(1))
	res.use_bias = use_bias
	if use_bias {res.bias = ar.zeros(T, {out_size})}
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
