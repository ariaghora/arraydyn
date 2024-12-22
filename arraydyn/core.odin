package arraydyn

import "core:fmt"
import "core:mem"

Array_Dyn :: struct($T: typeid) {
	data:       []T,
	shape:      []uint,
	strides:    []uint,
	contiguous: bool,
}

_shape_to_size :: #force_inline proc(shape: []uint) -> uint {
	size: uint = 1
	for s in shape {size *= s}
	return size
}

_array_alloc :: proc($T: typeid, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = new(Array_Dyn(T))
	size := _shape_to_size(shape)
	res.data = make([]T, size)
	res.shape = make([]uint, len(shape))
	res.strides = make([]uint, len(shape))
	res.contiguous = true

	// initialize shape and strides
	copy(res.shape, shape)
	stride: uint = 1
	for i := len(shape) - 1; i >= 0; i -= 1 {
		res.strides[i] = stride
		stride *= shape[i]
	}
	return res
}

clone :: proc(arr: ^Array_Dyn($T)) -> (res: ^Array_Dyn(T)) {
	res = new(Array_Dyn(T))
	res.data = make([]T, len(arr.data))
	res.shape = make([]uint, len(arr.shape))
	res.strides = make([]uint, len(arr.strides))
	res.contiguous = arr.contiguous

	copy(res.data, arr.data)
	copy(res.shape, arr.shape)
	copy(res.strides, arr.strides)

	return res
}

data_len :: proc(arr: ^Array_Dyn) -> uint {
	return _shape_to_size(arr.shape)
}

ones :: proc($T: typeid, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	for _, i in res.data {res.data[i] = T(1)}
	return res
}

new_with_init :: proc($T: typeid, init: []T, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	assert(len(res.data) == len(init))

	copy(res.data, init)
	return res
}

tranpose :: proc(arr: ^Array_Dyn($T)) -> (res: ^Array_Dyn(T)) {
	assert(len(arr.shape) == 2)

	new_shape := []uint{arr.shape[1], arr.shape[0]}
	res = _array_alloc(T, new_shape)

	rows := int(arr.shape[0])
	cols := int(arr.shape[1])

	for i in 0 ..< rows {
		for j in 0 ..< cols {
			res.data[j * rows + i] = arr.data[i * cols + j]
		}
	}

	return res
}

array_free :: proc(arr: ^Array_Dyn($T)) {
	delete(arr.data)
	delete(arr.shape)
	delete(arr.strides)
	free(arr)
}
