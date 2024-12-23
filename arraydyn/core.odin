package arraydyn

import "core:fmt"
import "core:mem"
import "core:slice"
import "core:strings"

// This structure implements a high level multidimensional array container using a
// linear array of data internally. The array is parametrized for any data type
// and supports strided access and broadcasting. The internal representation uses
// a C-contiguous storage layout (row-major order) with all the data stored in a
// single slice. A stride array is used to map N-dimensional coordinates to linear
// indices in the data array. The contiguous flag indicates if the array is stored
// in memory without gaps.
Array_Dyn :: struct($T: typeid) {
	data:       []T,
	shape:      []uint,
	strides:    []uint,
	contiguous: bool,
}

// Compute total size of an array by multiplying dimensions in shape
_shape_to_size :: #force_inline proc(shape: []uint) -> uint {
	size: uint = 1
	for s in shape {size *= s}
	return size
}

// Create a new n-dimensional array with the given shape. For each dimension i
// in the array shape[i] represents the size of that dimension. After allocation
// the array elements are left uninitialized.
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

// Get array data respecting strides. If shape and strides match the original
// array and data is contiguous, return a simple clone. Otherwise rearrange
// the data following target shape and strides to handle non-contiguous cases.
// Under the hood this function converts non-contiguous array storage into
// contiguous one.
_get_strided_data :: proc(arr: ^Array_Dyn($T), shape: []uint = nil, strides: []uint = nil) -> []T {
	target_strides := strides
	target_shape := shape
	if (strides == nil) {
		target_strides = arr.strides
	}
	if (shape == nil) {
		target_shape = arr.shape
	}

	if arr.contiguous &&
	   (slice.equal(arr.shape, target_shape) || shape == nil) &&
	   (slice.equal(arr.strides, target_strides) || strides == nil) {
		return slice.clone(arr.data)
	}

	// return strided
	size := _shape_to_size(target_shape)
	data := make([]T, size)
	i: uint
	for i in 0 ..< size {
		i_strided := _compute_strided_index(target_shape, target_strides, i)
		data[i] = arr.data[i_strided]
	}
	return data
}

// Deep copy of array data. The copy will be an exact replica of the original
// array, with exactly the same data, shape, strides and contiguous flag. The
// resulting array will be completely independent from the source.
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

// Returns the total number of elements in the array by multiplying all dimensions
// together. Takes into account that some dimensions may be empty (0) in which
// case the total size will also be zero. This function does not care about strides
// or how data is laid out in memory, it is just about the logical size.
data_len :: proc(arr: ^Array_Dyn) -> uint {
	return _shape_to_size(arr.shape)
}

// Create a new array filled with ones. Arrays are initialized for all data types by casting
// 1 to the target type, so for example this works with floating point data types, integers
// and even complex data types like bool or void types.
ones :: proc($T: typeid, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	for _, i in res.data {res.data[i] = T(1)}
	return res
}

// Create a new array with given data and shape. This function performs a copy
// of the input data, so the original array is not referenced in the new one.
new_with_init :: proc(init: []$T, shape: []uint) -> (res: ^Array_Dyn(T)) {
	res = _array_alloc(T, shape)
	if len(res.data) != len(init) {
		panic("Input data length must match array size computed from shape")
	}

	copy(res.data, init)
	return res
}

// Pretty print array with numpy-like formatting
print :: proc(arr: ^Array_Dyn($T)) {
	if len(arr.shape) == 0 {
		fmt.print("Array()")
		return
	}

	builder := strings.builder_make()
	defer strings.builder_destroy(&builder)

	strings.write_string(
		&builder,
		fmt.tprintf("Array(type=%v,shape=%v)\n", typeid_of(T), arr.shape),
	)
	indices := make([]uint, len(arr.shape))
	defer delete(indices)
	_print_recursive(arr, arr.shape, arr.strides, 0, indices, 1, &builder)
	strings.write_string(&builder, "\n")

	fmt.println(strings.to_string(builder))
}

_print_recursive :: proc(
	arr: ^Array_Dyn($T),
	shape: []uint,
	strides: []uint,
	depth: int,
	indices: []uint,
	indent: int,
	builder: ^strings.Builder,
) {
	if depth == len(shape) - 1 {
		// For the innermost dimension
		strings.write_byte(builder, '[')

		for i in 0 ..< shape[depth] {
			indices[depth] = i
			index := _compute_linear_index(indices, strides)

			if i > 0 {
				strings.write_byte(builder, ' ')
			}

			value := arr.data[index]
			switch type_info_of(type_of(value)) {
			case type_info_of(f32):
				strings.write_string(builder, fmt.tprintf("%.6f", value))
			case type_info_of(f64):
				strings.write_string(builder, fmt.tprintf("%.6f", value))
			case:
				strings.write_string(builder, fmt.tprintf("% 6v", value))
			}
		}
		strings.write_byte(builder, ']')
	} else {
		strings.write_byte(builder, '[')

		for i in 0 ..< shape[depth] {
			if i > 0 {
				strings.write_string(builder, "\n")
				for j := 0; j < depth + 1; j += 1 {
					strings.write_string(builder, " ")
				}
			}

			indices[depth] = i
			_print_recursive(arr, shape, strides, depth + 1, indices, indent + 1, builder)

			if i < shape[depth] - 1 {
				strings.write_byte(builder, ',')
			}
		}
		strings.write_string(builder, "]")
	}
}

_compute_linear_index :: proc(indices: []uint, strides: []uint) -> uint {
	index: uint = 0
	for i in 0 ..< len(indices) {
		index += indices[i] * strides[i]
	}
	return index
}

array_free :: proc(arr: ^Array_Dyn($T)) {
	delete(arr.data)
	delete(arr.shape)
	delete(arr.strides)
	free(arr)
}
