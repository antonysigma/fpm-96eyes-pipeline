#pragma once
#include <highfive/bits/H5DataType_misc.hpp>

namespace HighFive {

template <>
inline AtomicType<std::complex<float>>::AtomicType() {
    static struct ComplexType : public Object {
        ComplexType() {
            _hid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
            // h5py/numpy compatible datatype
            H5Tinsert(_hid, "r", 0, H5T_NATIVE_FLOAT);
            H5Tinsert(_hid, "i", sizeof(float), H5T_NATIVE_FLOAT);
        };
    } complexType;
    _hid = H5Tcopy(complexType.getId());
}

}  // namespace HighFive