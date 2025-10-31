#include <cstdint>

extern "C" {
    enum DataType : uint8_t {
        Float = 1,
        Unsigned = 2,
        Signed = 3,
    };

    enum Errors : uint8_t{
        NoErrors = 0,
        IncompatibleRanks = 1,
        IncompatibleShapes = 2,
    };

    enum MemSpace : uint8_t {
        CudaSpace = 1,
        CudaHostPinnedSpace = 2,
        HIPSpace = 3,
        HIPHostPinnedSpace = 4,
        HIPManagedSpace = 5,
        HostSpace = 6,
        SharedSpace = 7,
        SYCLDeviceUSMSpace = 8,
        SYCLHostUSMSpace = 9,
        SYCLSharedUSMSpace = 10,
    };

    enum Layout : uint8_t {
        LayoutLeft,
        LayoutRight,
        LayoutStride,
    };

}