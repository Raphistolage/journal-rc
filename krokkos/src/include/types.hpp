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
        HostSpace = 1,
        DeviceSpace = 2,
    };

    enum Layout : uint8_t {
        LayoutLeft = 0,
        LayoutRight = 1,
        LayoutStride = 2,
    };

}