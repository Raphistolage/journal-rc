use crate::common_types::Layout;

pub trait LayoutType {
    fn to_layout(&self) -> Layout;
}

pub struct LayoutLeft();
pub struct LayoutRight();
pub struct LayoutStride(
    //TODO : add strides
);

impl LayoutType for LayoutLeft {
    fn to_layout(&self) -> Layout {
        Layout::LayoutLeft
    }
}

impl LayoutType for LayoutRight {
    fn to_layout(&self) -> Layout {
        Layout::LayoutRight
    }
}

impl LayoutType for LayoutStride {
    fn to_layout(&self) -> Layout {
        Layout::LayoutStride
    }
}