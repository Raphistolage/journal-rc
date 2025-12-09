use std::fmt::Debug;

use crate::common_types::Layout;

pub trait LayoutType: Default + Debug {
    fn to_layout(&self) -> Layout;
}

#[derive(Default, Debug)]
pub struct LayoutLeft();
#[derive(Default, Debug)]
pub struct LayoutRight();
#[derive(Default, Debug)]
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
