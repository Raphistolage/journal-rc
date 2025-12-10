#[repr(u8)]
#[derive(PartialEq)]
pub enum Errors {
    NoErrors = 0,
    IncompatibleRanks = 1,
    IncompatibleShapes = 2,
}

