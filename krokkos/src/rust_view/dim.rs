use std::fmt::Debug;

pub trait Dimension: Debug + Into<Vec<usize>> + Clone + Default {
    const NDIM: u8;

    fn ndim(&self) -> u8;
    /// Compute the size of the dimension (number of elements)
    fn size(&self) -> usize {
        self.slice().iter().product()
    }

    fn slice(&self) -> &[usize];

    fn to_vec(&self) -> Vec<usize> {
        self.slice().to_vec()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str>;
}

#[derive(Debug, Clone, Default)]
pub struct Dim1 {
    shape: [usize; 1],
}

impl Dim1 {
    pub fn new(shape: &[usize; 1]) -> Self {
        Dim1 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 1] {
        &self.shape
    }
}

impl From<Dim1> for Vec<usize> {
    fn from(value: Dim1) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 1]> for Dim1 {
    fn from(value: &[usize; 1]) -> Self {
        Dim1 { shape: *value }
    }
}

impl Dimension for Dim1 {
    const NDIM: u8 = 1;

    fn ndim(&self) -> u8 {
        1
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str> {
        if slice.len() == Self::NDIM as usize {
            Ok(Dim1 { shape: [slice[0]] })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dim2 {
    shape: [usize; 2],
}

impl Dim2 {
    pub fn new(shape: &[usize; 2]) -> Self {
        Dim2 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 2] {
        &self.shape
    }
}

impl From<Dim2> for Vec<usize> {
    fn from(value: Dim2) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 2]> for Dim2 {
    fn from(value: &[usize; 2]) -> Self {
        Dim2 { shape: *value }
    }
}

impl Dimension for Dim2 {
    const NDIM: u8 = 2;

    fn ndim(&self) -> u8 {
        2
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(value: &[usize]) -> Result<Self, &'static str> {
        if value.len() == Self::NDIM as usize {
            Ok(Dim2 {
                shape: [value[0], value[1]],
            })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dim3 {
    shape: [usize; 3],
}

impl Dim3 {
    pub fn new(shape: &[usize; 3]) -> Self {
        Dim3 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 3] {
        &self.shape
    }
}

impl From<Dim3> for Vec<usize> {
    fn from(value: Dim3) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 3]> for Dim3 {
    fn from(value: &[usize; 3]) -> Self {
        Dim3 { shape: *value }
    }
}

impl Dimension for Dim3 {
    const NDIM: u8 = 3;

    fn ndim(&self) -> u8 {
        3
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str> {
        if slice.len() == Self::NDIM as usize {
            Ok(Dim3 {
                shape: [slice[0], slice[1], slice[2]],
            })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dim4 {
    shape: [usize; 4],
}

impl Dim4 {
    pub fn new(shape: &[usize; 4]) -> Self {
        Dim4 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 4] {
        &self.shape
    }
}

impl From<Dim4> for Vec<usize> {
    fn from(value: Dim4) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 4]> for Dim4 {
    fn from(value: &[usize; 4]) -> Self {
        Dim4 { shape: *value }
    }
}

impl Dimension for Dim4 {
    const NDIM: u8 = 4;

    fn ndim(&self) -> u8 {
        4
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str> {
        if slice.len() == Self::NDIM as usize {
            Ok(Dim4 {
                shape: [slice[0], slice[1], slice[2], slice[3]],
            })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dim5 {
    shape: [usize; 5],
}

impl Dim5 {
    pub fn new(shape: &[usize; 5]) -> Self {
        Dim5 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 5] {
        &self.shape
    }
}

impl From<Dim5> for Vec<usize> {
    fn from(value: Dim5) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 5]> for Dim5 {
    fn from(value: &[usize; 5]) -> Self {
        Dim5 { shape: *value }
    }
}

impl Dimension for Dim5 {
    const NDIM: u8 = 5;

    fn ndim(&self) -> u8 {
        5
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str> {
        if slice.len() == Self::NDIM as usize {
            Ok(Dim5 {
                shape: [slice[0], slice[1], slice[2], slice[3], slice[4]],
            })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dim6 {
    shape: [usize; 6],
}

impl Dim6 {
    pub fn new(shape: &[usize; 6]) -> Self {
        Dim6 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 6] {
        &self.shape
    }
}

impl From<Dim6> for Vec<usize> {
    fn from(value: Dim6) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 6]> for Dim6 {
    fn from(value: &[usize; 6]) -> Self {
        Dim6 { shape: *value }
    }
}

impl Dimension for Dim6 {
    const NDIM: u8 = 6;

    fn ndim(&self) -> u8 {
        6
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str> {
        if slice.len() == Self::NDIM as usize {
            Ok(Dim6 {
                shape: [slice[0], slice[1], slice[2], slice[3], slice[4], slice[5]],
            })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Dim7 {
    shape: [usize; 7],
}

impl Dim7 {
    pub fn new(shape: &[usize; 7]) -> Self {
        Dim7 { shape: *shape }
    }

    pub fn shapes(&self) -> &[usize; 7] {
        &self.shape
    }
}

impl From<Dim7> for Vec<usize> {
    fn from(value: Dim7) -> Self {
        value.shapes().into()
    }
}

impl From<&[usize; 7]> for Dim7 {
    fn from(value: &[usize; 7]) -> Self {
        Dim7 { shape: *value }
    }
}

impl Dimension for Dim7 {
    const NDIM: u8 = 7;

    fn ndim(&self) -> u8 {
        7
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }

    fn try_from_slice(slice: &[usize]) -> Result<Self, &'static str> {
        if slice.len() == Self::NDIM as usize {
            Ok(Dim7 {
                shape: [
                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6],
                ],
            })
        } else {
            Err("Invalid size slice for Dim1")
        }
    }
}
