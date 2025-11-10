pub trait Dimension {
    const NDIM: u8;

    fn ndim(&self) -> u8;
    /// Compute the size of the dimension (number of elements)
    fn size(&self) -> usize
    {
        self.slice().iter().product()
    }

    fn slice(&self) -> &[usize];
    
    fn to_vec(&self) -> Vec<usize> {
        self.slice().to_vec()
    }
}

pub struct Dim1 {
    shape: [usize; 1],
}

impl Dim1 {
    pub fn new(shape: &[usize; 1]) -> Self{
        Dim1 {
            shape: *shape
        }
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

impl Dimension for Dim1 {
    const NDIM: u8 = 1;

    fn ndim(&self) -> u8 {
        1
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}

pub struct Dim2 {
    shape: [usize; 2],
}

impl Dim2 {
    pub fn new(shape: &[usize; 2]) -> Self{
        Dim2 {
            shape: *shape
        }
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

impl Dimension for Dim2 {
    const NDIM: u8 = 2;

    fn ndim(&self) -> u8 {
        2
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}

pub struct Dim3 {
    shape: [usize; 3]
}

impl Dim3 {
    pub fn new(shape: &[usize; 3]) -> Self{
        Dim3 {
            shape: *shape
        }
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

impl Dimension for Dim3 {
    const NDIM: u8 = 3;

    fn ndim(&self) -> u8 {
        3
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}

pub struct Dim4 {
    shape: [usize; 4]
}

impl Dim4 {
    pub fn new(shape: &[usize; 4]) -> Self{
        Dim4 {
            shape: *shape
        }
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

impl Dimension for Dim4 {
    const NDIM: u8 = 4;

    fn ndim(&self) -> u8 {
        4
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}

pub struct Dim5 {
    shape: [usize; 5]
}

impl Dim5 {
    pub fn new(shape: &[usize; 5]) -> Self{
        Dim5 {
            shape: *shape
        }
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

impl Dimension for Dim5 {
    const NDIM: u8 = 5;

    fn ndim(&self) -> u8 {
        5
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}

pub struct Dim6 {
    shape: [usize; 6]
}

impl Dim6 {
    pub fn new(shape: &[usize; 6]) -> Self{
        Dim6 {
            shape: *shape
        }
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

impl Dimension for Dim6 {
    const NDIM: u8 = 6;

    fn ndim(&self) -> u8 {
        6
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}

pub struct Dim7 {
    shape: [usize; 7]
}

impl Dim7 {
    pub fn new(shape: &[usize; 7]) -> Self{
        Dim7 {
            shape: *shape
        }
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

impl Dimension for Dim7 {
    const NDIM: u8 = 7;

    fn ndim(&self) -> u8 {
        7
    }

    fn slice(&self) -> &[usize] {
        self.shapes()
    }
}