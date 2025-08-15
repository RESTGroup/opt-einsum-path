use crate::*;

pub type SizeType = usize;
pub type TensorShapeType = Vec<usize>;
pub type PathType = Vec<TensorShapeType>;
pub type ArrayIndexType = BTreeSet<char>;
pub type SizeDictType = BTreeMap<char, usize>;
pub use crate::contract::OptimizeKind;

#[derive(Debug, Clone)]
pub struct ContractionType {
    pub indices: String,
    pub idx_rm: ArrayIndexType,
    pub einsum_str: String,
    pub remaining: Option<Vec<String>>,
    pub do_blas: bool,
}

pub enum MemoryLimitType {
    None,
    MaxInput,
    Size(SizeType),
}

/* #region implementations */

impl From<SizeType> for MemoryLimitType {
    fn from(size: SizeType) -> Self {
        MemoryLimitType::Size(size)
    }
}

impl From<Option<SizeType>> for MemoryLimitType {
    fn from(size: Option<SizeType>) -> Self {
        match size {
            Some(s) => MemoryLimitType::Size(s),
            None => MemoryLimitType::None,
        }
    }
}

impl From<&'static str> for MemoryLimitType {
    fn from(size: &'static str) -> Self {
        match size {
            "max_input" => MemoryLimitType::MaxInput,
            _ => panic!("Invalid MemoryLimitType string: {size}"),
        }
    }
}

/* #endregion */
