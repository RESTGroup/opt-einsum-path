use crate::*;

pub type SizeType = f64;
pub type TensorShapeType = Vec<usize>;
pub type PathType = Vec<TensorShapeType>;
pub type ArrayIndexType = BTreeSet<char>;
pub type SizeDictType = BTreeMap<char, usize>;
pub use crate::paths::OptimizeKind;

#[derive(Debug, Clone)]
pub struct ContractionType {
    pub indices: String,
    pub idx_rm: ArrayIndexType,
    pub einsum_str: String,
    pub remaining: Option<Vec<String>>,
    pub do_blas: Option<&'static str>,
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
        match size.replace("_", "-").replace(" ", "-").to_lowercase().as_str() {
            "max-input" => MemoryLimitType::MaxInput,
            "" | "none" | "no-limit" => MemoryLimitType::None,
            _ => panic!("Invalid MemoryLimitType string: {size}"),
        }
    }
}

/* #endregion */
