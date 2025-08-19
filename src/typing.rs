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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeLimitType {
    None,
    MaxInput,
    Size(SizeType),
}

/* #region implementations */

impl From<SizeType> for SizeLimitType {
    fn from(size: SizeType) -> Self {
        SizeLimitType::Size(size)
    }
}

impl From<Option<SizeType>> for SizeLimitType {
    fn from(size: Option<SizeType>) -> Self {
        match size {
            Some(s) => SizeLimitType::Size(s),
            None => SizeLimitType::None,
        }
    }
}

impl From<&'static str> for SizeLimitType {
    fn from(size: &'static str) -> Self {
        match size.replace("_", "-").replace(" ", "-").to_lowercase().as_str() {
            "max-input" => SizeLimitType::MaxInput,
            "" | "none" | "no-limit" => SizeLimitType::None,
            _ => panic!("Invalid MemoryLimitType string: {size}"),
        }
    }
}

/* #endregion */
