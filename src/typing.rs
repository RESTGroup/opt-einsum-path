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
