pub mod contract;
pub mod helpers;
pub mod paths;

use std::collections::{BTreeMap, BTreeSet};

pub type TensorShapeType = Vec<usize>;
pub type PathType = Vec<TensorShapeType>;
pub type ArrayIndexType = BTreeSet<char>;
pub type SizeDictType = BTreeMap<char, usize>;
pub type ContractionListType = Vec<(usize, usize, String, Option<Vec<String>>, bool)>;
