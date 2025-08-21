//! # Notes for API Document
//!
//! The most important function is [`contract_path`], which is used to find the optimal contraction
//! path for a given einsum expression.
#![doc = include_str!("../readme.md")]
pub mod blas;
pub mod contract;
pub mod helpers;
pub mod parser;
pub mod paths;
pub mod typing;

pub use contract::contract_path;
pub use paths::PathOptimizer;

pub(crate) use crate::typing::*;
pub(crate) use itertools::Itertools;
pub(crate) use num::{BigUint, FromPrimitive, One, ToPrimitive, Zero};
pub(crate) use rand::Rng;
pub(crate) use std::borrow::Borrow;
pub(crate) use std::cmp::Reverse;
pub(crate) use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
