use crate::*;

pub struct NoOptimize;

impl PathOptimizer for NoOptimize {
    fn optimize_path(
        &mut self,
        inputs: &[&ArrayIndexType],
        _output: &ArrayIndexType,
        _size_dict: &SizeDictType,
        _memory_limit: Option<SizeType>,
    ) -> PathType {
        vec![(0..inputs.len()).collect()]
    }
}
