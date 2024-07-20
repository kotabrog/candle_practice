use anyhow::Result;
use candle_core::Tensor;

pub fn assert_eq_tensor(t1: &Tensor, t2: &Tensor) -> Result<()> {
    assert_eq!(t1.shape(), t2.shape());
    assert_eq!(t1.dtype(), t2.dtype());
    assert!(t1
        .eq(t2)?
        .flatten_all()?
        .to_vec1::<u8>()?
        .iter()
        .all(|&x| x == 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_assert_eq_tensor() -> Result<()> {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0], (3, 1, 1, 1), &Device::Cpu)?;
        let t2 = Tensor::from_slice(&[1.0, 2.0, 3.0], (3, 1, 1, 1), &Device::Cpu)?;
        assert_eq_tensor(&t1, &t2)?;
        Ok(())
    }
}
