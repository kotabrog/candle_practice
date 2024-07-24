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

    #[test]
    #[should_panic]
    fn test_assert_eq_tensor_fail_value() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0], (3, 1, 1, 1), &Device::Cpu).unwrap();
        let t2 = Tensor::from_slice(&[1.0, 2.0, 4.0], (3, 1, 1, 1), &Device::Cpu).unwrap();
        assert_eq_tensor(&t1, &t2).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_assert_eq_tensor_fail_shape() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 1, 2, 1), &Device::Cpu).unwrap();
        let t2 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 1, 1, 2), &Device::Cpu).unwrap();
        assert_eq_tensor(&t1, &t2).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_assert_eq_tensor_fail_dtype() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0], (3, 1, 1, 1), &Device::Cpu).unwrap();
        let t2 = Tensor::from_slice(&[1 as i64, 2, 3], (3, 1, 1, 1), &Device::Cpu).unwrap();
        assert_eq_tensor(&t1, &t2).unwrap();
    }
}
