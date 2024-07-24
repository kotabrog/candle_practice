use anyhow::Result;
use candle_core::Device;
use candle_datasets::vision::Dataset;
use std::path::Path;

use super::CacheDataset;

pub fn load_with_cache_path<P: AsRef<Path>>(file_path: P, device: &Device) -> Result<Dataset> {
    if file_path.as_ref().exists() {
        Dataset::load_cache(file_path, device)
    } else {
        let m = candle_datasets::vision::mnist::load()?;
        m.save_cache(file_path)?;
        Ok(m)
    }
}

pub fn load(device: &Device) -> Result<Dataset> {
    let path = Path::new("~/.cache/candle_ext/mnist.pt");
    load_with_cache_path(path, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Shape;

    #[test]
    fn test_load() -> Result<()> {
        let file_path = Path::new("/root/.cache/candle_ext/mnist.pt");
        let device = Device::cuda_if_available(0)?;
        let dataset = load_with_cache_path(file_path, &device)?;
        assert_eq!(dataset.train_images.shape(), &Shape::from(&[60000, 784]));
        assert_eq!(dataset.train_labels.shape(), &Shape::from(&[60000]));
        assert_eq!(dataset.test_images.shape(), &Shape::from(&[10000, 784]));
        assert_eq!(dataset.test_labels.shape(), &Shape::from(&[10000]));
        assert_eq!(dataset.labels, 10);
        assert!(file_path.exists());
        Ok(())
    }
}
