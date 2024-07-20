use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_datasets::vision::Dataset;
use std::{collections::HashMap, path::Path};

pub trait CashDataset {
    fn save_cash<P: AsRef<Path>>(&self, file_path: P) -> Result<()>;
    fn load_cash<P: AsRef<Path>>(file_path: P, device: &Device) -> Result<Self>
    where
        Self: Sized;
}

impl CashDataset for Dataset {
    fn save_cash<P: AsRef<Path>>(&self, file_path: P) -> Result<()> {
        let file_path = file_path.as_ref();
        let data = vec![
            ("train_images", self.train_images.clone()),
            ("train_labels", self.train_labels.clone()),
            ("test_images", self.test_images.clone()),
            ("test_labels", self.test_labels.clone()),
            ("labels", Tensor::try_from(self.labels as u32)?),
        ]
        .into_iter()
        .collect::<HashMap<&str, Tensor>>();
        Ok(candle_core::safetensors::save(&data, file_path)?)
    }

    fn load_cash<P: AsRef<Path>>(file_path: P, device: &Device) -> Result<Self>
    where
        Self: Sized,
    {
        let file_path = file_path.as_ref();
        let mut data = candle_core::safetensors::load(file_path, device)?;
        let path_display = file_path.display();
        let train_images = data
            .remove("train_images")
            .ok_or_else(|| anyhow::anyhow!("train_images not found in {}", path_display))?;
        let train_labels = data
            .remove("train_labels")
            .ok_or_else(|| anyhow::anyhow!("train_labels not found in {}", path_display))?;
        let test_images = data
            .remove("test_images")
            .ok_or_else(|| anyhow::anyhow!("test_images not found in {}", path_display))?;
        let test_labels = data
            .remove("test_labels")
            .ok_or_else(|| anyhow::anyhow!("test_labels not found in {}", path_display))?;
        let labels = data
            .remove("labels")
            .ok_or_else(|| anyhow::anyhow!("labels not found in {}", path_display))?
            .to_scalar::<u32>()? as usize;
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
            labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::assert_eq_tensor;
    use candle_core::{Device, Tensor};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn test_save_dataset() -> Result<()> {
        let mean = 0.0;
        let std = 1.0;

        let dev = Device::cuda_if_available(0)?;

        let dataset = Dataset {
            train_images: Tensor::randn(mean, std, (100, 1, 28, 28), &dev)?,
            train_labels: Tensor::randn(mean, std, (100,), &dev)?,
            test_images: Tensor::randn(mean, std, (100, 1, 28, 28), &dev)?,
            test_labels: Tensor::randn(mean, std, (100,), &dev)?,
            labels: 10,
        };
        let file_path = PathBuf::from("test_save_dataset.pt");
        dataset.save_cash(&file_path).unwrap();
        fs::remove_file(&file_path).unwrap();
        Ok(())
    }

    #[test]
    fn test_load_dataset() -> Result<()> {
        let mean = 0.0;
        let std = 1.0;

        let dev = Device::cuda_if_available(0)?;

        let dataset = Dataset {
            train_images: Tensor::randn(mean, std, (100, 1, 28, 28), &dev)?,
            train_labels: Tensor::randn(mean, std, (100,), &dev)?,
            test_images: Tensor::randn(mean, std, (100, 1, 28, 28), &dev)?,
            test_labels: Tensor::randn(mean, std, (100,), &dev)?,
            labels: 10,
        };
        let file_path = PathBuf::from("test_load_dataset.pt");
        dataset.save_cash(&file_path).unwrap();
        let loaded_dataset = Dataset::load_cash(&file_path, &dev).unwrap();
        fs::remove_file(&file_path).unwrap();
        assert_eq_tensor(&dataset.train_images, &loaded_dataset.train_images)?;
        assert_eq_tensor(&dataset.train_labels, &loaded_dataset.train_labels)?;
        assert_eq_tensor(&dataset.test_images, &loaded_dataset.test_images)?;
        assert_eq_tensor(&dataset.test_labels, &loaded_dataset.test_labels)?;
        assert_eq!(dataset.labels, loaded_dataset.labels);
        Ok(())
    }

    // #[test]
    // fn test_mnist() -> Result<()> {
    //     let m = candle_datasets::vision::mnist::load()?;
    //     println!("train-images: {:?}", m.train_images.shape());
    //     println!("train-labels: {:?}", m.train_labels.shape());
    //     println!("test-images: {:?}", m.test_images.shape());
    //     println!("test-labels: {:?}", m.test_labels.shape());
    //     println!("labels: {:?}", m.labels);
    //     panic!()
    // }
}
