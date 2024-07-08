use rand::prelude::*;
use candle_core::{Device, DType, Tensor, Result, D};
use candle_nn::{Conv2d, Linear, Dropout, VarMap, VarBuilder, ModuleT, Optimizer};

const LABELS: usize = 10;
const BSIZE: usize = 64;

#[derive(Debug)]
struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl ConvNet {
    fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}

struct TrainingArgs {
    learning_rate: f64,
    epochs: usize,
}

pub fn main() -> anyhow::Result<()> {
    let training_args = TrainingArgs {
        learning_rate: 0.001,
        epochs: 4,
    };

    let m = candle_datasets::vision::mnist::load()?;
    println!("train-images: {:?}", m.train_images.shape());
    println!("train-labels: {:?}", m.train_labels.shape());
    println!("test-images: {:?}", m.test_images.shape());
    println!("test-labels: {:?}", m.test_labels.shape());

    let dev = Device::cuda_if_available(0)?;

    let train_labels: Tensor = m.train_labels;
    let train_images: Tensor = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ConvNet::new(vs.clone())?;

    let mut adamw = candle_nn::AdamW::new_lr(varmap.all_vars(), training_args.learning_rate)?;

    let test_images: Tensor = m.test_images.to_device(&dev)?;
    let test_labels: Tensor = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let n_batches = train_images.dim(0)? / BSIZE;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..training_args.epochs + 1 {
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());

        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * BSIZE, BSIZE)?;
            let train_labels = train_labels.narrow(0, batch_idx * BSIZE, BSIZE)?;

            let logits = model.forward(&train_images, true)?;
            let log_sm = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
            let loss = candle_nn::loss::nll(&log_sm, &train_labels)?;
            adamw.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }

        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }
    Ok(())
}
