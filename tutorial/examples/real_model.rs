use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use hf_hub::api::sync::Api;

fn main() {
    let api = Api::new().unwrap();
    let repo = api.model("bert-base-uncased".to_string());
    let weights = repo.get("model.safetensors").unwrap();
    let weights = candle_core::safetensors::load(weights, &Device::Cpu).unwrap();

    let weight = weights
        .get("bert.encoder.layer.0.attention.self.query.weight")
        .unwrap();
    let bias = weights
        .get("bert.encoder.layer.0.attention.self.query.bias")
        .unwrap();

    let linear = Linear::new(weight.clone(), Some(bias.clone()));

    let input_ids = Tensor::zeros((3, 768), DType::F32, &Device::Cpu).unwrap();
    let output = linear.forward(&input_ids).unwrap();
    println!("{:?}", output);
}
