use candle_core::Device;
use hf_hub::api::sync::Api;

fn main() {
    let api = Api::new().unwrap();
    let repo = api.model("bert-base-uncased".to_string());
    let weights = repo.get("model.safetensors").unwrap();
    let weights = candle_core::safetensors::load(weights, &Device::Cpu);
    println!("{:?}", weights);
}
