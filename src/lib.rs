use anyhow::{self, Result};
use candle_core::{DType, Device, Tensor, WithDType, D};
use candle_nn as nn;
use candle_nn::{Module, VarBuilder, RNN};
use std::fs::File;


const FLOATING_DTYPE: DType = DType::F32;
const LONG_DTYPE: DType = DType::I64;
const MODEL_SAFETENSORS_FILENAME: &str = "./model/model.safetensors";


fn load_optispeech_cnx_model() ->Result<OptiSpeechCNXModel> {
    let config = OptiSpeechCNXConfig::default();
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[MODEL_SAFETENSORS_FILENAME], FLOATING_DTYPE, &device
        )?
    };
    let model = OptiSpeechCNXModel::load(config, vb)?;
    Ok(model)
}

#[derive(Debug, Clone)]
struct OptiSpeechCNXConfig {
    n_vocab: usize,
    max_source_positions: usize,
    num_enc_layers: usize,
    num_dec_layers: usize,
    enc_dec_dim: usize,
    enc_dec_intermediate_dim: usize,
    num_vocoder_layers: usize,
    vocoder_dim: usize,
    vocoder_intermediate_dim: usize,
    n_fft: usize,
    hop_size: usize
}

impl Default for OptiSpeechCNXConfig {
    fn default() -> Self {
        Self {
            n_vocab: 250,
            max_source_positions: 2000,
            num_enc_layers: 4,
            num_dec_layers: 4,
            enc_dec_dim: 256,
            enc_dec_intermediate_dim: 1024,
            num_vocoder_layers: 8,
            vocoder_dim: 384,
            vocoder_intermediate_dim: 1280,
            n_fft: 2048,
            hop_size: 300
        }
    }
}

struct ScaledSinusoidalEmbedding {
    scale: f64,
    inv_freq: Tensor,
}

impl ScaledSinusoidalEmbedding {
    const THETA: &'static [f32] = &[2000.0];

    fn load(vb: &VarBuilder, dim: usize) -> Result<Self> {
        let scale = vb.get(&[1], "scale")?.squeeze(0)?.to_scalar::<f32>()? as f64;
        let half_dim = dim / 2;
        let freq_seq = Tensor::arange(0f32, half_dim as f32, &Device::Cpu)?;
        let freq_seq = (freq_seq / half_dim as f64)?;
        let inv_freq = Tensor::from_slice(Self::THETA, (1,), &Device::Cpu)?
            .broadcast_pow(&(freq_seq * -1.0f64)?)?;
        Ok(Self { scale, inv_freq })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, seq_length, _dim) = x.dims3()?;
        let pos = Tensor::arange(0f32, seq_length as f32, x.device())?;
        let mut emb = pos.unsqueeze(1)?.broadcast_mul(&self.inv_freq)?;
        emb = Tensor::cat(&[emb.sin()?, emb.cos()?], 1)?;
        Ok((emb * self.scale)?)
    }
}


struct TextEmbedding {
    embed_scale: Tensor,
    embed_tokens: nn::Embedding,
    embed_positions: ScaledSinusoidalEmbedding,
}

impl TextEmbedding {
    fn load(vb: &VarBuilder, n_vocab: usize, dim: usize, max_source_positions: usize) -> Result<Self> {
        let embed_scale = Tensor::from_vec(
            vec![(dim as f32).sqrt()],
            1,
            &Device::Cpu
        )?;
        let embed_tokens = nn::embedding(n_vocab, dim, vb.pp("embed_tokens"))?;
        let embed_positions = ScaledSinusoidalEmbedding::load(
            &vb.pp("embed_positions"),
            dim,
        )?;
        Ok(Self { embed_scale, embed_tokens, embed_positions })
    }
    fn forward(&self, src_tokens: &Tensor) -> Result<Tensor> {
        let embed_tokens = self.embed_tokens.forward(&src_tokens)?;
        let embed = embed_tokens.broadcast_mul(&self.embed_scale)?;
        let positions = self.embed_positions.forward(&src_tokens)?;
        let x = embed.add(&positions)?;
        Ok(x)
    }
}

struct ConvNeXtLayer {
    dwconv: nn::Conv1d,
    norm: nn::LayerNorm,
    pwconv1: nn::Linear,
    // act: nn::GELU,
    pwconv2: nn::Linear,
    gamma: Tensor
}
    
impl ConvNeXtLayer {
    fn load(vb: &VarBuilder, dim: usize, intermediate_dim: usize) -> Result<Self> {
        let dwconv = nn::conv1d(
            dim,
            dim,
            7,
            nn::conv::Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: dim,
            },
            vb.pp("dwconv")
        )?;
        let norm = nn::layer_norm(
            dim,
            nn::LayerNormConfig {
                eps: 1e-6,
                remove_mean: false,
                affine: true
            },
            vb.pp("norm"),
        )?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1: nn::linear(dim, intermediate_dim, vb.pp("pwconv1"))?,
            pwconv2: nn::linear(intermediate_dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.dwconv.forward(&x)?;
        let x = x.transpose(1, 2)?;  // (B, C, T) -> (B, T, C)
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.pwconv2.forward(&x)?;
        let x = self.gamma.matmul(&x)?;
        let x = x.transpose(1, 2)?;
        let x = x.add(residual)?;
        Ok(x)
    }
}

struct ConvNeXtBackbone {
    layers: Vec<ConvNeXtLayer>,
    final_layer_norm: nn::LayerNorm,
    layer_scale_init_value: f32
}


impl ConvNeXtBackbone {
    fn load(
        vb: &VarBuilder,
        num_layers: usize,
        dim: usize,
        intermediate_dim: usize,
        layer_scale_init_value: Option<f32>
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = ConvNeXtLayer::load(&vb.pp(format!("convnext.{i}")), dim, intermediate_dim)?;
            layers.push(layer);
        }
        let final_layer_norm = nn::layer_norm(
            dim,
            nn::LayerNormConfig {
                eps: 1e-6,
                remove_mean: false,
                affine: true
            },
            vb.pp("final_layer_norm")
        )?;
        let layer_scale_init_value = layer_scale_init_value.unwrap_or(1.0 / num_layers as f32);
        Ok(Self { layers, final_layer_norm, layer_scale_init_value})
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?;
        for layer in self.layers.iter() {
            let x = layer.forward(&x)?;
            // TODO: handle mask
        }
        let x = x.transpose(1, 2)?;
        let x = self.final_layer_norm.forward(&x)?;
        Ok(x)
    }
}

struct WaveNeXtVocoder {
    embed: nn::Conv1d,
    norm: nn::LayerNorm,
    backbone: ConvNeXtBackbone,
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl WaveNeXtVocoder {
    fn load(
        vb: &VarBuilder,
        num_layers: usize,
        input_dim: usize,
        dim: usize,
        intermediate_dim: usize,
        layer_scale_init_value: Option<f32>,
        n_fft: usize,
        hop_size: usize
    ) -> Result<Self> {
        let embed = nn::conv1d(
            input_dim,
            dim,
            7,
            nn::conv::Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1
            },
            vb.pp("embed")
        )?;
        let norm = nn::layer_norm(
            dim,
            nn::LayerNormConfig {
                eps: 1e-6,
                remove_mean: false,
                affine: true
            },
            vb.pp("norm")
        )?;
        let backbone =ConvNeXtBackbone::load(
            &vb.pp("backbone"),
            num_layers,
            dim,
            intermediate_dim,
            layer_scale_init_value,
        )?;
         // head
        let l_fft = n_fft + 2;
        let l_shift = hop_size;
        let linear_1 = nn::linear(dim, l_fft, vb.pp("head.linear_1"))?;
        let linear_2 = nn::linear_no_bias(l_fft, l_shift, vb.pp("head.linear_2"))?;
        Ok(Self {embed, norm, backbone, linear_1, linear_2})
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(&x)?;
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self.backbone.forward(&x)?; // masking
        // Head
        let (b, c, t) = x.dims3()?;
        let x = self.linear_1.forward(&x)?;
        let x = self.linear_2.forward(&x)?;
        let audio = x.reshape((b, c * t))?;
        let audio = audio.clamp(-1.0, 1.0)?;
        Ok(audio)
    }
}

struct OptiSpeechCNXModel {
    text_embedding: TextEmbedding,
    encoder: ConvNeXtBackbone,
    decoder: ConvNeXtBackbone,
    vocoder: WaveNeXtVocoder,
}

impl OptiSpeechCNXModel {
    fn load(config: OptiSpeechCNXConfig, vb: VarBuilder) -> Result<Self> {
        let text_embedding = TextEmbedding::load(
            &vb.pp("text_embedding"),
            config.n_vocab,
            config.enc_dec_dim,
            config.max_source_positions
        )?;
        let encoder =ConvNeXtBackbone::load(
            &vb.pp("encoder"),
            config.num_enc_layers,
            config.enc_dec_dim,
            config.enc_dec_intermediate_dim,
            None,
        )?;
        let decoder =ConvNeXtBackbone::load(
            &vb.pp("decoder"),
            config.num_dec_layers,
            config.enc_dec_dim,
            config.enc_dec_intermediate_dim,
            None,
        )?;
        let vocoder = WaveNeXtVocoder::load(
            &vb.pp("wav_generator"),
            config.num_vocoder_layers,
            config.enc_dec_dim,
            config.vocoder_dim,
            config.vocoder_intermediate_dim,
            None,
            config.n_fft,
            config.hop_size
        )?;
        Ok(Self{
            text_embedding,
            encoder,
            decoder,
            vocoder
        })
    }
    fn forward(&self, src_tokens: &Tensor) -> Result<Tensor> {
        let x = self.text_embedding.forward(&src_tokens)?;
        let x = self.encoder.forward(&x)?;
        let x = self.decoder.forward(&x)?;
        let audio = self.vocoder.forward(&x)?;
        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use super::*;

    const INPUT_IDS: &[i64] = &[
        27, 88, 3, 34, 53, 48, 3, 45, 137, 116, 39, 3, 136, 65, 138, 42, 110, 73, 40, 52, 3, 32, 68, 138, 102, 3, 55, 53, 46, 3, 28, 88, 37, 136, 116, 52, 3, 27, 88, 30, 3, 38, 136, 27, 88, 37, 3, 46, 73, 3, 34, 53, 48, 3, 88, 106, 3, 137, 68, 40, 3, 64, 3, 38, 136, 116, 37, 35, 3, 30, 136, 31, 88, 3, 68, 40, 55, 73, 3, 37, 136, 53, 38, 73, 40, 30, 74, 12
    ];

    #[test]
    fn should_load_weights() -> Result<()> {
        let model = load_optispeech_cnx_model()?;
        Ok(())
    }
    fn should_forward_pass() -> Result<()> {
        let model = load_optispeech_cnx_model()?;
        let input_ids =
            Tensor::from_slice(INPUT_IDS, INPUT_IDS.len(), &Device::Cpu)?.unsqueeze(0)?;
        let audio = model.forward(&input_ids)?;
        Ok(())
    }
}
