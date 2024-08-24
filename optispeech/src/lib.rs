use anyhow::{self, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn as nn;
use candle_nn::{Module, VarBuilder};
use std::path::Path;


const FLOAT_DTYPE: DType = DType::F32;
const LONG_DTYPE: DType = DType::I64;


/// Load an OptiSpeech model built with the ConvNeXt backbone
fn load_optispeech_cnx_model<P: AsRef<Path>>(
    config: Option<OptiSpeechCNXConfig>,
    model_file_path: P,
) -> Result<OptiSpeechCNXModel> {
    let config = config.unwrap_or_default();
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_file_path], FLOAT_DTYPE, &device)?
    };
    let model = OptiSpeechCNXModel::load(config, vb)?;
    Ok(model)
}

#[derive(Debug, Clone)]
struct OptiSpeechCNXConfig {
    sample_rate: usize,
    num_channels: usize,
    sample_width: usize,
    n_vocab: usize,
    max_source_positions: f32,
    num_enc_layers: usize,
    num_dec_layers: usize,
    enc_dec_dim: usize,
    enc_dec_intermediate_dim: usize,
    dp_num_layers: usize,
    dp_intermediate_dim: usize,
    dp_kernel_size: usize,
    dp_clip_val: f64,
    pp_num_layers: usize,
    pp_intermediate_dim: usize,
    pp_kernel_size: usize,
    pp_embed_kernel_size: usize,
    ep_num_layers: usize,
    ep_intermediate_dim: usize,
    ep_kernel_size: usize,
    ep_embed_kernel_size: usize,
    num_vocoder_layers: usize,
    vocoder_dim: usize,
    vocoder_intermediate_dim: usize,
    n_fft: usize,
    hop_size: usize,
    d_factor: f64,
    p_factor: f64,
    e_factor: f64,
}

impl Default for OptiSpeechCNXConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            num_channels: 1,
            sample_width: 2,
            n_vocab: 250,
            max_source_positions: 2000.0,
            num_enc_layers: 4,
            num_dec_layers: 4,
            enc_dec_dim: 256,
            enc_dec_intermediate_dim: 1024,
            dp_num_layers: 2,
            dp_intermediate_dim: 384,
            dp_kernel_size: 3,
            dp_clip_val: 1e-8,
            pp_num_layers: 5,
            pp_intermediate_dim: 256,
            pp_kernel_size: 5,
            pp_embed_kernel_size: 9,
            ep_num_layers: 2,
            ep_intermediate_dim: 384,
            ep_kernel_size: 3,
            ep_embed_kernel_size: 9,
            num_vocoder_layers: 8,
            vocoder_dim: 384,
            vocoder_intermediate_dim: 1280,
            n_fft: 2048,
            hop_size: 300,
            d_factor: 1.1,
            p_factor: 1.6,
            e_factor: 1.3,
        }
    }
}


fn layer_norm(dim: usize, eps: Option<f64>, vb: VarBuilder) -> Result<nn::LayerNorm> {
    let mut layer_config = nn::LayerNormConfig::default();
    if let Some(eps) = eps {
        layer_config.eps = eps;
    }
    Ok(nn::layer_norm(dim, layer_config, vb)?)
}

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: Option<usize>,
    dilation: Option<usize>,
    groups: Option<usize>,
    vb: VarBuilder,
) -> Result<nn::Conv1d> {
    let mut layer_config = nn::conv::Conv1dConfig::default();
    if let Some(padding) = padding {
        layer_config.padding = padding;
    }
    if let Some(groups) = groups {
        layer_config.groups = groups;
    }
    if let Some(dilation) = dilation {
        layer_config.dilation = dilation;
    }
    Ok(nn::conv1d(
        in_channels,
        out_channels,
        kernel_size,
        layer_config,
        vb
    )?)
}

struct ScaledSinusoidalEmbedding {
    scale: f64,
    inv_freq: Tensor,
}

impl ScaledSinusoidalEmbedding {

    fn load(vb: &VarBuilder, dim: usize, theta: f32) -> Result<Self> {
        let scale = vb.get(&[1], "scale")?.squeeze(0)?.to_scalar::<f32>()? as f64;
        let half_dim = dim / 2;
        let freq_seq = Tensor::arange(0i64, half_dim as i64, &Device::Cpu)?.to_dtype(FLOAT_DTYPE)?;
        let freq_seq = (freq_seq / half_dim as f64)?;
        let inv_freq = Tensor::from_vec(vec![theta], (1,), &Device::Cpu)?
            .broadcast_pow(&(freq_seq * -1.0f64)?)?;
        Ok(Self { scale, inv_freq })
    }
}

impl nn::Module for ScaledSinusoidalEmbedding {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor, > {
        let (_batch, seq_length) = x.dims2()?;
        let pos = Tensor::arange(0i64, seq_length as i64, x.device())?.to_dtype(FLOAT_DTYPE)?;
        let mut emb = pos.unsqueeze(1)?.broadcast_mul(&self.inv_freq)?;
        emb = Tensor::cat(&[emb.sin()?, emb.cos()?], 1)?;
        Ok((emb * self.scale)?)
    }
}

struct GaussianUpsampling {
    delta: f32,
}

impl Default for GaussianUpsampling {
    fn default() -> Self {
        Self { delta: 0.1 }
    }
}

impl GaussianUpsampling {
    fn upsample_features(&self, x: &Tensor, durations: &Tensor) -> Result<Tensor> {
        let device = x.device();
        let (B, __, _) = x.dims3()?;
        let t_feats = durations.sum(1)?.squeeze(0)?.to_scalar::<f32>()?;
        let t = Tensor::arange(0i64, t_feats as i64, &device)?
            .unsqueeze(0)?
            .repeat((B, 1))?
            .to_dtype(FLOAT_DTYPE)?;
        let c = (durations.cumsum(1)? - durations / 2.0f64)?;
        let energy = (
            Tensor::try_from(-1.0f64 * self.delta as f64)?
            .unsqueeze(0)?
            .to_dtype(FLOAT_DTYPE)?
            .broadcast_mul(
                &(t.unsqueeze(2)?.broadcast_sub(&c.unsqueeze(1)?)?.powf(2.0f64)?)
            )
        )?;
        let p_attn = nn::ops::softmax(&energy, 2)?;
        let x = p_attn.matmul(&x)?;
        Ok(x)
    }
}

struct TextEmbedding {
    embed_scale: f64,
    embed_tokens: nn::Embedding,
    embed_positions: ScaledSinusoidalEmbedding,
}

impl TextEmbedding {
    fn load(
        vb: &VarBuilder,
        n_vocab: usize,
        dim: usize,
        max_source_positions: f32,
    ) -> Result<Self> {
        let embed_scale = (dim as f64).sqrt();
        let embed_tokens = nn::embedding(n_vocab, dim, vb.pp("embed_tokens"))?;
        let embed_positions = ScaledSinusoidalEmbedding::load(&vb.pp("embed_positions"), dim, max_source_positions)?;
        Ok(Self {
            embed_scale,
            embed_tokens,
            embed_positions,
        })
    }
}

impl nn::Module for TextEmbedding {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor, > {
        let src_tokens = x;
        let embeddings = self.embed_tokens.forward(&src_tokens)?;
        let embed = (embeddings * self.embed_scale)?;
        let positions = self.embed_positions.forward(&src_tokens)?;
        let x = embed.add(&positions.unsqueeze(0)?)?;
        Ok(x)
    }
}

struct ConvNeXtLayer {
    dwconv: nn::Conv1d,
    norm: nn::LayerNorm,
    pwconv1: nn::Linear,
    pwconv2: nn::Linear,
    gamma: Tensor,
}

impl ConvNeXtLayer {
    fn load(vb: &VarBuilder, dim: usize, intermediate_dim: usize) -> Result<Self> {
        let dwconv = conv1d(
            dim, dim, 7, Some(3), None, Some(dim), vb.pp("dwconv")
        )?;
        let norm = layer_norm(dim, Some(1e-6), vb.pp("norm"))?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1: nn::linear(dim, intermediate_dim, vb.pp("pwconv1"))?,
            pwconv2: nn::linear(intermediate_dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
        })
    }
}

impl nn::Module for ConvNeXtLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor, > {
        let residual = x.clone();
        let x = self.dwconv.forward(&x)?;
        let x = x.transpose(1, 2)?; // (B, C, T) -> (B, T, C)
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.pwconv2.forward(&x)?;
        let x = self.gamma.unsqueeze(0)?.broadcast_mul(&x)?;
        let x = x.transpose(1, 2)?;
        let x = residual.add(&x)?;
        Ok(x)
    }
}

struct ConvNeXtBackbone {
    convnexts: nn::Sequential,
    final_layer_norm: nn::LayerNorm,
}

impl ConvNeXtBackbone {
    fn load(
        vb: &VarBuilder,
        num_layers: usize,
        dim: usize,
        intermediate_dim: usize,
    ) -> Result<Self> {
        let mut convnexts = nn::seq();
        for i in 0..num_layers {
            let layer =
                ConvNeXtLayer::load(&vb.pp(format!("convnext.{i}")), dim, intermediate_dim)?;
            convnexts = convnexts.add(layer);
        }
        let final_layer_norm = layer_norm(
            dim,
            Some(1e-6),
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self { convnexts, final_layer_norm, })
    }
}

impl nn::Module for ConvNeXtBackbone {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor, > {
        let x = x.transpose(1, 2)?;
        let x = self.convnexts.forward(&x)?;
        let x = x.transpose(1, 2)?;
        let x = self.final_layer_norm.forward(&x)?;
        Ok(x)
    }
}

struct VariancePredictor {
    convs: nn::Sequential,
    linear: nn::Linear,
}

impl VariancePredictor {
    fn load(
        vb: &VarBuilder,
        dim: usize,
        num_layers: usize,
        intermediate_dim: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let padding = (kernel_size - 1) / 2;
        let mut convs = nn::seq();
        for i in 0..num_layers {
            let vb = vb.pp(format!("conv.{i}"));
            let input_dim = if i == 0 { dim } else { intermediate_dim };
            let conv_layer = conv1d(
                input_dim,
                intermediate_dim,
                kernel_size,
                Some(padding),
                None,
                None,
                vb.pp("0"),
            )?;
            convs = convs
                .add(|x: &Tensor| x.transpose(1, 2))
                .add(conv_layer)
                .add(|x: &Tensor| x.relu())
                .add(|x: &Tensor| x.transpose(1, 2))
                .add(layer_norm(intermediate_dim, None, vb.pp("2"))?);
        }
        let linear = nn::linear(intermediate_dim, 1, vb.pp("linear"))?;
        Ok(Self { convs, linear })
    }
}

impl nn::Module for VariancePredictor {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor, > {
        let x = self.convs.forward(&x)?;
        let x = self.linear.forward(&x)?;
        Ok(x)
    }
}

struct DurationPredictor {
    var_predictor: VariancePredictor,
    clip_val: f64,
}

impl DurationPredictor {
    fn load(
        vb: &VarBuilder,
        dim: usize,
        num_layers: usize,
        intermediate_dim: usize,
        kernel_size: usize,
        clip_val: f64,
    ) -> Result<Self> {
        let var_predictor =
            VariancePredictor::load(&vb, dim, num_layers, intermediate_dim, kernel_size)?;
        Ok(Self {
            var_predictor,
            clip_val,
        })
    }
    fn infer(&self, x: &Tensor, factor: Option<f64>) -> Result<Tensor> {
        let log_durations = self.var_predictor.forward(&x)?;
        // from log to linear domain
        let durations = (log_durations.exp()? - self.clip_val)?;
        let durations = (durations * factor.unwrap_or(1.0f64))?;
        let durations = durations.ceil()?;
        let durations = durations.clamp(0.0, f64::INFINITY)?;
        let durations = durations.squeeze(2)?;
        Ok(durations)
    }
}

struct PitchPredictor {
    predictor: VariancePredictor,
    embed: nn::Conv1d,
}

impl PitchPredictor {
    fn load(
        vb: &VarBuilder,
        dim: usize,
        num_layers: usize,
        intermediate_dim: usize,
        kernel_size: usize,
        embed_kernel_size: usize,
    ) -> Result<Self> {
        let predictor = VariancePredictor::load(
            &vb.pp("predictor"),
            dim,
            num_layers,
            intermediate_dim,
            kernel_size,
        )?;
        let padding = (embed_kernel_size - 1) / 2;
        let embed = conv1d(
            1,
            dim,
            embed_kernel_size,
            Some(padding),
            None,
            None, 
            vb.pp("embed.0"),
        )?;
        Ok(Self { predictor, embed })
    }
    fn infer(&self, x: &Tensor, factor: Option<f64>) -> Result<Tensor> {
        let preds = self.predictor.forward(&x)?;
        let preds = (preds * factor.unwrap_or(1.0f64))?;
        let emb = self.embed.forward(&preds.transpose(1, 2)?)?;
        let emb = emb.transpose(1, 2)?;
        let x = x.add(&emb)?;
        Ok(x)
    }
}

// No difference
type EnergyPredictor = PitchPredictor;

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
        n_fft: usize,
        hop_size: usize,
    ) -> Result<Self> {
        let embed = conv1d(
            input_dim,
            dim,
            7,
            Some(3),
            None,
            None,
            vb.pp("embed"),
        )?;
        let norm = layer_norm(dim, Some(1e-6), vb.pp("norm"),)?;
        let backbone = ConvNeXtBackbone::load(
            &vb.pp("backbone"),
            num_layers,
            dim,
            intermediate_dim,
        )?;
        // head
        let l_fft = n_fft + 2;
        let l_shift = hop_size;
        let linear_1 = nn::linear(dim, l_fft, vb.pp("head.linear_1"))?;
        let linear_2 = nn::linear_no_bias(l_fft, l_shift, vb.pp("head.linear_2"))?;
        Ok(Self {
            embed,
            norm,
            backbone,
            linear_1,
            linear_2,
        })
    }
}

impl nn::Module for WaveNeXtVocoder {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.embed.forward(&x.transpose(1, 2)?)?;
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self.backbone.forward(&x)?; // masking
        // Head
        let (b, _, _) = x.dims3()?;
        let x = self.linear_1.forward(&x)?;
        let x = self.linear_2.forward(&x)?;
        let (_, c, t) = x.dims3()?;
        let audio = x.reshape((b,  c * t))?;
        let audio = audio.clamp(-1.0, 1.0)?;
        Ok(audio)
    }
}

struct InferenceConfig {
    d_factor: f64,
    p_factor: f64,
    e_factor: f64,
}

struct OptiSpeechCNXModel {
    inference_config: InferenceConfig,
    text_embedding: TextEmbedding,
    encoder: ConvNeXtBackbone,
    duration_predictor: DurationPredictor,
    pitch_predictor: PitchPredictor,
    energy_predictor: EnergyPredictor,
    gaussian_upsampling: GaussianUpsampling,
    decoder: ConvNeXtBackbone,
    vocoder: WaveNeXtVocoder,
}

impl OptiSpeechCNXModel {
    fn load(config: OptiSpeechCNXConfig, vb: VarBuilder) -> Result<Self> {
        let text_embedding = TextEmbedding::load(
            &vb.pp("text_embedding"),
            config.n_vocab,
            config.enc_dec_dim,
            config.max_source_positions,
        )?;
        let encoder = ConvNeXtBackbone::load(
            &vb.pp("encoder"),
            config.num_enc_layers,
            config.enc_dec_dim,
            config.enc_dec_intermediate_dim,
        )?;
        let duration_predictor = DurationPredictor::load(
            &vb.pp("duration_predictor"),
            config.enc_dec_dim,
            config.dp_num_layers,
            config.dp_intermediate_dim,
            config.dp_kernel_size,
            config.dp_clip_val,
        )?;
        let pitch_predictor = PitchPredictor::load(
            &vb.pp("pitch_predictor"),
            config.enc_dec_dim,
            config.pp_num_layers,
            config.pp_intermediate_dim,
            config.pp_kernel_size,
            config.pp_embed_kernel_size,
        )?;
        let energy_predictor = EnergyPredictor::load(
            &vb.pp("energy_predictor"),
            config.enc_dec_dim,
            config.ep_num_layers,
            config.ep_intermediate_dim,
            config.ep_kernel_size,
            config.ep_embed_kernel_size,
        )?;
        let gaussian_upsampling = GaussianUpsampling::default();
        let decoder = ConvNeXtBackbone::load(
            &vb.pp("decoder"),
            config.num_dec_layers,
            config.enc_dec_dim,
            config.enc_dec_intermediate_dim,
        )?;
        let vocoder = WaveNeXtVocoder::load(
            &vb.pp("wav_generator"),
            config.num_vocoder_layers,
            config.enc_dec_dim,
            config.vocoder_dim,
            config.vocoder_intermediate_dim,
            config.n_fft,
            config.hop_size,
        )?;
        let inference_config = InferenceConfig {
            d_factor: config.d_factor,
            p_factor: config.p_factor,
            e_factor: config.e_factor,
        };
        Ok(Self {
            inference_config,
            text_embedding,
            encoder,
            duration_predictor,
            pitch_predictor,
            energy_predictor,
            gaussian_upsampling,
            decoder,
            vocoder,
        })
    }
    fn synthesise(&self,
        src_tokens: &Tensor,
        d_factor: Option<f64>,
        p_factor: Option<f64>,
        e_factor: Option<f64>,
    ) -> Result<Tensor> {
        let (d_factor, p_factor, e_factor) = (
            d_factor.or(Some(self.inference_config.d_factor)),
            p_factor.or(Some(self.inference_config.p_factor)),
            e_factor.or(Some(self.inference_config.e_factor)),
        );
        let token_emb = self.text_embedding.forward(&src_tokens)?;
        let enc_out = self.encoder.forward(&token_emb)?;
        let durations = self.duration_predictor.infer(&enc_out, d_factor)?;
        let xp = self.pitch_predictor.infer(&enc_out, p_factor)?;
        let xe = self.energy_predictor.infer(&xp, e_factor)?;
        let upsampled = self.gaussian_upsampling.upsample_features(&xe, &durations)?;
        let dec_out = self.decoder.forward(&upsampled)?;
        let audio = self.vocoder.forward(&dec_out)?;
        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use audio_ops::{AudioSamples, write_wave_samples_to_file};
    use std::path::PathBuf;

    const MODEL_SAFETENSORS_FILENAME: &str = "./model/model.safetensors";
    const INPUT_IDS: &[i64] = &[28, 27, 88, 3, 55, 73, 3, 40, 136, 31, 88, 39, 3, 116, 48, 3, 136, 53, 38, 73, 12];

    #[test]
    fn should_load_weights() -> Result<()> {
        let model = load_optispeech_cnx_model(None, MODEL_SAFETENSORS_FILENAME)?;
        Ok(())
    }
    #[test]
    fn test_forward_pass() -> Result<()> {
        let config = OptiSpeechCNXConfig::default();
        let model = load_optispeech_cnx_model(
            Some(config.clone()),
            &MODEL_SAFETENSORS_FILENAME
        )?;
        let input_ids =
            Tensor::from_slice(INPUT_IDS, INPUT_IDS.len(), &Device::Cpu)?.unsqueeze(0)?;
        let audio = model.synthesise(&input_ids, None, None, None).unwrap();
        let audio_data: Vec<f32> = audio.squeeze(0)?.to_vec1()?;
        let samples: AudioSamples = audio_data.into();
        write_wave_samples_to_file(
            &PathBuf::from("output.wav"),
            samples.to_i16_vec().iter(),
            config.sample_rate as u32,
            config.num_channels as u32,
            config.sample_width as u32,
        )?;
        Ok(())
    }
}
