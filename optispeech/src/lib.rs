use anyhow::{self, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn as nn;
use candle_nn::{Module, VarBuilder};
use audio_ops::AudioSamples;
use std::path::Path;

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const FLOAT_DTYPE: DType = DType::F32;
const LONG_DTYPE: DType = DType::I64;


pub enum Accelerator  {
    Cpu,
    WebGpu(usize)
}

impl Accelerator   {
    fn device(&self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::WebGpu(ordinal) => Ok(Device::new_wgpu_sync(*ordinal)?)
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptiSpeechCNXConfig {
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

fn pad_sequences<'a>(
    sequences: &'a [&[i64]],
    padding_value: Option<i64>,
) -> Result<(Tensor, Tensor)> {
    let lengths = Vec::from_iter(sequences.iter().map(|s| s.len() as i64));
    let max_length = match lengths.iter().max() {
        Some(val) => *val as usize,
        None => anyhow::bail!("Empty input to pad sequences"),
    };
    let padding_value = padding_value.unwrap_or(0);
    let padded_seqs = Vec::from_iter(
        sequences
            .iter()
            .map(|s| {
                let mut out = Vec::with_capacity(max_length);
                out.extend(s.iter().cloned());
                out.resize(max_length, padding_value);
                out
            })
            .flatten(),
    );
    let out = Tensor::from_vec(padded_seqs, (sequences.len(), max_length), &Device::Cpu)?;
    let lengths = Tensor::from_slice(&lengths, lengths.len(), &Device::Cpu)?;
    Ok((out, lengths))
}

fn unpad_sequences<'a>(x: &Tensor, lengths: &Tensor) -> Result<Vec<Vec<f32>>> {
    let lengths = lengths.to_dtype(LONG_DTYPE)?.to_vec1::<i64>()?;
    let mut vecs = x.to_vec2::<f32>()?;
    for (ref mut v, length) in vecs.iter_mut().zip(lengths.iter()) {
        // v.drain(*length as usize..);
    }
    Ok(vecs)
}

fn sequence_mask(lengths: &Tensor, max_length: Option<i64>) -> Result<Tensor> {
    let max_length = match max_length {
        Some(val) => val,
        None => lengths.max(0)?.to_scalar::<i64>()?,
    };
    let x = Tensor::arange(0i64, max_length, lengths.device())?.to_dtype(lengths.dtype())?;
    let mask = x
        .unsqueeze(0)?
        .broadcast_lt(&lengths.unsqueeze(1)?)?
        .to_dtype(LONG_DTYPE)?;
    Ok(mask)
}

fn layer_norm(dim: usize, eps: Option<f64>, vb: VarBuilder, device: Option<&Device>) -> Result<nn::LayerNorm> {
    let mut layer_config = nn::LayerNormConfig::default();
    if let Some(eps) = eps {
        layer_config.eps = eps;
    }
    let ws = vb.get(dim, "weight")?;
    let ws = if let Some(dev) = device {
        ws.to_device(dev)?
    } else {
        ws
    };
    if layer_config.affine {
        let bs = vb.get(dim, "bias")?;
        let bs = if let Some(dev) = device {
            bs.to_device(dev)?
        } else {
            bs
        };
        Ok(nn::LayerNorm::new(ws, bs, layer_config.eps))
    } else {
        Ok(nn::LayerNorm::new_no_bias(ws, layer_config.eps))
    }
}

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    padding: Option<usize>,
    dilation: Option<usize>,
    groups: Option<usize>,
    vb: VarBuilder,
    device: Option<&Device>
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
    let ws = vb.get(
        (out_channels, in_channels / layer_config.groups, kernel_size),
        "weight",
    )?;
    let bs = vb.get(out_channels, "bias")?;
    let (ws, bs) = if let Some(dev) = device {
        (
          ws.to_device(dev)?,
          bs.to_device(dev)?
        )
    } else {
      (ws, bs)
    };
    Ok(nn::Conv1d::new(ws, Some(bs), layer_config))
}


fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder, device: Option<&Device>) -> Result<nn::Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let bs = vb.get(out_dim, "bias")?;
    let (ws, bs) = if let Some(dev) = device {
        (
          ws.to_device(dev)?,
          bs.to_device(dev)?,
        )
    } else {
      (ws, bs)
    };
    Ok(nn::Linear::new(ws, Some(bs)))
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder, device: Option<&Device>) -> Result<nn::Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    let ws = if let Some(dev) = device {
      ws.to_device(dev)?
    } else {
        ws
    };
    Ok(nn::Linear::new(ws, None))
}

struct ScaledSinusoidalEmbedding {
    scale: f64,
    inv_freq: Tensor,
}

impl ScaledSinusoidalEmbedding {
    fn load(vb: &VarBuilder, dim: usize, theta: f32) -> Result<Self> {
        let scale = vb.get(&[1], "scale")?.squeeze(0)?.to_scalar::<f32>()? as f64;
        let half_dim = dim / 2;
        let freq_seq =
            Tensor::arange(0i64, half_dim as i64, &Device::Cpu)?.to_dtype(FLOAT_DTYPE)?;
        let freq_seq = (freq_seq / half_dim as f64)?;
        let inv_freq = Tensor::from_vec(vec![theta], (1,), &Device::Cpu)?
            .broadcast_pow(&(freq_seq * -1.0f64)?)?;
        Ok(Self { scale, inv_freq })
    }
}

impl nn::Module for ScaledSinusoidalEmbedding {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
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
        let (batch, __, _) = x.dims3()?;
        let t_feats = durations.sum(1)?.max(0)?.to_scalar::<f32>()?;
        let t = Tensor::arange(0i64, t_feats as i64, &device)?
            .unsqueeze(0)?
            .repeat((batch, 1))?
            .to_dtype(FLOAT_DTYPE)?;
        let c = (durations.cumsum(1)? - durations / 2.0f64)?;
        let energy = (Tensor::try_from(-1.0f64 * self.delta as f64)?
            .unsqueeze(0)?
            .to_dtype(FLOAT_DTYPE)?
            .broadcast_mul(
                &(t.unsqueeze(2)?
                    .broadcast_sub(&c.unsqueeze(1)?)?
                    .powf(2.0f64)?),
            ))?;
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
        let embed_positions =
            ScaledSinusoidalEmbedding::load(&vb.pp("embed_positions"), dim, max_source_positions)?;
        Ok(Self {
            embed_scale,
            embed_tokens,
            embed_positions,
        })
    }
}

impl nn::Module for TextEmbedding {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let src_tokens = x;
        let embeddings = self.embed_tokens.forward(&src_tokens)?;
        let embed = (embeddings * self.embed_scale)?;
        let positions = self.embed_positions.forward(&src_tokens)?;
        let x = embed.broadcast_add(&positions.unsqueeze(0)?)?;
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
    fn load(vb: &VarBuilder, dim: usize, intermediate_dim: usize, device: Option<&Device>) -> Result<Self> {
        let dwconv = conv1d(
            dim,
            dim,
            7,
            Some(3),
            None,
            Some(dim),
            vb.pp("dwconv"),
            device
        )?;
        let norm = layer_norm(dim, Some(1e-6), vb.pp("norm"), device)?;
        let gamma = vb.get(dim, "gamma")?;
        let gamma = if let Some(dev) = device {
            gamma.to_device(dev)?
        } else {
            gamma
        };
        Ok(Self {
            dwconv,
            norm,
            pwconv1: linear(dim, intermediate_dim, vb.pp("pwconv1"), device)?,
            pwconv2: linear(intermediate_dim, dim, vb.pp("pwconv2"), device)?,
            gamma
        })
    }
}

impl nn::Module for ConvNeXtLayer {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone().to_device(x.device())?;
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
        device: Option<&Device>
    ) -> Result<Self> {
        let mut convnexts = nn::seq();
        for i in 0..num_layers {
            let layer =
                ConvNeXtLayer::load(&vb.pp(format!("convnext.{i}")), dim, intermediate_dim, device)?;
            convnexts = convnexts.add(layer);
        }
        let final_layer_norm = layer_norm(dim, Some(1e-6), vb.pp("final_layer_norm"), device)?;
        Ok(Self {
            convnexts,
            final_layer_norm,
        })
    }
}

impl nn::Module for ConvNeXtBackbone {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
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
        device: Option<&Device>
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
                device
            )?;
            convs = convs
                .add(|x: &Tensor| x.transpose(1, 2))
                .add(conv_layer)
                .add(|x: &Tensor| x.relu())
                .add(|x: &Tensor| x.transpose(1, 2))
                .add(layer_norm(intermediate_dim, None, vb.pp("2"), device)?);
        }
        let linear = linear(intermediate_dim, 1, vb.pp("linear"), device)?;
        Ok(Self { convs, linear })
    }
}

impl nn::Module for VariancePredictor {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
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
        device: Option<&Device>
    ) -> Result<Self> {
        let var_predictor =
            VariancePredictor::load(&vb, dim, num_layers, intermediate_dim, kernel_size, device)?;
        Ok(Self {
            var_predictor,
            clip_val,
        })
    }
    fn infer(&self, x: &Tensor, mask: Option<&Tensor>, factor: Option<f64>) -> Result<Tensor> {
        let log_durations = self.var_predictor.forward(&x)?;
        // from log to linear domain
        let mut durations = (log_durations.exp()? - self.clip_val)?;
        if let Some(factor) = factor {
            durations = (durations * factor)?;
        }
        let durations = (durations * factor.unwrap_or(1.0f64))?;
        let durations = durations.ceil()?;
        let mut durations = durations.clamp(0.0, f64::INFINITY)?;
        if let Some(mask) = mask {
            durations = durations.broadcast_mul(mask)?;
        }
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
        device: Option<&Device>
    ) -> Result<Self> {
        let predictor = VariancePredictor::load(
            &vb.pp("predictor"),
            dim,
            num_layers,
            intermediate_dim,
            kernel_size,
            device
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
            device
        )?;
        Ok(Self { predictor, embed })
    }
    fn infer(&self, x: &Tensor, mask: Option<&Tensor>, factor: Option<f64>) -> Result<Tensor> {
        let mut preds = self.predictor.forward(&x)?;
        if let Some(factor) = factor {
            preds = (preds * factor)?
        }
        let emb = self.embed.forward(&preds.transpose(1, 2)?)?;
        let emb = emb.transpose(1, 2)?;
        let mut x = x.add(&emb)?;
        if let Some(mask) = mask {
            x = x.broadcast_mul(mask)?;
        }
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
        device: Option<&Device>,
    ) -> Result<Self> {
        let embed = conv1d(
            input_dim,
            dim,
            7,
            Some(3),
            None,
            None,
            vb.pp("embed"),
            device
          )?;
        let norm = layer_norm(dim, Some(1e-6), vb.pp("norm"), device)?;
        let backbone =
            ConvNeXtBackbone::load(&vb.pp("backbone"), num_layers, dim, intermediate_dim, device)?;
        // head
        let l_fft = n_fft + 2;
        let l_shift = hop_size;
        let linear_1 = linear(dim, l_fft, vb.pp("head.linear_1"), device)?;
        let linear_2 = linear_no_bias(l_fft, l_shift, vb.pp("head.linear_2"), device)?;
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
        let audio = x.reshape((b, c * t))?;
        Ok(audio)
    }
}

pub struct InferenceConfig {
    pub sample_rate: usize,
    pub d_factor: f64,
    pub p_factor: f64,
    pub e_factor: f64,
}

pub struct InferenceOutput {
    pub audio_samples: Vec<AudioSamples>,
    pub inference_ms: f64,
    pub sample_rate: usize,
}

impl InferenceOutput {
    pub fn iter_audio(&self) -> impl Iterator<Item=&AudioSamples> {
        self.audio_samples.iter()
    }
    pub fn latency(&self) -> f64 {
        self.inference_ms
    }
    pub fn rtf(&self) -> f64 {
        let num_samples: usize = self
            .audio_samples
            .iter()
            .map(|s| s.as_vec().len())
            .sum();
        let audio_ms = (num_samples as f64 / self.sample_rate as f64) * 1000.0;
        self.inference_ms / audio_ms
    }
}


pub struct OptiSpeechCNXModel {
    pub inference_config: InferenceConfig,
    pub device: Option<Device>,
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
    pub const NUM_CHANNELS: usize = 1;
    pub const SAMPLE_WIDTH: usize = 2;

    /// Load an OptiSpeech model built with the ConvNeXt backbone
    pub fn from_path<P: AsRef<Path>>(
        model_file_path: P,
        config: Option<OptiSpeechCNXConfig>,
        accelerator: Option<Accelerator>
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let vb_device = Device::Cpu;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file_path], FLOAT_DTYPE, &vb_device)? };
        let device = accelerator.map(|accel| accel.device().unwrap());
        let model = Self::load(config, vb, device)?;
        Ok(model)
    }
    fn load(config: OptiSpeechCNXConfig, vb: VarBuilder, dev: Option<Device>) -> Result<Self> {
        let device = dev.as_ref();
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
            device
        )?;
        let duration_predictor = DurationPredictor::load(
            &vb.pp("duration_predictor"),
            config.enc_dec_dim,
            config.dp_num_layers,
            config.dp_intermediate_dim,
            config.dp_kernel_size,
            config.dp_clip_val,
            None,
        )?;
        let pitch_predictor = PitchPredictor::load(
            &vb.pp("pitch_predictor"),
            config.enc_dec_dim,
            config.pp_num_layers,
            config.pp_intermediate_dim,
            config.pp_kernel_size,
            config.pp_embed_kernel_size,
            None
        )?;
        let energy_predictor = EnergyPredictor::load(
            &vb.pp("energy_predictor"),
            config.enc_dec_dim,
            config.ep_num_layers,
            config.ep_intermediate_dim,
            config.ep_kernel_size,
            config.ep_embed_kernel_size,
            None
        )?;
        let gaussian_upsampling = GaussianUpsampling::default();
        let decoder = ConvNeXtBackbone::load(
            &vb.pp("decoder"),
            config.num_dec_layers,
            config.enc_dec_dim,
            config.enc_dec_intermediate_dim,
            device
        )?;
        let vocoder = WaveNeXtVocoder::load(
            &vb.pp("wav_generator"),
            config.num_vocoder_layers,
            config.enc_dec_dim,
            config.vocoder_dim,
            config.vocoder_intermediate_dim,
            config.n_fft,
            config.hop_size,
            None
        )?;
        let inference_config = InferenceConfig {
            sample_rate: config.sample_rate,
            d_factor: config.d_factor,
            p_factor: config.p_factor,
            e_factor: config.e_factor,
        };
        Ok(Self {
            inference_config,
            device: dev,
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
    pub fn prepare_input(&self, input_ids: &[&[i64]]) -> Result<(Tensor, Tensor)> {
        Ok(pad_sequences(input_ids, None)?)
    }
    pub fn synthesise(
        &self,
        inputs: &Tensor,
        input_lengths: &Tensor,
        d_factor: Option<f64>,
        p_factor: Option<f64>,
        e_factor: Option<f64>,
    ) -> Result<InferenceOutput> {
        let cpu_device = Device::Cpu;
        let device = self.device.as_ref().unwrap_or(&cpu_device);
        let (d_factor, p_factor, e_factor) = (
            d_factor.or(Some(self.inference_config.d_factor)),
            p_factor.or(Some(self.inference_config.p_factor)),
            e_factor.or(Some(self.inference_config.e_factor)),
        );
        let timer = std::time::Instant::now();
        let (batch, max_length) = inputs.dims2()?;
        // Input masks
        let mask = if batch > 1 {
            let mask = sequence_mask(&input_lengths, None)?
                .unsqueeze(2)?
                .to_dtype(FLOAT_DTYPE)?
                .to_device(&cpu_device)?;
            Some(mask)
        } else {
            None
        };
        let device_mask = None; //mask.clone().map(|m| m.to_device(device).unwrap());
        let input_padding_mask = match mask {
            Some(ref m) => Some(m.ones_like()?.sub(m)?),
            None => None,
        };
        let token_emb = self.text_embedding.forward(&inputs)?;
        let token_emb = token_emb.to_device(&device)?;
        let enc_out = self.encoder.forward(&token_emb)?;
        let enc_out = enc_out.to_device(&cpu_device)?;
        let durations = self
            .duration_predictor
            .infer(&enc_out, mask.as_ref(), d_factor)?;
        let xp = self
            .pitch_predictor
            .infer(&enc_out, mask.as_ref(), p_factor)?;
        let mut xe = self.energy_predictor.infer(&xp, mask.as_ref(), e_factor)?;
        if let Some(ref mask) = device_mask {
            xe = xe.broadcast_mul(&mask)?;
        }
        let xe = xe.to_device(&cpu_device)?;
        let duration_sum = durations.sum(1)?;
        let target_mask = if batch > 1 {
            Some(
                sequence_mask(&duration_sum.to_dtype(LONG_DTYPE)?, None)?
                    .unsqueeze(2)?
                    .to_dtype(FLOAT_DTYPE)?,
            )
        } else {
            None
        };
        let mut upsampled = self
            .gaussian_upsampling
            .upsample_features(&xe, &durations)?;
        if let Some(ref target_mask) = target_mask {
            upsampled = upsampled.broadcast_mul(&target_mask)?;
        }
        let upsampled = upsampled.to_device(&device)?;
        let dec_out = self.decoder.forward(&upsampled)?;
        let dec_out = dec_out.to_device(&cpu_device)?;
        let audio = self.vocoder.forward(&dec_out)?;
        let audio = audio.to_device(&cpu_device)?;
        let audio = audio.clamp(-1.0, 1.0)?;
        let inference_ms = timer.elapsed().as_millis();
        let audio_samples: Vec<AudioSamples> = unpad_sequences(&audio, &duration_sum)?
            .into_iter()
            .map(|s| s.into())
            .collect();
        Ok(InferenceOutput {
            audio_samples,
            inference_ms: inference_ms as f64,
            sample_rate: self.inference_config.sample_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use audio_ops::{write_wave_samples_to_file, AudioSamples};
    use std::path::PathBuf;

    const MODEL_SAFETENSORS_FILENAME: &str = "../model/model.safetensors";
    const INPUT_IDS: &[&[i64]] = &[
        &[28, 27, 88, 3, 55, 73, 3, 40, 136, 31, 88, 39, 3, 116, 48, 3, 136, 53, 38, 73, 12],
        &[136, 35, 138, 48, 73, 40, 3, 55, 88, 3, 136, 65, 138, 102, 46, 45, 3, 36, 47, 138, 3, 37, 136, 68, 138, 38, 3, 45, 136, 41, 138, 102, 45, 74, 102, 35, 3, 65, 138, 102, 3, 80, 136, 116, 48, 74, 40, 30, 3, 28, 27, 88, 3, 37, 136, 65, 138, 52, 39, 88, 37, 3, 38, 136, 68, 138, 52, 10, 136, 53, 40, 45, 74, 30, 3, 55, 73, 3, 39, 136, 53, 40, 3, 88, 40, 55, 73, 3, 80, 102, 136, 35, 138, 40, 3, 46, 136, 76, 138, 28, 73, 40, 12],
    ];

    #[test]
    fn should_load_weights() -> Result<()> {
        let accelerator = Accelerator::WebGpu(0);
        let model = OptiSpeechCNXModel::from_path(MODEL_SAFETENSORS_FILENAME, None, Some(accelerator))?;
        Ok(())
    }
    #[test]
    fn test_forward_pass() -> Result<()> {
        let config = OptiSpeechCNXConfig::default();
        let accelerator = Accelerator::WebGpu(0);
        let model = OptiSpeechCNXModel::from_path(
            MODEL_SAFETENSORS_FILENAME,
            Some(config.clone()),
            Some(accelerator)
        )?;
        let (inputs, input_lengths) = pad_sequences(INPUT_IDS, Some(0))?;
        let synth_out = model
            .synthesise(&inputs, &input_lengths, None, None, None)
            .unwrap();
        // dbg!("Infer: {inference_ms}, audio: {audio_ms}, rtf: {rtf}");
        for (idx, samples) in synth_out.iter_audio().enumerate() {
            let filename = format!("output_{idx}.wav");
            write_wave_samples_to_file(
                &PathBuf::from(filename),
                samples.to_i16_vec().iter(),
                config.sample_rate as u32,
                config.num_channels as u32,
                config.sample_width as u32,
            )?;
        }
        Ok(())
    }
}
