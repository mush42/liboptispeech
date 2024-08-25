use anyhow::{self, Result};
use audio_ops::write_wave_samples_to_file;
use clap::Parser;
use optispeech::OptiSpeechCNXModel;
use serde::Deserialize;
use xxhash_rust::xxh3::xxh3_64;
use std::fs::File;
use std::io::{self, prelude::*};
use std::path::PathBuf;
use std::sync::OnceLock;


fn main() -> Result<()> {
    setup_logging();

    let args = Cli::parse();

    log::info!("Loading model from file: {}", args.model_path.display());
    let model = match OptiSpeechCNXModel::from_path(&args.model_path, None) {
        Ok(m) => m,
        Err(e) => {
            log::error!(
                "Unable to load voice model from file: {}.",
                args.model_path.display(),
            );
            return Err(e)
        }
    };

    if args.input_file.is_none() {
        log::info!("Accepting input from stdin since `--input-file` is not passed.");
    }

    if args.output_dir.is_none() {
        log::info!("Writing audio data to stdout since `--output-dir` is not passed");
    }

    let interactive = args.input_file.is_none();

    let mut model_input = get_input(&args)?;
    if interactive {
        loop {
            ospeak_main(&model, &args, &model_input)?;
            model_input = get_input(&args)?;
        }
    } else {
        log::info!("Reading input from file: {}", args.input_file.as_ref().unwrap().display());
        ospeak_main(&model, &args, &model_input)?;
    }

    Ok(())
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input file (default `stdin`)
    #[arg(value_name = "MODEL_PATH")]
    model_path: PathBuf,
    /// Input file (default `stdin`)
    #[arg(short = 'f', long, value_name = "INPUT_FILE")]
    input_file: Option<PathBuf>,
    /// Output directory (will use  `stdout` if not set)
    #[arg(short, long, value_name = "OUTPUT_DIR")]
    output_dir: Option<PathBuf>,
    /// Language code (for multilingual models)
    #[arg(short, long, value_name = "LANG")]
    lang: Option<String>,
    /// Speaker name (for multi-speaker models)
    #[arg(short, long, value_name = "SPEAKER")]
    speaker: Option<String>,
    /// Duration scaling factor (controls speech rate)
    #[arg(long, short, required = false)]
    d_factor: Option<f32>,
    /// Pitch scaling factor (controls inflection/expressiveness)
    #[arg(long, short, required = false)]
    p_factor: Option<f32>,
    /// Energy scaling factor (controls speech intensity/prominence)
    #[arg(long, short, required = false)]
    e_factor: Option<f32>,
}

#[derive(Deserialize, Default)]
struct ModelInput {
    phoneme_ids: Vec<i64>,
    language: Option<String>,
    speaker: Option<String>,
    d_factor: Option<f64>,
    p_factor: Option<f64>,
    e_factor: Option<f64>,
    filename_suffix: Option<String>
}

fn ospeak_main(
    model: &OptiSpeechCNXModel,
    args: &Cli,
    model_input: &ModelInput
) -> Result<()> {
    let phoneme_ids = &[model_input.phoneme_ids.as_slice()];
    let (input_ids, input_lengths) = model.prepare_input(phoneme_ids.as_slice())?;
    let synth_out = model.synthesise(
        &input_ids,
        &input_lengths,
        model_input.d_factor.or_else(|| args.d_factor.map(|v| v as f64)),
        model_input.p_factor.or_else(|| args.p_factor.map(|v| v as f64)),
        model_input.e_factor.or_else(|| args.e_factor.map(|v| v as f64)),
    )?;
    log::info!("RTF: {}", synth_out.rtf());
    log::info!("Latency: {}", synth_out.latency());

    let file_suffix = model_input.filename_suffix.clone().unwrap_or_default();
    for (idx, samples) in synth_out.iter_audio().enumerate() { 
        if let Some(ref directory) = args.output_dir {
            if !directory.exists() {
                std::fs::create_dir_all(directory)?;
            }
            let output_filename = format!("{}_{}.wav", idx, &file_suffix);
            let output_filename = directory.join(output_filename);
            write_wave_samples_to_file(
                &output_filename,
                samples.to_i16_vec().iter(),
                model.inference_config.sample_rate as u32,
                OptiSpeechCNXModel::NUM_CHANNELS as u32,
                OptiSpeechCNXModel::SAMPLE_WIDTH as u32,
            )?;
            log::info!("Wrote audio file to: {}", output_filename.display());
        } else {
            let audio_bytes = samples.as_wave_bytes();
            write_to_stdout(audio_bytes.as_slice())?;
        }
    }
    Ok(())
}

fn get_input(args: &Cli) -> anyhow::Result<ModelInput> {
    let mut input_buffer = String::new();
    if let Some(ref input_filename) = args.input_file {
        let mut file = File::open(input_filename)?;
        file.read_to_string(&mut input_buffer)?;
    } else {
        let stdin = io::stdin();
        stdin.read_line(&mut input_buffer)?;
    }

    let mut model_input: ModelInput =  match serde_json::from_str(&input_buffer){
        Ok(val) => val,
        Err(e) => {
            log::error!("Invalid JSON input. Please try again.");
            return get_input(args);
        }
    };
    model_input.filename_suffix = Some(xxh3_64(input_buffer.as_bytes()).to_string());
    Ok(model_input)
}

fn write_to_stdout(data: &[u8]) -> anyhow::Result<()> {
    let mut stdout = io::stdout().lock();
    stdout.write_all(data)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn setup_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().filter_or("OSPEAK_LOG", "info"))
        .init();
}
