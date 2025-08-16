import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import pickle
from dataclasses import dataclass
from audiocraft.models.musicgen import MusicGen
from audiocraft.data.audio import audio_write
import logging
import torchaudio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SectionConfig:
    """Configuration for a single section of a layer"""

    name: str
    start: float
    duration: float
    prompt: str
    volume: float = 1.0
    loop: bool = False
    fade_out: bool = False
    fade_in: bool = False
    use_continuation: bool = False


class MusicProducer:
    def __init__(self, config_path: str):
        """Initialize the music producer with a YAML configuration file"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.song_config = self.config["song"]
        self.sections = self.config["sections"]
        self.layers = self.config["layers"]
        self.processing = self.config.get("processing", {})
        self.advanced = self.config.get("advanced", {})

        self.sample_rate = self.song_config.get("sample_rate", 32000)
        self.models_cache = {}
        self.generated_layers = {}
        self.style_audio = None

        if self.advanced.get("use_cache", False):
            self.cache_dir = Path(self.advanced.get("cache_dir", "cache/"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, model_name: str) -> MusicGen:
        """Get or load a MusicGen model, applying generation parameters."""
        if model_name not in self.models_cache:
            logger.info(f"Loading model: {model_name}")
            model = MusicGen.get_pretrained(model_name)
            self.models_cache[model_name] = model

        # Always set params before generation, as they might be shared
        model = self.models_cache[model_name]
        if self.advanced.get("generation_params"):
            params = self.advanced["generation_params"]
            model.set_generation_params(
                use_sampling=params.get("use_sampling", True),
                temperature=params.get("temperature", 1.0),
                top_k=params.get("top_k", 250),
                top_p=params.get("top_p", 0.0),
                cfg_coef=params.get("cfg_coef", 3.0),
            )
        return model

    def get_cache_key(
        self,
        layer_name: str,
        section_name: str,
        prompt: str,
        use_style: bool,
        use_continuation: bool,
    ) -> str:
        """Generate a unique cache key."""
        key_str = f"{layer_name}_{section_name}_{prompt}_{use_style}_{use_continuation}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load a generated section from cache."""
        if not self.advanced.get("use_cache", False):
            return None
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_key}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def save_to_cache(self, cache_key: str, audio: torch.Tensor):
        """Save a generated section to cache."""
        if not self.advanced.get("use_cache", False):
            return
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(audio, f)

    def load_style_reference(self):
        """Load and prepare the style reference audio."""
        style_config = self.advanced.get("style_reference")
        if not style_config or not style_config.get("path"):
            return

        style_path = Path(style_config["path"])
        if not style_path.exists():
            logger.error(f"Style reference audio not found at: {style_path}")
            return

        logger.info(f"Loading style reference from: {style_path}")
        audio, sr = torchaudio.load(style_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # MusicGen-Style expects stereo, convert if mono
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        self.style_audio = audio

    def generate_section(
        self,
        model: MusicGen,
        prompt: str,
        duration: float,
        continuation_audio: Optional[torch.Tensor] = None,
        style_audio: Optional[torch.Tensor] = None,
        use_style: bool = False,
    ) -> torch.Tensor:
        """Generate a single section of audio with best practices."""
        actual_duration = min(duration, 30.0)
        model.set_generation_params(duration=actual_duration)
        wav = None

        try:
            if use_style and style_audio is not None and "style" in model.name:
                logger.info(f"Generating with style conditioning: '{prompt}'")
                style_config = self.advanced.get("style_reference", {})
                model.set_style_conditioner_params(
                    excerpt_length=style_config.get("excerpt_length", 3.0)
                )
                wav = model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=style_audio,
                    melody_sample_rate=self.sample_rate,
                )

            elif continuation_audio is not None:
                # Use continuation only for extending a track in time.
                # MusicGen needs the prompt audio to be shorter than the generation duration.
                prompt_duration = continuation_audio.shape[-1] / self.sample_rate
                if prompt_duration >= actual_duration:
                    logger.warning(
                        f"Continuation audio ({prompt_duration:.1f}s) is longer than target duration ({actual_duration:.1f}s). Generating from text instead."
                    )
                    wav = model.generate([prompt])
                else:
                    logger.info(f"Generating with temporal continuation: '{prompt}'")
                    wav = model.generate_continuation(
                        continuation_audio, self.sample_rate, [prompt], progress=True
                    )

            else:
                logger.info(f"Generating from text: '{prompt}'")
                wav = model.generate([prompt])

        except Exception as e:
            logger.error(f"Model generation failed for prompt '{prompt}': {e}")
            # Return silence on failure
            return torch.zeros(1, int(actual_duration * self.sample_rate), device=model.device)

        return (
            wav[0]
            if wav is not None
            else torch.zeros(1, int(actual_duration * self.sample_rate), device=model.device)
        )

    def apply_fade(
        self,
        audio: torch.Tensor,
        fade_in: bool,
        fade_out: bool,
        fade_duration: float = 2.0,
    ) -> torch.Tensor:
        """Apply a cosine fade in/out to audio."""
        length = audio.shape[-1]
        fade_samples = min(int(fade_duration * self.sample_rate), length // 2)
        if fade_samples == 0:
            return audio

        if fade_in:
            fade_curve = torch.cos(torch.linspace(1.57, 0, fade_samples, device=audio.device))
            audio[..., :fade_samples] *= fade_curve
        if fade_out:
            fade_curve = torch.cos(torch.linspace(0, 1.57, fade_samples, device=audio.device))
            audio[..., -fade_samples:] *= fade_curve
        return audio

    def loop_with_crossfade(
        self, audio: torch.Tensor, target_samples: int, crossfade_duration: float
    ) -> torch.Tensor:
        """Loop audio with a beat-aligned crossfade to fill target duration."""
        source_samples = audio.shape[-1]
        if source_samples >= target_samples:
            return audio[..., :target_samples]
        if source_samples == 0:
            return torch.zeros(1, target_samples, device=audio.device)

        bpm = self.song_config.get("bpm", 128)
        beat_samples = int((60 / bpm) * self.sample_rate)
        crossfade_samples = min(
            int(crossfade_duration * self.sample_rate),
            source_samples // 4,
            beat_samples,
        )

        output = torch.zeros(1, target_samples, device=audio.device)
        output[..., :source_samples] = audio

        pos = source_samples - crossfade_samples
        while pos < target_samples:
            remaining_len = source_samples
            fade_len = min(crossfade_samples, target_samples - pos)

            fade_out = torch.cos(torch.linspace(0, 1.57, fade_len, device=audio.device))
            fade_in = torch.cos(torch.linspace(1.57, 0, fade_len, device=audio.device))

            output[..., pos : pos + fade_len] *= fade_out
            output[..., pos : pos + fade_len] += audio[..., :fade_len] * fade_in

            fill_len = min(source_samples - fade_len, target_samples - pos - fade_len)
            if fill_len > 0:
                output[..., pos + fade_len : pos + fade_len + fill_len] = audio[
                    ..., fade_len : fade_len + fill_len
                ]

            pos += source_samples - crossfade_samples

        return output

    def generate_layer(self, layer_config: Dict) -> torch.Tensor:
        """Generate all sections for a single layer with overlap-based transitions."""
        layer_name = layer_config["name"]
        model_name = layer_config.get("model", "facebook/musicgen-small")
        model = self.get_model(model_name)

        use_style = layer_config.get("use_style", False)
        if use_style and "style" not in model.name:
            logger.warning(
                f"Layer '{layer_name}' has use_style=true but is not using a 'style' model. Style conditioning will be ignored."
            )
            use_style = False

        total_duration = max(s["start"] + s["duration"] for s in self.sections.values())
        total_samples = int(total_duration * self.sample_rate)
        layer_audio = torch.zeros(1, total_samples, device=model.device)

        last_section_audio = None
        style_reference = None
        
        # Sort sections by start time to ensure proper continuation
        sorted_sections = sorted(
            layer_config.get("sections", []),
            key=lambda x: self.sections[x["section"]]["start"]
        )
        
        # Generate initial style reference if using style model
        if use_style and sorted_sections:
            first_section = sorted_sections[0]
            style_prompt = layer_config.get("style_prompt", first_section.get("prompt", ""))
            logger.info(f"Generating style reference for layer '{layer_name}'")
            style_reference = self.generate_section(
                model, style_prompt, 8.0, None, self.style_audio, False
            )

        for idx, section_config in enumerate(sorted_sections):
            section_name = section_config["section"]
            section_info = self.sections[section_name]

            prompt = section_config["prompt"]
            duration = section_info["duration"]
            start_time = section_info["start"]
            volume = section_config.get("volume", 1.0)

            # Overlap-based generation for natural transitions
            continuation_audio = None
            transitions = self.advanced.get("transitions", {})
            overlap_duration = transitions.get("overlap_duration", 4.0)  # Generate extra bars for overlap
            enable_continuation = section_config.get("use_continuation", idx > 0)
            
            # Generate with overlap if not the first section
            actual_duration = duration
            if idx > 0 and enable_continuation:
                actual_duration = duration + overlap_duration
                logger.info(f"Generating {actual_duration:.1f}s (includes {overlap_duration:.1f}s overlap) for section '{section_name}'")
            
            if enable_continuation and last_section_audio is not None:
                # Use full musical context from previous section
                continuation_window = min(10.0, last_section_audio.shape[-1] / self.sample_rate)
                continuation_samples = int(continuation_window * self.sample_rate)
                
                if last_section_audio.shape[-1] > continuation_samples:
                    continuation_audio = last_section_audio[..., -continuation_samples:]
                    logger.info(f"Using last {continuation_samples/self.sample_rate:.1f}s for continuation")
                else:
                    continuation_audio = last_section_audio
                    logger.info(f"Using full previous section for continuation")
            
            cache_key = self.get_cache_key(
                layer_name, section_name, prompt, use_style, continuation_audio is not None
            )
            section_audio = self.load_from_cache(cache_key)

            if section_audio is None:
                section_audio = self.generate_section(
                    model,
                    prompt,
                    actual_duration,  # Use extended duration for overlap
                    continuation_audio,
                    style_reference if use_style else self.style_audio,  # Use consistent style reference
                    use_style,
                )
                self.save_to_cache(cache_key, section_audio)

            section_audio = section_audio.to(model.device)

            if section_config.get("loop"):
                required_samples = int(duration * self.sample_rate)
                crossfade_duration = section_config.get("crossfade_duration", 0.25)
                section_audio = self.loop_with_crossfade(
                    section_audio, required_samples, crossfade_duration
                )

            section_audio = self.apply_fade(
                section_audio,
                section_config.get("fade_in", False),
                section_config.get("fade_out", False),
                section_config.get("fade_duration", 2.0),
            )

            section_audio *= volume
            
            # Trim overlap from generated audio to get the actual section
            if idx > 0 and enable_continuation and section_audio.shape[-1] > duration * self.sample_rate:
                # Keep the overlap portion at the beginning (it's the natural transition)
                # Trim from the end to match target duration
                target_samples = int(duration * self.sample_rate)
                section_audio = section_audio[..., :target_samples]
                logger.info(f"Trimmed section to {target_samples/self.sample_rate:.1f}s after overlap generation")
            
            last_section_audio = section_audio.clone()
            start_sample = int(start_time * self.sample_rate)
            if start_sample >= total_samples:
                continue

            length = min(section_audio.shape[-1], total_samples - start_sample)
            layer_audio[..., start_sample : start_sample + length] += section_audio[
                ..., :length
            ]

        return layer_audio

    def produce(self, export_stems: bool = False):
        """Main production process."""
        logger.info(f"Starting production of: {self.song_config['name']}")

        self.load_style_reference()

        for section_name, section_info in self.sections.items():
            if section_info["duration"] > 30:
                logger.warning(
                    f"Section '{section_name}' is {section_info['duration']}s long. Generations are capped at 30s and may not fill the section unless looped."
                )

        layer_audios = []
        for i, layer_config in enumerate(self.layers):
            logger.info(
                f"Processing layer {i+1}/{len(self.layers)}: {layer_config['name']}"
            )
            layer_audio = self.generate_layer(layer_config)
            self.generated_layers[layer_config["name"]] = layer_audio.cpu()
            layer_volume = layer_config.get("volume", 1.0)
            layer_audios.append(layer_audio.cpu() * layer_volume)

        logger.info("Mixing final track...")
        total_duration = max(s["start"] + s["duration"] for s in self.sections.values())
        total_samples = int(total_duration * self.sample_rate)

        final_mix = torch.zeros(1, total_samples)
        for audio in layer_audios:
            length = min(audio.shape[-1], total_samples)
            final_mix[..., :length] += audio[..., :length]

        final_mix *= self.song_config.get("master_volume", 1.0)

        if self.processing.get("prevent_clipping", True):
            max_val = torch.abs(final_mix).max()
            if max_val > 0.99:
                final_mix = torch.tanh(final_mix * 0.9) / torch.tanh(torch.tensor(0.9))
                logger.warning(f"Applied soft limiting (peak was {max_val:.2f})")

        output_path = Path(self.song_config.get("output_file", "output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to: {output_path}")
        audio_write(
            str(output_path.with_suffix("")),
            final_mix,
            self.sample_rate,
            strategy=self.processing.get("normalization_strategy", "loudness"),
            loudness_compressor=True,
        )

        if export_stems:
            stems_dir = output_path.parent / f"{output_path.stem}_stems"
            stems_dir.mkdir(exist_ok=True)
            for name, audio in self.generated_layers.items():
                stem_path = stems_dir / f"{name}.wav"
                audio_write(
                    str(stem_path.with_suffix("")),
                    audio,
                    self.sample_rate,
                    strategy="loudness",
                    loudness_compressor=True,
                )
                logger.info(f"Exported stem: {stem_path}")

        logger.info("Production complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AudioCraft Layered Music Producer")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--export-stems", action="store_true", help="Export individual layer stems"
    )
    args = parser.parse_args()

    producer = MusicProducer(args.config)
    producer.produce(export_stems=args.export_stems)


if __name__ == "__main__":
    main()
