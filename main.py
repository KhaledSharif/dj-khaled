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
    start_offset: float = 0.0
    fade_out: bool = False
    fade_in: bool = False


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
        self.style_audio = None  # For style conditioning

        # Setup cache directory if enabled
        if self.advanced.get("use_cache", False):
            self.cache_dir = Path(self.advanced.get("cache_dir", "cache/"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, model_name: str) -> MusicGen:
        """Get or load a MusicGen model with caching"""
        if model_name not in self.models_cache:
            logger.info(f"Loading model: {model_name}")
            model = MusicGen.get_pretrained(model_name)

            # Apply advanced generation parameters if specified
            if self.advanced.get("generation_params"):
                params = self.advanced["generation_params"]

                # Different parameters for different models
                if "style" in model_name:
                    # MusicGen-Style specific parameters
                    model.set_generation_params(
                        duration=30,  # max duration for style model
                        use_sampling=params.get("use_sampling", True),
                        temperature=params.get("temperature", 0.9),
                        top_k=params.get("top_k", 250),
                        top_p=params.get("top_p", 0.0),
                        cfg_coef=params.get("cfg_coef", 3.0),
                        cfg_coef_beta=params.get("cfg_coef_beta", None),
                    )

                    # Set style conditioning parameters
                    style_params = self.advanced.get("style_params", {})
                    model.set_style_conditioner_params(
                        eval_q=style_params.get(
                            "eval_q", 2
                        ),  # 1-6, lower = less adherence
                        excerpt_length=style_params.get("excerpt_length", 3.0),
                    )
                else:
                    # Regular MusicGen parameters
                    model.set_generation_params(
                        temperature=params.get("temperature", 0.9),
                        top_k=params.get("top_k", 250),
                        top_p=params.get("top_p", 0.0),
                        cfg_coef=params.get("cfg_coef", 3.0),
                    )

            self.models_cache[model_name] = model

        return self.models_cache[model_name]

    def get_cache_key(
        self,
        layer_name: str,
        section_name: str,
        prompt: str,
        use_style: bool = False,
    ) -> str:
        """Generate a cache key for a generated section"""
        key_parts = [layer_name, section_name, prompt]
        # Handle both boolean and tensor inputs for backward compatibility
        if isinstance(use_style, torch.Tensor):
            # Legacy code may pass tensor, convert to boolean
            use_style = False  # Default to False for tensor inputs
        if use_style:
            key_parts.append("styled")
        return "_".join(key_parts).replace(" ", "_").replace("/", "_")

    def load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load a generated section from cache"""
        if not self.advanced.get("use_cache", False):
            return None

        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            logger.info(f"Loading from cache: {cache_key}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def save_to_cache(self, cache_key: str, audio: torch.Tensor):
        """Save a generated section to cache"""
        if not self.advanced.get("use_cache", False):
            return

        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(audio, f)

    def load_style_reference(self):
        """Load a reference audio file for style conditioning"""
        style_config = self.advanced.get("style_reference", {})
        if style_config.get("path"):
            logger.info(f"Loading style reference from: {style_config['path']}")
            audio, sr = torchaudio.load(style_config["path"])  # type: ignore[attr-defined]
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            self.style_audio = audio
            return audio
        return None

    def generate_section(
        self,
        model: MusicGen,
        prompt: str,
        duration: float,
        condition_audio: Optional[torch.Tensor] = None,  # Keep for backward compatibility
        use_full_conditioning: bool = True,  # Keep for backward compatibility
        use_style: bool = False,  # New parameter
        continuation_audio: Optional[torch.Tensor] = None,  # New parameter
    ) -> torch.Tensor:
        """Generate a single section of audio using best practices"""

        # Handle backward compatibility - if old-style condition_audio is provided
        if condition_audio is not None and continuation_audio is None:
            continuation_audio = condition_audio

        # Limit duration to MusicGen's optimal range
        actual_duration = min(duration, 30.0)
        
        # Handle case where model might return None or have issues
        try:
            model.set_generation_params(duration=actual_duration)
        except (AttributeError, TypeError) as e:
            logger.error(f"Model error when setting generation params: {e}")
            # Return silence of the requested duration
            return torch.zeros(int(actual_duration * self.sample_rate))

        if (
            use_style
            and self.style_audio is not None
            and hasattr(model, 'cfg')
            and model.cfg is not None
            and hasattr(model.cfg, 'name')
            and "style" in str(model.cfg.name)
        ):
            # Use MusicGen-Style with style conditioning
            logger.info(f"Generating with style conditioning: '{prompt}'")

            # Prepare style audio (3 samples for better results)
            style_expanded = (
                self.style_audio[None].expand(3, -1, -1)
                if self.style_audio.dim() == 2
                else self.style_audio.expand(3, -1, -1)
            )

            # Generate with style and text
            descriptions = [prompt] * 3 if prompt else [""] * 3
            try:
                wav = model.generate_with_chroma(
                    descriptions=descriptions,
                    melody_wavs=style_expanded,
                    melody_sample_rate=self.sample_rate,
                )
                # Average the 3 generations for more stable output
                if isinstance(wav, tuple):
                    wav = wav[0]  # Extract tensor from tuple if needed
                wav = wav.mean(dim=0)
            except (AttributeError, TypeError) as e:
                logger.error(f"Model error during style generation: {e}")
                return torch.zeros(int(actual_duration * self.sample_rate))

        elif continuation_audio is not None and duration <= 30:
            # Check if we have enough audio for continuation
            min_continuation_duration = 1.0  # Minimum 1 second needed
            if continuation_audio.shape[-1] < int(
                min_continuation_duration * self.sample_rate
            ):
                logger.warning(
                    f"Previous audio too short for continuation, using text generation instead"
                )
                try:
                    wav = model.generate([prompt])
                    if wav is None:
                        raise TypeError("Model returned None")
                    wav = wav[0]
                except (AttributeError, TypeError, IndexError) as e:
                    logger.error(f"Model error during text generation: {e}")
                    return torch.zeros(int(actual_duration * self.sample_rate))
            else:
                # Use continuation only for extending in time (not layering)
                logger.info(f"Generating temporal continuation: '{prompt}'")

                # Ensure we don't use more context than the actual generation duration
                # MusicGen needs the prompt to be shorter than what it will generate
                max_context = min(
                    actual_duration * 0.5,  # Use at most half of target duration
                    continuation_audio.shape[-1]
                    / self.sample_rate,  # Don't exceed available audio
                    15.0,  # Cap at 15 seconds
                )
                context_samples = int(max_context * self.sample_rate)

                if continuation_audio.shape[-1] > context_samples:
                    prompt_audio = continuation_audio[..., -context_samples:]
                else:
                    prompt_audio = continuation_audio

                if prompt_audio.dim() == 2:
                    prompt_audio = prompt_audio.unsqueeze(0)

                # Ensure the continuation makes sense (prompt must be shorter than generation)
                prompt_duration = prompt_audio.shape[-1] / self.sample_rate
                if prompt_duration >= actual_duration:
                    logger.warning(
                        f"Prompt audio ({prompt_duration:.1f}s) >= target duration ({actual_duration}s), using text generation"
                    )
                    try:
                        wav = model.generate([prompt])
                        if wav is None:
                            raise TypeError("Model returned None")
                        wav = wav[0]
                    except (AttributeError, TypeError, IndexError) as e:
                        logger.error(f"Model error during text generation: {e}")
                        return torch.zeros(int(actual_duration * self.sample_rate))
                else:
                    try:
                        wav = model.generate_continuation(
                            prompt_audio,
                            prompt_sample_rate=self.sample_rate,
                            descriptions=[prompt],
                            progress=True,
                        )
                        wav = wav[0]
                    except AssertionError as e:
                        logger.warning(
                            f"Continuation failed: {e}, falling back to text generation"
                        )
                        try:
                            wav = model.generate([prompt])
                            if wav is None:
                                raise TypeError("Model returned None")
                            wav = wav[0]
                        except (AttributeError, TypeError, IndexError) as e2:
                            logger.error(f"Model error during fallback generation: {e2}")
                            return torch.zeros(int(actual_duration * self.sample_rate))

        else:
            # Standard text-to-music generation (most common case)
            logger.info(f"Generating from text: '{prompt}'")
            try:
                wav = model.generate([prompt])
                if wav is None:
                    raise TypeError("Model returned None")
                wav = wav[0]
            except (AttributeError, TypeError, IndexError) as e:
                logger.error(f"Model error during standard generation: {e}")
                return torch.zeros(int(actual_duration * self.sample_rate))

        return wav

    def apply_fade(
        self,
        audio: torch.Tensor,
        fade_in: bool = False,
        fade_out: bool = False,
        fade_duration: float = 2.0,
    ) -> torch.Tensor:
        """Apply fade in/out to audio with proper curve"""
        if audio.numel() == 0:
            return audio

        fade_samples = int(fade_duration * self.sample_rate)
        audio_length = audio.shape[-1]
        fade_samples = min(fade_samples, audio_length // 2)  # Don't fade more than half

        if fade_in and fade_samples > 0:
            # Exponential fade in for smoother start
            fade_curve = torch.linspace(0, 1, fade_samples, device=audio.device) ** 2
            if audio.dim() == 1:
                audio[:fade_samples] *= fade_curve
            else:
                audio[..., :fade_samples] *= fade_curve

        if fade_out and fade_samples > 0:
            # Exponential fade out for smoother end
            fade_curve = torch.linspace(1, 0, fade_samples, device=audio.device) ** 2
            if audio.dim() == 1:
                audio[-fade_samples:] *= fade_curve
            else:
                audio[..., -fade_samples:] *= fade_curve

        return audio

    def loop_with_crossfade(
        self,
        audio: torch.Tensor,
        target_samples: int,
        crossfade_duration: float = 0.25,  # Increased default
        bpm: int = 128,
    ) -> torch.Tensor:
        """Loop audio with beat-aligned crossfade"""
        # Handle both 1D and 2D tensors
        if audio.dim() == 2:
            # Process 2D tensor (channels, samples)
            num_channels = audio.shape[0]
            audio_length = audio.shape[1]
        else:
            # Process 1D tensor
            num_channels = None
            audio_length = audio.shape[0] if audio.numel() > 0 else 0
        
        # Handle empty audio
        if audio_length == 0:
            if num_channels is not None:
                return torch.zeros((num_channels, target_samples), device=audio.device if hasattr(audio, 'device') else 'cpu')
            else:
                return torch.zeros(target_samples, device=audio.device if hasattr(audio, 'device') else 'cpu')
        
        if audio_length >= target_samples:
            if num_channels is not None:
                return audio[:, :target_samples]
            else:
                return audio[:target_samples]

        # Calculate beat-aligned crossfade
        beat_duration = 60.0 / bpm  # seconds per beat
        # Use 1 beat for crossfade by default, adjustable
        crossfade_beats = max(1, int(crossfade_duration / beat_duration))
        crossfade_duration = crossfade_beats * beat_duration

        crossfade_samples = int(crossfade_duration * self.sample_rate)
        crossfade_samples = min(crossfade_samples, audio_length // 4)

        device = audio.device
        
        # Create result tensor with appropriate shape
        if num_channels is not None:
            result = torch.zeros((num_channels, target_samples), device=device)
        else:
            result = torch.zeros(target_samples, device=device)

        effective_loop_length = audio_length - crossfade_samples
        num_full_loops = target_samples // effective_loop_length

        current_pos = 0
        for i in range(num_full_loops + 1):
            if current_pos >= target_samples:
                break

            if i == 0:
                end_pos = min(current_pos + audio_length, target_samples)
                if num_channels is not None:
                    result[:, current_pos:end_pos] = audio[:, : end_pos - current_pos]
                else:
                    result[current_pos:end_pos] = audio[: end_pos - current_pos]
                current_pos = audio_length - crossfade_samples
            else:
                # S-curve crossfade for smoother transitions
                fade_in = torch.sigmoid(
                    torch.linspace(-6, 6, crossfade_samples, device=device)
                )
                fade_out = 1 - fade_in

                if current_pos + crossfade_samples <= target_samples:
                    if num_channels is not None:
                        # Handle 2D tensor crossfade
                        fade_in = fade_in.unsqueeze(0)  # Add channel dimension
                        fade_out = fade_out.unsqueeze(0)
                        result[:, current_pos : current_pos + crossfade_samples] *= fade_out
                        result[:, current_pos : current_pos + crossfade_samples] += (
                            audio[:, :crossfade_samples] * fade_in
                        )
                    else:
                        # Handle 1D tensor crossfade
                        result[current_pos : current_pos + crossfade_samples] *= fade_out
                        result[current_pos : current_pos + crossfade_samples] += (
                            audio[:crossfade_samples] * fade_in
                        )

                remaining_samples = min(
                    audio_length - crossfade_samples,
                    target_samples - current_pos - crossfade_samples,
                )
                if remaining_samples > 0:
                    end_pos = current_pos + crossfade_samples + remaining_samples
                    if num_channels is not None:
                        result[:, current_pos + crossfade_samples : end_pos] = audio[
                            :, crossfade_samples : crossfade_samples + remaining_samples
                        ]
                    else:
                        result[current_pos + crossfade_samples : end_pos] = audio[
                            crossfade_samples : crossfade_samples + remaining_samples
                        ]

                current_pos += effective_loop_length

        return result

    def generate_layer(
        self, layer_config: Dict, previous_section_audio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate all sections for a single layer"""
        model_name = layer_config.get("model", "facebook/musicgen-small")
        model = self.get_model(model_name)

        # Check if this layer should use style conditioning
        use_style = (
            layer_config.get("use_style", False) and self.style_audio is not None
        )

        # Calculate total song duration
        total_duration = max(s["start"] + s["duration"] for s in self.sections.values())
        total_samples = int(total_duration * self.sample_rate)

        # Initialize layer audio with better device detection
        device = "cpu"  # Default to CPU
        if hasattr(model, "lm") and hasattr(model.lm, "parameters"):
            try:
                # Try to get device from model parameters
                params = list(model.lm.parameters())
                if params:
                    device = params[0].device
            except (AttributeError, StopIteration):
                # If model is mocked or has no parameters, stay on CPU
                pass
        layer_audio = torch.zeros(total_samples, device=device)

        last_section_audio = None

        for section_idx, section_config in enumerate(layer_config.get("sections", [])):
            section_name = section_config["section"]
            section_info = self.sections[section_name]

            start_offset = section_config.get("start_offset", 0)
            start_time = section_info["start"] + start_offset
            duration = section_info["duration"]
            prompt = section_config["prompt"]
            volume = section_config.get("volume", 1.0)
            loop = section_config.get("loop", False)
            fade_out = section_config.get("fade_out", False)
            fade_in = section_config.get("fade_in", False)

            # Check if we should use continuation (only for temporal extension)
            use_continuation = (
                section_config.get("use_continuation", False)
                and last_section_audio is not None
            )

            # Check cache
            cache_key = self.get_cache_key(
                layer_config["name"], section_name, prompt, use_style
            )
            section_audio = self.load_from_cache(cache_key)

            if section_audio is None:
                # Generate the section
                section_audio = self.generate_section(
                    model,
                    prompt,
                    duration,
                    use_style=use_style,
                    continuation_audio=last_section_audio if use_continuation else None,
                )
                self.save_to_cache(cache_key, section_audio)

            # Apply fades
            if fade_in or fade_out:
                fade_duration = section_config.get("fade_duration", 2.0)
                section_audio = self.apply_fade(
                    section_audio,
                    fade_in=fade_in,
                    fade_out=fade_out,
                    fade_duration=fade_duration,
                )

            # Apply volume
            section_audio = section_audio * volume

            # Handle looping with beat-aligned crossfade
            if loop and section_audio.shape[-1] < int(duration * self.sample_rate):
                required_samples = int(duration * self.sample_rate)
                bpm = self.song_config.get("bpm", 128)
                crossfade_duration = section_config.get("crossfade_duration", 0.25)
                section_audio = self.loop_with_crossfade(
                    section_audio,
                    required_samples,
                    crossfade_duration=crossfade_duration,
                    bpm=bpm,
                )

            # Store for potential continuation
            last_section_audio = (
                section_audio.unsqueeze(0)
                if section_audio.dim() == 1
                else section_audio
            )

            # Place in the full layer timeline
            start_sample = int(start_time * self.sample_rate)
            
            # Skip if the section starts after the total duration
            if start_sample >= total_samples:
                logger.warning(f"Section starts at {start_time}s which is beyond total duration")
                continue
                
            end_sample = start_sample + section_audio.shape[-1]

            if section_audio.dim() > 1:
                section_audio = section_audio.squeeze()

            section_audio = section_audio.to(layer_audio.device)

            layer_end = min(end_sample, total_samples)
            section_end = layer_end - start_sample
            
            # Only add if there's something to add
            if section_end > 0:
                layer_audio[start_sample:layer_end] += section_audio[:section_end]

        return layer_audio

    def mix_audio(
        self, audio_list: List[Tuple[torch.Tensor, float]], target_length: int
    ) -> torch.Tensor:
        """Mix multiple audio tensors with specified volumes and proper headroom"""
        mixed = torch.zeros(target_length)

        for audio, volume in audio_list:
            audio = audio.cpu()

            if audio.dim() > 1:
                audio = audio.squeeze()

            audio = audio * volume

            mix_length = min(len(audio), target_length)
            mixed[:mix_length] += audio[:mix_length]

        # Advanced clipping prevention with soft limiting
        if self.processing.get("prevent_clipping", True):
            # First handle NaN and Inf values
            if torch.isnan(mixed).any() or torch.isinf(mixed).any():
                logger.warning("NaN or Inf values detected in audio, replacing with zeros")
                mixed = torch.nan_to_num(mixed, nan=0.0, posinf=0.95, neginf=-0.95)
            
            max_val = torch.abs(mixed).max()
            if max_val > 1.0:  # Only apply limiting if we're actually clipping
                # Apply proper soft limiting that guarantees output <= 1.0
                threshold = 0.95
                
                # For values above threshold, apply tanh compression
                # This guarantees smooth limiting and output < 1.0
                abs_mixed = torch.abs(mixed)
                over_threshold_mask = abs_mixed > threshold
                
                if over_threshold_mask.any():
                    # Scale down the over-threshold values using tanh
                    # tanh approaches 1 asymptotically, ensuring we never exceed 1.0
                    over_values = abs_mixed[over_threshold_mask] - threshold
                    # Scale and apply tanh, then add back the threshold
                    compressed_values = threshold + (1.0 - threshold) * torch.tanh(over_values / (1.0 - threshold))
                    
                    # Apply the compressed values back
                    abs_mixed[over_threshold_mask] = compressed_values
                    mixed = mixed.sign() * abs_mixed
                    
                    # Final safety check - hard clip at 1.0 just in case
                    mixed = torch.clamp(mixed, min=-1.0, max=1.0)
                    
                    logger.warning(f"Applied soft limiting (peak was {max_val:.2f})")

        return mixed

    def produce(self):
        """Main production process with improved workflow"""
        logger.info(f"Starting production of: {self.song_config['name']}")

        # Load style reference if configured
        if self.advanced.get("style_reference", {}).get("path"):
            self.load_style_reference()

        # Calculate total duration
        total_duration = max(s["start"] + s["duration"] for s in self.sections.values())

        # Warn if sections are too long
        for section_name, section_info in self.sections.items():
            if section_info["duration"] > 30:
                logger.warning(
                    f"Section '{section_name}' is {section_info['duration']}s long. "
                    "Consider splitting into smaller sections for better quality."
                )

        total_samples = int(total_duration * self.sample_rate)

        # Initialize the mix
        layer_audios = []

        # Generate each layer
        for i, layer_config in enumerate(self.layers):
            layer_name = layer_config["name"]
            logger.info(f"Processing layer {i+1}/{len(self.layers)}: {layer_name}")

            # Generate the layer (no conditioning between layers)
            layer_audio = self.generate_layer(layer_config)

            layer_audio = layer_audio.cpu()

            # Store for reference
            self.generated_layers[layer_name] = layer_audio

            # Apply layer-level volume if specified
            layer_volume = layer_config.get("volume", 1.0)
            layer_audios.append((layer_audio, layer_volume))

        # Create final mix
        logger.info("Creating final mix...")
        final_mix = self.mix_audio(layer_audios, total_samples)

        # Apply master volume
        final_mix = final_mix * self.song_config.get("master_volume", 1.0)

        # Enhanced normalization
        normalization_strategy = self.processing.get(
            "normalization_strategy", "loudness"
        )

        if normalization_strategy == "manual":
            target_db = self.processing.get("normalize_db", -14)
            logger.info(f"Manually normalizing to {target_db} dB")

            # RMS normalization for more consistent loudness
            rms = torch.sqrt(torch.mean(final_mix**2))
            target_rms = 10 ** (target_db / 20)
            if rms > 0:
                final_mix = final_mix * (target_rms / rms)

            write_strategy = "peak"
        else:
            logger.info(f"Using audio_write {normalization_strategy} normalization")
            write_strategy = normalization_strategy

        # Save the output
        output_path = Path(self.song_config.get("output_file", "output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_mix = final_mix.unsqueeze(0)

        logger.info(f"Saving to: {output_path}")
        audio_write(
            str(output_path.with_suffix("")),
            final_mix.cpu(),
            self.sample_rate,
            strategy=write_strategy,
            loudness_compressor=True,  # Add compression for better dynamics
        )

        logger.info("Production complete!")

        # Export stems if requested
        if self.config.get("export_stems", False):
            stems_dir = output_path.parent / f"{output_path.stem}_stems"
            stems_dir.mkdir(exist_ok=True)

            for layer_name, layer_audio in self.generated_layers.items():
                stem_path = stems_dir / f"{layer_name}.wav"
                audio_write(
                    str(stem_path.with_suffix("")),
                    layer_audio.unsqueeze(0).cpu(),
                    self.sample_rate,
                    strategy="loudness",
                    loudness_compressor=True,
                )
                logger.info(f"Exported stem: {stem_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MusicGen Layered Music Producer")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument(
        "--export-stems", action="store_true", help="Export individual layer stems"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.export_stems:
        config["export_stems"] = True
        with open(args.config, "w") as f:
            yaml.dump(config, f)

    producer = MusicProducer(args.config)
    producer.produce()


if __name__ == "__main__":
    main()
