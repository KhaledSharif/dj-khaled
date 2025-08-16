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

        # Setup cache directory if enabled
        if self.advanced.get("use_cache", False):
            self.cache_dir = Path(self.advanced.get("cache_dir", "cache/"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model(self, model_name: str) -> MusicGen:
        """Get or load a MusicGen model with caching"""
        if model_name not in self.models_cache:
            logger.info(f"Loading model: {model_name}")
            # Handle both old-style and new-style model names
            if "/" not in model_name:
                model = MusicGen.get_pretrained(model_name)
            else:
                model = MusicGen.get_pretrained(model_name)

            # Apply advanced generation parameters if specified
            if self.advanced.get("generation_params"):
                params = self.advanced["generation_params"]
                model.set_generation_params(
                    temperature=params.get("temperature", 1.0),
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
        condition_audio: Optional[torch.Tensor] = None,
    ) -> str:
        """Generate a cache key for a generated section"""
        key_parts = [layer_name, section_name, prompt]
        if condition_audio is not None:
            # Add a hash of the conditioning audio
            audio_hash = hashlib.md5(
                condition_audio.cpu().numpy().tobytes()
            ).hexdigest()[:8]
            key_parts.append(audio_hash)
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

    def generate_section(
        self,
        model: MusicGen,
        prompt: str,
        duration: float,
        condition_audio: Optional[torch.Tensor] = None,
        use_full_conditioning: bool = True,
    ) -> torch.Tensor:
        """Generate a single section of audio"""
        model.set_generation_params(duration=duration)

        if condition_audio is not None:
            # Ensure conditioning audio is the right shape and on the right device
            if condition_audio.dim() == 2:
                condition_audio = condition_audio.unsqueeze(0)

            # FIX 2: Option to use full conditioning audio or a representative sample
            if use_full_conditioning:
                # Use more conditioning audio for better context, but respect model limits
                # MusicGen typically can handle around 30 seconds of prompt audio
                total_samples = condition_audio.shape[-1]
                # Limit to 8 seconds to avoid assertion errors, but more than the original 2 seconds
                max_prompt_duration = min(8.0, duration * 0.5)  # Use up to 8 seconds or half the target duration
                prompt_samples = min(int(self.sample_rate * max_prompt_duration), total_samples)
                
                if total_samples > prompt_samples:
                    # Sample from the end portion for better continuity
                    # This gives recent context while avoiding the assertion error
                    prompt_audio = condition_audio[..., -prompt_samples:]
                else:
                    prompt_audio = condition_audio
            else:
                # Legacy behavior: use only the last few seconds
                prompt_duration = min(2.0, duration / 4)
                prompt_samples = int(self.sample_rate * prompt_duration)
                
                if condition_audio.shape[-1] > prompt_samples:
                    prompt_audio = condition_audio[..., -prompt_samples:]
                else:
                    prompt_audio = condition_audio

            # Generate as continuation of the prompt audio
            logger.info(f"Generating continuation with prompt: '{prompt}'")
            wav = model.generate_continuation(
                prompt_audio,
                prompt_sample_rate=self.sample_rate,
                descriptions=[prompt],
                progress=True,
            )
        else:
            logger.info(f"Generating: '{prompt}'")
            wav = model.generate([prompt])

        return wav[0]  # Return first (and only) generation

    def apply_fade(
        self, audio: torch.Tensor, fade_out: bool = False, fade_duration: float = 2.0
    ) -> torch.Tensor:
        """Apply fade in/out to audio"""
        if audio.numel() == 0:  # Handle empty audio
            return audio
            
        fade_samples = int(fade_duration * self.sample_rate)
        audio_length = audio.shape[-1]
        
        # Limit fade samples to audio length
        fade_samples = min(fade_samples, audio_length)
        
        if fade_out and fade_samples > 0:
            # Create fade curve on the same device as the audio
            fade_curve = torch.linspace(1, 0, fade_samples, device=audio.device)
            if audio.dim() == 1:
                audio[-fade_samples:] *= fade_curve
            else:
                audio[..., -fade_samples:] *= fade_curve

        return audio
    
    def loop_with_crossfade(
        self, audio: torch.Tensor, target_samples: int, crossfade_duration: float = 0.05
    ) -> torch.Tensor:
        """Loop audio with crossfade to prevent clicks and pops"""
        audio_length = audio.shape[-1]
        if audio_length >= target_samples:
            return audio[:target_samples]
        
        # Calculate crossfade samples
        crossfade_samples = int(crossfade_duration * self.sample_rate)
        crossfade_samples = min(crossfade_samples, audio_length // 4)  # Don't crossfade more than 25% of the loop
        
        # Create the looped audio
        device = audio.device
        result = torch.zeros(target_samples, device=device)
        
        # Calculate how many full loops we need
        effective_loop_length = audio_length - crossfade_samples
        num_full_loops = target_samples // effective_loop_length
        
        current_pos = 0
        for i in range(num_full_loops + 1):
            if current_pos >= target_samples:
                break
                
            if i == 0:
                # First iteration: place the full audio
                end_pos = min(current_pos + audio_length, target_samples)
                result[current_pos:end_pos] = audio[:end_pos - current_pos]
                current_pos = audio_length - crossfade_samples
            else:
                # Subsequent iterations: crossfade with previous
                # Create fade curves
                fade_in = torch.linspace(0, 1, crossfade_samples, device=device)
                fade_out = torch.linspace(1, 0, crossfade_samples, device=device)
                
                # Apply crossfade at the loop point
                if current_pos + crossfade_samples <= target_samples:
                    # Fade out the end of previous iteration
                    result[current_pos:current_pos + crossfade_samples] *= fade_out
                    # Add faded in beginning of new iteration
                    result[current_pos:current_pos + crossfade_samples] += audio[:crossfade_samples] * fade_in
                
                # Add the rest of this loop iteration
                remaining_samples = min(audio_length - crossfade_samples, target_samples - current_pos - crossfade_samples)
                if remaining_samples > 0:
                    end_pos = current_pos + crossfade_samples + remaining_samples
                    result[current_pos + crossfade_samples:end_pos] = audio[crossfade_samples:crossfade_samples + remaining_samples]
                
                current_pos += effective_loop_length
        
        return result

    def mix_audio(
        self, audio_list: List[Tuple[torch.Tensor, float]], target_length: int
    ) -> torch.Tensor:
        """Mix multiple audio tensors with specified volumes"""
        # Use CPU for mixing to avoid device conflicts
        mixed = torch.zeros(target_length)

        for audio, volume in audio_list:
            # Move to CPU for mixing
            audio = audio.cpu()
            
            # Ensure audio is 1D
            if audio.dim() > 1:
                audio = audio.squeeze()

            # Apply volume
            audio = audio * volume

            # Add to mix (handling different lengths)
            mix_length = min(len(audio), target_length)
            mixed[:mix_length] += audio[:mix_length]

        # Prevent clipping if enabled
        if self.processing.get("prevent_clipping", True):
            max_val = torch.abs(mixed).max()
            if max_val > 1.0:
                logger.warning(
                    f"Clipping detected (max: {max_val:.2f}), normalizing..."
                )
                mixed = mixed / max_val * 0.95

        return mixed

    def generate_layer(
        self, layer_config: Dict, condition_mix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate all sections for a single layer"""
        model_name = layer_config.get("model", "facebook/musicgen-small")
        model = self.get_model(model_name)

        # Calculate total song duration
        total_duration = max(s["start"] + s["duration"] for s in self.sections.values())
        total_samples = int(total_duration * self.sample_rate)

        # Initialize layer audio on the same device as the model
        device = model.device if hasattr(model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        layer_audio = torch.zeros(total_samples, device=device)

        for section_config in layer_config.get("sections", []):
            section_name = section_config["section"]
            section_info = self.sections[section_name]

            # FIX 1: start_offset should only affect when the audio starts, not its duration
            start_offset = section_config.get("start_offset", 0)
            start_time = section_info["start"] + start_offset
            # Keep the full duration - the audio should play for the entire section duration
            duration = section_info["duration"]
            prompt = section_config["prompt"]
            volume = section_config.get("volume", 1.0)
            loop = section_config.get("loop", False)
            fade_out = section_config.get("fade_out", False)

            # Prepare conditioning audio for this section if needed
            section_condition = None
            if condition_mix is not None:
                start_sample = int(start_time * self.sample_rate)
                end_sample = int((start_time + duration) * self.sample_rate)
                section_condition = condition_mix[start_sample:end_sample].unsqueeze(0)

            # Check cache
            cache_key = self.get_cache_key(
                layer_config["name"], section_name, prompt, section_condition
            )
            section_audio = self.load_from_cache(cache_key)

            if section_audio is None:
                # Generate the section with full conditioning option
                use_full_conditioning = self.advanced.get("use_full_conditioning", True)
                section_audio = self.generate_section(
                    model, prompt, duration, section_condition, use_full_conditioning
                )
                self.save_to_cache(cache_key, section_audio)

            # Apply effects
            if fade_out:
                section_audio = self.apply_fade(section_audio, fade_out=True)

            # Apply volume
            section_audio = section_audio * volume

            # FIX 3: Handle looping with crossfade to prevent clicks
            if loop and section_audio.shape[-1] < int(duration * self.sample_rate):
                required_samples = int(duration * self.sample_rate)
                section_audio = self.loop_with_crossfade(
                    section_audio, required_samples, crossfade_duration=0.05
                )

            # Place in the full layer timeline
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + section_audio.shape[-1]

            if section_audio.dim() > 1:
                section_audio = section_audio.squeeze()

            # Ensure both tensors are on the same device
            section_audio = section_audio.to(layer_audio.device)

            # Mix into layer (in case of overlapping sections)
            layer_end = min(end_sample, total_samples)
            section_end = layer_end - start_sample
            layer_audio[start_sample:layer_end] += section_audio[:section_end]

        return layer_audio

    def produce(self):
        """Main production process"""
        logger.info(f"Starting production of: {self.song_config['name']}")

        # Calculate total duration
        total_duration = max(s["start"] + s["duration"] for s in self.sections.values())
        total_samples = int(total_duration * self.sample_rate)

        # Initialize the mix
        layer_audios = []

        # Generate each layer sequentially
        for i, layer_config in enumerate(self.layers):
            layer_name = layer_config["name"]
            logger.info(f"Processing layer {i+1}/{len(self.layers)}: {layer_name}")

            # Determine conditioning audio
            condition_audio = None
            if "condition_on" in layer_config and layer_config["condition_on"]:
                # Create a mix of specified layers for conditioning
                condition_layers = []
                for cond_layer_name in layer_config["condition_on"]:
                    if cond_layer_name in self.generated_layers:
                        condition_layers.append(
                            (self.generated_layers[cond_layer_name], 1.0)
                        )

                if condition_layers:
                    condition_audio = self.mix_audio(condition_layers, total_samples)

            # Generate the layer
            layer_audio = self.generate_layer(layer_config, condition_audio)

            # Move to CPU for consistency in mixing
            layer_audio = layer_audio.cpu()

            # Store for future conditioning
            self.generated_layers[layer_name] = layer_audio
            layer_audios.append((layer_audio, 1.0))  # Full volume for final mix

        # Create final mix
        logger.info("Creating final mix...")
        final_mix = self.mix_audio(layer_audios, total_samples)

        # Apply master volume
        final_mix = final_mix * self.song_config.get("master_volume", 1.0)

        # FIX 4: Choose normalization strategy - either manual OR audio_write's strategy
        normalization_strategy = self.processing.get("normalization_strategy", "loudness")
        
        if normalization_strategy == "manual":
            # Manual normalization based on peak or target dB
            if self.processing.get("normalize", True):
                target_db = self.processing.get("normalize_db", -14)
                logger.info(f"Manually normalizing to {target_db} dB peak")
                max_val = torch.abs(final_mix).max()
                target_peak = 10 ** (target_db / 20)
                if max_val > 0:
                    final_mix = final_mix * (target_peak / max_val)
            write_strategy = "peak"  # Use peak strategy to preserve our manual normalization
        else:
            # Let audio_write handle normalization (default behavior)
            logger.info(f"Using audio_write {normalization_strategy} normalization")
            write_strategy = normalization_strategy

        # Save the output
        output_path = Path(self.song_config.get("output_file", "output.wav"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Reshape for audio_write (needs 2D tensor)
        final_mix = final_mix.unsqueeze(0)

        logger.info(f"Saving to: {output_path}")
        audio_write(
            str(output_path.with_suffix("")),
            final_mix.cpu(),
            self.sample_rate,
            strategy=write_strategy,
        )

        logger.info("Production complete!")

        # Also save individual layers if requested
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

    # Load config and override stem export if specified
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.export_stems:
        config["export_stems"] = True
        # Save back temporarily
        with open(args.config, "w") as f:
            yaml.dump(config, f)

    # Run production
    producer = MusicProducer(args.config)
    producer.produce()


if __name__ == "__main__":
    main()
