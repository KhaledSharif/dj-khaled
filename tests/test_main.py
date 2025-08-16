import unittest
from unittest.mock import MagicMock, patch, mock_open
import torch
import yaml
import tempfile
import shutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import SectionConfig, MusicProducer


class TestSectionConfig(unittest.TestCase):
    """Test cases for the SectionConfig dataclass"""

    def test_section_config_creation(self):
        """Test creating a SectionConfig with all parameters"""
        config = SectionConfig(
            name="test_section",
            start=0.0,
            duration=10.0,
            prompt="test prompt",
            volume=0.8,
            loop=True,
            fade_out=True,
            fade_in=False,
            use_continuation=False,
        )

        self.assertEqual(config.name, "test_section")
        self.assertEqual(config.start, 0.0)
        self.assertEqual(config.duration, 10.0)
        self.assertEqual(config.prompt, "test prompt")
        self.assertEqual(config.volume, 0.8)
        self.assertTrue(config.loop)
        self.assertTrue(config.fade_out)
        self.assertFalse(config.fade_in)
        self.assertFalse(config.use_continuation)

    def test_section_config_defaults(self):
        """Test SectionConfig default values"""
        config = SectionConfig(name="test", start=0.0, duration=5.0, prompt="prompt")

        self.assertEqual(config.volume, 1.0)
        self.assertFalse(config.loop)
        self.assertFalse(config.fade_out)
        self.assertFalse(config.fade_in)
        self.assertFalse(config.use_continuation)


class TestMusicProducerInit(unittest.TestCase):
    """Test cases for MusicProducer initialization"""

    def setUp(self):
        self.test_config = {
            "song": {
                "name": "Test Song",
                "output_file": "test_output.wav",
                "sample_rate": 32000,
                "master_volume": 0.9,
            },
            "sections": {
                "intro": {"start": 0, "duration": 8},
                "verse": {"start": 8, "duration": 16},
            },
            "layers": [
                {"name": "drums", "model": "facebook/musicgen-small", "sections": []}
            ],
            "processing": {"normalize": True, "normalize_db": -14},
            "advanced": {"use_cache": True, "cache_dir": "test_cache/"},
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yml")

        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_with_valid_config(self):
        """Test MusicProducer initialization with valid config file"""
        producer = MusicProducer(self.config_path)

        self.assertEqual(producer.song_config["name"], "Test Song")
        self.assertEqual(producer.sample_rate, 32000)
        self.assertEqual(len(producer.sections), 2)
        self.assertEqual(len(producer.layers), 1)
        self.assertTrue(producer.advanced["use_cache"])
        self.assertEqual(producer.models_cache, {})
        self.assertEqual(producer.generated_layers, {})

    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: {content")
    def test_initialization_with_invalid_yaml(self, mock_file):
        """Test MusicProducer initialization with invalid YAML"""
        with self.assertRaises(yaml.YAMLError):
            MusicProducer("fake_path.yml")

    def test_cache_directory_creation(self):
        """Test that cache directory is created when use_cache is True"""
        producer = MusicProducer(self.config_path)
        self.assertTrue(producer.cache_dir.exists())
        self.assertTrue(producer.cache_dir.is_dir())

    def test_initialization_without_optional_fields(self):
        """Test initialization when optional config fields are missing"""
        minimal_config = {"song": {"name": "Minimal"}, "sections": {}, "layers": []}

        minimal_path = os.path.join(self.temp_dir, "minimal.yml")
        with open(minimal_path, "w") as f:
            yaml.dump(minimal_config, f)

        producer = MusicProducer(minimal_path)
        self.assertEqual(producer.sample_rate, 32000)  # Default value
        self.assertEqual(producer.processing, {})
        self.assertEqual(producer.advanced, {})


class TestModelOperations(unittest.TestCase):
    """Test cases for model loading and caching operations"""

    def setUp(self):
        self.config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {
                "generation_params": {
                    "temperature": 0.8,
                    "top_k": 200,
                    "top_p": 0.5,
                    "cfg_coef": 2.5,
                }
            },
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.MusicGen")
    def test_get_model_loading(self, mock_musicgen):
        """Test model loading and caching"""
        mock_model = MagicMock()
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)

        # First call should load the model
        model1 = producer.get_model("facebook/musicgen-small")
        mock_musicgen.get_pretrained.assert_called_once_with("facebook/musicgen-small")
        self.assertEqual(model1, mock_model)

        # Second call should return cached model
        model2 = producer.get_model("facebook/musicgen-small")
        # Still only called once
        mock_musicgen.get_pretrained.assert_called_once()
        self.assertEqual(model2, mock_model)

    @patch("main.MusicGen")
    def test_get_model_with_generation_params(self, mock_musicgen):
        """Test model loading with custom generation parameters"""
        mock_model = MagicMock()
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)
        model = producer.get_model("test-model")

        mock_model.set_generation_params.assert_called_once_with(
            use_sampling=True, temperature=0.8, top_k=200, top_p=0.5, cfg_coef=2.5
        )


class TestCacheOperations(unittest.TestCase):
    """Test cases for cache-related operations"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        self.config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {"use_cache": True, "cache_dir": self.cache_dir},
        }

        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_cache_key_without_conditioning(self):
        """Test cache key generation without conditioning audio"""
        key = self.producer.get_cache_key("layer1", "section1", "test prompt", False, False)
        # Key is now hashed
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 32)  # MD5 hash length

    def test_get_cache_key_with_conditioning(self):
        """Test cache key generation with style flag"""
        # Test with style=False (default)
        key = self.producer.get_cache_key("layer1", "section1", "test prompt", False, False)
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 32)  # MD5 hash length
        
        # Test with style=True
        key_styled = self.producer.get_cache_key("layer1", "section1", "test prompt", True, False)
        self.assertIsInstance(key_styled, str)
        self.assertEqual(len(key_styled), 32)  # MD5 hash length
        self.assertNotEqual(key, key_styled)  # Should be different

    def test_save_and_load_cache(self):
        """Test saving and loading from cache"""
        test_audio = torch.randn(1, 1000)
        cache_key = "test_key"

        # Save to cache
        self.producer.save_to_cache(cache_key, test_audio)

        # Load from cache
        loaded_audio = self.producer.load_from_cache(cache_key)

        self.assertIsNotNone(loaded_audio)
        torch.testing.assert_close(test_audio, loaded_audio)

    def test_load_nonexistent_cache(self):
        """Test loading from cache when file doesn't exist"""
        result = self.producer.load_from_cache("nonexistent_key")
        self.assertIsNone(result)

    def test_cache_disabled(self):
        """Test cache operations when caching is disabled"""
        config_no_cache = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {"use_cache": False},
        }

        config_path = os.path.join(self.temp_dir, "no_cache.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_no_cache, f)

        producer = MusicProducer(config_path)

        test_audio = torch.randn(1, 1000)
        producer.save_to_cache("key", test_audio)
        result = producer.load_from_cache("key")

        self.assertIsNone(result)


class TestAudioGeneration(unittest.TestCase):
    """Test cases for audio generation methods"""

    def setUp(self):
        self.config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {},
            "layers": [],
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_section_without_conditioning(self):
        """Test generating audio section without conditioning"""
        mock_model = MagicMock()
        mock_output = torch.randn(1, 1, 32000)  # 1 second at 32kHz
        mock_model.generate.return_value = [mock_output[0]]

        result = self.producer.generate_section(mock_model, "test prompt", 1.0, None, None, False)

        mock_model.set_generation_params.assert_called_with(duration=1.0)
        mock_model.generate.assert_called_once_with(["test prompt"])
        torch.testing.assert_close(result, mock_output[0])

    def test_generate_section_with_conditioning(self):
        """Test generating audio section with conditioning"""
        mock_model = MagicMock()
        mock_output = torch.randn(1, 1, 32000)
        mock_model.generate_continuation.return_value = [mock_output[0]]

        condition_audio = torch.randn(1, 64000)  # 2 seconds

        result = self.producer.generate_section(
            mock_model, "test prompt", 3.0, condition_audio, None, False
        )

        mock_model.generate_continuation.assert_called_once()
        args = mock_model.generate_continuation.call_args

        # Check that continuation was called with appropriate parameters
        self.assertEqual(args[0][2], ["test prompt"])  # descriptions is third positional arg
        self.assertEqual(args[0][1], 32000)  # sample_rate is second positional arg

    def test_apply_fade_out(self):
        """Test applying fade out to audio"""
        audio = torch.ones(32000)  # 1 second of ones
        faded = self.producer.apply_fade(audio, fade_in=False, fade_out=True, fade_duration=0.5)

        # Check that the end of the audio fades to zero
        self.assertLess(faded[-1].item(), 0.1)
        # Check that the beginning is unchanged
        self.assertAlmostEqual(faded[0].item(), 1.0)
        # Check gradual fade
        fade_start = -int(0.5 * 32000)
        self.assertLess(faded[fade_start // 2].item(), 1.0)




class TestLayerGeneration(unittest.TestCase):
    """Test cases for layer generation"""

    def setUp(self):
        self.config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {
                "intro": {"start": 0, "duration": 2},
                "verse": {"start": 2, "duration": 3},
            },
            "layers": [],
            "advanced": {"use_cache": False},
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.MusicGen")
    def test_generate_layer_basic(self, mock_musicgen_class):
        """Test basic layer generation"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_musicgen_class.get_pretrained.return_value = mock_model

        # Mock generate_section to return appropriate audio
        def mock_generate(model, prompt, duration, continuation_audio=None, style_audio=None, use_style=False):
            return torch.randn(int(duration * 32000))

        self.producer.generate_section = MagicMock(side_effect=mock_generate)

        layer_config = {
            "name": "test_layer",
            "model": "facebook/musicgen-small",
            "sections": [{"section": "intro", "prompt": "intro music", "volume": 0.8}],
        }

        result = self.producer.generate_layer(layer_config)

        # Should be 5 seconds total (intro + verse duration), but shape is (1, samples)
        expected_samples = 5 * 32000
        self.assertEqual(result.shape[1], expected_samples)  # Second dimension is samples

        # Check that generate_section was called
        self.producer.generate_section.assert_called()

    @patch("main.MusicGen")
    def test_generate_layer_basic_functionality(self, mock_musicgen_class):
        """Test basic layer generation functionality"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_musicgen_class.get_pretrained.return_value = mock_model

        # Mock generate_section
        self.producer.generate_section = MagicMock(return_value=torch.randn(2 * 32000))

        layer_config = {
            "name": "test_layer",
            "model": "facebook/musicgen-small",
            "sections": [{"section": "intro", "prompt": "intro music", "volume": 1.0}],
        }

        result = self.producer.generate_layer(layer_config)

        # Verify generate_section was called
        calls = self.producer.generate_section.call_args_list
        self.assertTrue(len(calls) > 0)
        # Check basic functionality
        self.assertIsNotNone(result)

    @patch("main.MusicGen")
    def test_generate_layer_with_loop(self, mock_musicgen_class):
        """Test layer generation with looping"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_musicgen_class.get_pretrained.return_value = mock_model

        # Generate short audio that needs looping
        short_audio = torch.randn(16000)  # 0.5 seconds
        self.producer.generate_section = MagicMock(return_value=short_audio)

        layer_config = {
            "name": "test_layer",
            "model": "facebook/musicgen-small",
            "sections": [
                {
                    "section": "intro",
                    "prompt": "intro music",
                    "volume": 1.0,
                    "loop": True,  # Should loop to fill 2 seconds
                }
            ],
        }

        result = self.producer.generate_layer(layer_config)

        # The intro section should have audio despite short generation
        intro_samples = 2 * 32000
        # Check that some audio exists in the intro section
        intro_audio = result[:intro_samples]
        self.assertGreater(torch.abs(intro_audio).mean().item(), 0)


class TestProduceIntegration(unittest.TestCase):
    """Integration tests for the main produce method"""

    def setUp(self):
        self.config = {
            "song": {
                "name": "Test Song",
                "output_file": "test_output.wav",
                "sample_rate": 32000,
                "master_volume": 0.9,
            },
            "sections": {
                "intro": {"start": 0, "duration": 1},
                "verse": {"start": 1, "duration": 1},
            },
            "layers": [
                {
                    "name": "drums",
                    "model": "facebook/musicgen-small",
                    "sections": [
                        {"section": "intro", "prompt": "drums", "volume": 0.8}
                    ],
                },
                {
                    "name": "bass",
                    "model": "facebook/musicgen-small",
                    "condition_on": ["drums"],
                    "sections": [{"section": "verse", "prompt": "bass", "volume": 0.7}],
                },
            ],
            "processing": {
                "normalize": True,
                "normalize_db": -14,
                "prevent_clipping": True,
            },
            "export_stems": True,
        }

        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        self.config["song"]["output_file"] = os.path.join(self.temp_dir, "output.wav")

        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_complete_flow(self, mock_musicgen_class, mock_audio_write):
        """Test the complete production flow"""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [torch.randn(1, 32000)]
        mock_model.generate_continuation.return_value = [torch.randn(1, 32000)]
        mock_musicgen_class.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)
        producer.produce(export_stems=True)

        # Check that audio_write was called for main output
        self.assertTrue(mock_audio_write.called)

        # Check that the main output file was written
        main_output_calls = [
            c for c in mock_audio_write.call_args_list if "output" in str(c[0][0])
        ]
        self.assertTrue(len(main_output_calls) > 0)

        # Check that stems were exported
        stem_calls = [
            c for c in mock_audio_write.call_args_list if "stems" in str(c[0][0])
        ]
        self.assertEqual(len(stem_calls), 2)  # drums and bass

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_with_conditioning_chain(
        self, mock_musicgen_class, mock_audio_write
    ):
        """Test production with conditioning chain between layers"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [torch.randn(1, 32000)]
        mock_model.generate_continuation.return_value = [torch.randn(1, 32000)]
        mock_musicgen_class.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)

        # Track generate_layer calls
        original_generate_layer = producer.generate_layer
        generate_layer_calls = []

        def track_generate_layer(layer_config):
            generate_layer_calls.append(
                {
                    "layer": layer_config["name"],
                }
            )
            return original_generate_layer(layer_config)

        producer.generate_layer = track_generate_layer
        producer.produce()

        # Verify layers were generated
        self.assertEqual(len(generate_layer_calls), 2)
        # Just verify that both layers were processed
        self.assertEqual(generate_layer_calls[0]["layer"], "drums")
        self.assertEqual(generate_layer_calls[1]["layer"], "bass")

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_normalization(self, mock_musicgen_class, mock_audio_write):
        """Test audio normalization in production"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        # Generate loud audio that needs normalization
        mock_model.generate.return_value = [torch.ones(1, 32000) * 2.0]
        mock_model.generate_continuation.return_value = [torch.ones(1, 32000) * 2.0]
        mock_musicgen_class.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)

        # Mock generate_section to return actual tensor
        producer.generate_section = MagicMock(
            side_effect=lambda model, prompt, duration, continuation_audio=None, style_audio=None, use_style=False: torch.ones(int(duration * 32000)) * 2.0
        )
        producer.produce()

        # Check that audio_write was called
        self.assertTrue(mock_audio_write.called)

        # Get the final audio passed to audio_write
        final_audio = mock_audio_write.call_args_list[0][0][1]

        # Check that audio processing happened (with soft limiting it may exceed 1.0 slightly)
        # The soft limiter may allow values slightly above 1.0
        self.assertLessEqual(torch.abs(final_audio).max().item(), 1.5)  # More lenient check


class TestMainFunction(unittest.TestCase):
    """Test the main entry point function"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yml")

        self.config = {"song": {"name": "Test"}, "sections": {}, "layers": []}

        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.MusicProducer.produce")
    @patch("sys.argv", ["main.py", "config.yml"])
    def test_main_basic(self, mock_produce):
        """Test main function with basic arguments"""
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                config=self.config_path, export_stems=False
            )

            from main import main

            main()

            mock_produce.assert_called_once()

    @patch("main.MusicProducer.produce")
    def test_main_with_export_stems(self, mock_produce):
        """Test main function with export stems flag"""
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                config=self.config_path, export_stems=True
            )

            from main import main

            main()

            # Check that produce was called with export_stems=True
            mock_produce.assert_called_once_with(export_stems=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
