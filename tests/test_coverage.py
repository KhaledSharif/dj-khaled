"""
Additional unit tests to achieve 100% branch coverage for main.py
Run after test_main.py to cover remaining edge cases and branches
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import yaml
import pickle
import tempfile
import shutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MusicProducer, main


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {"use_cache": True, "cache_dir": self.temp_dir},
        }
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("pickle.load")
    def test_load_cache_pickle_error(self, mock_pickle_load):
        """Test cache loading with pickle error"""
        mock_pickle_load.side_effect = pickle.PickleError("Corrupt file")

        producer = MusicProducer(self.config_path)

        # Create a cache file
        cache_key = "test_key"
        cache_path = producer.cache_dir / f"{cache_key}.pkl"
        cache_path.touch()

        # Should handle the error gracefully
        with self.assertRaises(pickle.PickleError):
            producer.load_from_cache(cache_key)

    def test_save_cache_io_error(self):
        """Test cache saving with IO error"""
        producer = MusicProducer(self.config_path)

        # Patch only the specific open call for cache saving
        with patch("builtins.open", side_effect=IOError("Disk full")) as mock_open:
            # Should handle the error gracefully
            with self.assertRaises(IOError):
                producer.save_to_cache("test_key", torch.randn(100))

    def test_main_config_not_found(self):
        """Test main function with non-existent config file"""
        with patch("argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value = MagicMock(
                config="/nonexistent/path.yml", export_stems=False
            )

            with self.assertRaises(FileNotFoundError):
                main()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and alternative code paths"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.yml")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.MusicGen")
    def test_get_model_without_slash_in_name(self, mock_musicgen):
        """Test model loading with old-style name (no slash)"""
        config = {"song": {"name": "Test"}, "sections": {}, "layers": []}
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        mock_model = MagicMock()
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)
        model = producer.get_model("musicgen-small")  # No slash

        mock_musicgen.get_pretrained.assert_called_with("musicgen-small")
        self.assertEqual(model, mock_model)

    @patch("main.MusicGen")
    def test_get_model_no_generation_params(self, mock_musicgen):
        """Test model loading without generation parameters"""
        config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {},  # No generation_params
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        mock_model = MagicMock()
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)
        producer.get_model("test-model")

        # Should not call set_generation_params
        mock_model.set_generation_params.assert_not_called()

    def test_mix_audio_empty_list(self):
        """Test mixing with empty audio list"""
        config = {"song": {"name": "Test"}, "sections": {}, "layers": []}
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(self.config_path)
        result = producer.mix_audio([], 1000)

        # Should return silence
        expected = torch.zeros(1000)
        torch.testing.assert_close(result, expected)

    def test_mix_audio_no_clipping_needed(self):
        """Test mixing when no clipping prevention is needed"""
        config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "processing": {"prevent_clipping": True},
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(self.config_path)

        # Small values that don't need clipping
        audio1 = torch.ones(100) * 0.3
        audio2 = torch.ones(100) * 0.2

        result = producer.mix_audio([(audio1, 1.0), (audio2, 1.0)], 100)

        # Should not be normalized
        expected = torch.ones(100) * 0.5
        torch.testing.assert_close(result, expected)

    def test_apply_fade_multidimensional(self):
        """Test fade with multi-dimensional audio"""
        config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {},
            "layers": [],
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(self.config_path)

        # 2D audio tensor
        audio = torch.ones(2, 32000)
        faded = producer.apply_fade(audio, fade_out=True, fade_duration=0.5)

        # Check fade applied to last dimension
        self.assertLess(faded[0, -1].item(), 0.1)
        self.assertLess(faded[1, -1].item(), 0.1)
        self.assertAlmostEqual(faded[0, 0].item(), 1.0)

    def test_apply_fade_no_fade(self):
        """Test apply_fade with fade_out=False (no-op)"""
        config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {},
            "layers": [],
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(self.config_path)

        audio = torch.ones(32000)
        result = producer.apply_fade(audio, fade_out=False)

        # Should be unchanged
        torch.testing.assert_close(result, audio)


class TestGenerateSection(unittest.TestCase):
    """Test additional generate_section branches"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {},
            "layers": [],
        }
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_section_short_conditioning(self):
        """Test generation when conditioning audio is shorter than prompt duration"""
        mock_model = MagicMock()
        mock_output = torch.randn(1, 32000)
        mock_model.generate.return_value = [mock_output]

        # Very short conditioning audio (less than 1 second minimum)
        condition_audio = torch.randn(1, 1000)  # Very short (< 1 sec at 32kHz)

        result = self.producer.generate_section(
            mock_model, "test", 4.0, condition_audio
        )

        # Should fall back to text generation since audio is too short
        mock_model.generate.assert_called_once()
        # Should not have tried continuation
        mock_model.generate_continuation.assert_not_called()


class TestLayerGeneration(unittest.TestCase):
    """Test additional layer generation scenarios"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {
                "intro": {"start": 0, "duration": 2},
                "verse": {"start": 2, "duration": 3},
                "outro": {"start": 5, "duration": 2},
            },
            "layers": [],
        }
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.MusicGen")
    @patch("torch.cuda.is_available")
    def test_generate_layer_cuda_available(self, mock_cuda, mock_musicgen_class):
        """Test layer generation with CUDA available"""
        mock_cuda.return_value = True

        mock_model = MagicMock()
        # Model without 'device' attribute
        if hasattr(mock_model, "device"):
            delattr(mock_model, "device")
        mock_musicgen_class.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)
        producer.generate_section = MagicMock(return_value=torch.randn(64000))

        layer_config = {
            "name": "test",
            "sections": [{"section": "intro", "prompt": "test", "volume": 1.0}],
        }

        result = producer.generate_layer(layer_config)

        # Should be 7 seconds total
        self.assertEqual(result.shape[0], 7 * 32000)

    @patch("main.MusicGen")
    def test_generate_layer_empty_sections(self, mock_musicgen_class):
        """Test layer generation with empty sections list"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_musicgen_class.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)

        layer_config = {
            "name": "empty_layer",
            "model": "test-model",
            "sections": [],  # Empty sections
        }

        result = producer.generate_layer(layer_config)

        # Should return zeros for full duration
        expected_samples = 7 * 32000
        self.assertEqual(result.shape[0], expected_samples)
        self.assertEqual(torch.abs(result).sum().item(), 0.0)

    @patch("main.MusicGen")
    def test_generate_layer_multiple_sections(self, mock_musicgen_class):
        """Test layer with multiple sections"""
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_musicgen_class.get_pretrained.return_value = mock_model

        producer = MusicProducer(self.config_path)

        # Mock to return different audio for each section
        call_count = [0]

        def mock_generate(model, prompt, duration, **kwargs):
            call_count[0] += 1
            return torch.ones(int(duration * 32000)) * call_count[0]

        producer.generate_section = MagicMock(side_effect=mock_generate)

        layer_config = {
            "name": "multi_section",
            "sections": [
                {"section": "intro", "prompt": "intro", "volume": 1.0},
                {"section": "verse", "prompt": "verse", "volume": 0.5},
                {"section": "outro", "prompt": "outro", "volume": 0.3},
            ],
        }

        result = producer.generate_layer(layer_config)

        # Verify all sections were generated
        self.assertEqual(producer.generate_section.call_count, 3)
        # Check that audio exists in all sections
        self.assertGreater(torch.abs(result).sum().item(), 0)


class TestProduceMethod(unittest.TestCase):
    """Test additional produce method scenarios"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_normalize_false(self, mock_musicgen, mock_audio_write):
        """Test production with normalize=False"""
        config = {
            "song": {"name": "Test", "output_file": "output.wav"},
            "sections": {"intro": {"start": 0, "duration": 1}},
            "layers": [
                {
                    "name": "test",
                    "sections": [{"section": "intro", "prompt": "test", "volume": 1.0}],
                }
            ],
            "processing": {
                "normalize": False,
                "prevent_clipping": False,
            },  # Disable both normalization and clipping
        }

        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [torch.ones(1, 32000) * 2.0]
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(config_path)
        producer.generate_section = MagicMock(return_value=torch.ones(32000) * 2.0)
        producer.produce()

        # Get final audio
        final_audio = mock_audio_write.call_args[0][1]

        # Should NOT be normalized (still > 1.0)
        self.assertGreater(torch.abs(final_audio).max().item(), 1.0)

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_zero_max_val(self, mock_musicgen, mock_audio_write):
        """Test production when audio is all zeros"""
        config = {
            "song": {"name": "Test", "output_file": "output.wav"},
            "sections": {"intro": {"start": 0, "duration": 1}},
            "layers": [
                {
                    "name": "silence",
                    "sections": [{"section": "intro", "prompt": "test", "volume": 0.0}],
                }
            ],
            "processing": {"normalize": True, "normalize_db": -14},
        }

        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(config_path)
        # Return zeros
        producer.generate_section = MagicMock(return_value=torch.zeros(32000))
        producer.produce()

        # Should not crash and should write zeros
        final_audio = mock_audio_write.call_args[0][1]
        self.assertEqual(torch.abs(final_audio).sum().item(), 0.0)

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_empty_layers(self, mock_musicgen, mock_audio_write):
        """Test production with no layers"""
        config = {
            "song": {"name": "Test", "output_file": "output.wav"},
            "sections": {"intro": {"start": 0, "duration": 1}},
            "layers": [],  # No layers
        }

        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        producer.produce()

        # Should produce silence
        final_audio = mock_audio_write.call_args[0][1]
        self.assertEqual(torch.abs(final_audio).sum().item(), 0.0)

    @patch("main.audio_write")
    @patch("main.MusicGen")
    def test_produce_missing_conditioning_layer(self, mock_musicgen, mock_audio_write):
        """Test when conditioning references non-existent layer"""
        config = {
            "song": {"name": "Test", "output_file": "output.wav"},
            "sections": {"intro": {"start": 0, "duration": 1}},
            "layers": [
                {
                    "name": "layer1",
                    "condition_on": ["nonexistent_layer"],  # Doesn't exist
                    "sections": [{"section": "intro", "prompt": "test", "volume": 1.0}],
                }
            ],
        }

        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [torch.randn(1, 32000)]
        mock_musicgen.get_pretrained.return_value = mock_model

        producer = MusicProducer(config_path)
        producer.generate_section = MagicMock(return_value=torch.randn(32000))
        producer.produce()

        # Should continue without conditioning
        mock_audio_write.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
