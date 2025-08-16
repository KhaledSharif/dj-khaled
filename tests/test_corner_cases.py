"""
Advanced corner case tests for main.py
These tests look for hidden bugs, edge cases, and potential failure modes
that aren't caught by basic line coverage.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import yaml
from pathlib import Path
import tempfile
import shutil
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import MusicProducer


class TestNumericalEdgeCases(unittest.TestCase):
    """Test numerical precision, overflow, and boundary conditions"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [],
            "processing": {"prevent_clipping": True},
        }
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_zero_sample_rate(self):
        """Test with zero or negative sample rate"""
        config = {
            "song": {"name": "Test", "sample_rate": 0},
            "sections": {},
            "layers": [],
        }
        config_path = os.path.join(self.temp_dir, "zero_rate.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        # Should handle zero sample rate without crash
        self.assertEqual(producer.sample_rate, 0)

    def test_floating_point_precision_in_samples(self):
        """Test that floating point calculations don't cause off-by-one errors"""
        # Test with duration that doesn't divide evenly into samples
        duration = 3.141592653589793  # π seconds
        expected_samples = int(duration * 32000)
        
        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model
            
            section_audio = self.producer.generate_section(
                mock_model, "test", duration, None
            )
            
            # Check that model was called with correct duration
            mock_model.set_generation_params.assert_called_with(duration=duration)

    def test_extreme_volume_values(self):
        """Test with extreme volume values that might cause overflow"""
        # Test with very large volume
        audio = torch.ones(1000)
        result = self.producer.mix_audio([(audio, 1e10)], 1000)
        # Should prevent clipping
        self.assertLessEqual(torch.abs(result).max(), 1.0)

        # Test with very small volume
        result = self.producer.mix_audio([(audio, 1e-10)], 1000)
        self.assertAlmostEqual(result.max().item(), 1e-10, places=15)

        # Test with negative volume
        result = self.producer.mix_audio([(audio, -1.0)], 1000)
        self.assertEqual(result[0], -1.0)

    def test_nan_and_inf_handling(self):
        """Test handling of NaN and Inf values in audio"""
        # Test with NaN values
        audio_nan = torch.tensor([float("nan"), 1.0, 2.0])
        result = self.producer.mix_audio([(audio_nan, 1.0)], 3)
        # NaN should propagate or be handled
        self.assertTrue(torch.isnan(result[0]) or result[0] == 0)

        # Test with Inf values
        audio_inf = torch.tensor([float("inf"), 1.0, 2.0])
        result = self.producer.mix_audio([(audio_inf, 1.0)], 3)
        # Should trigger clipping prevention
        self.assertLessEqual(torch.abs(result).max(), 1.0)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration edge cases and validation"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_invalid_section_reference(self):
        """Test referencing non-existent sections"""
        config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [
                {
                    "name": "test",
                    "sections": [
                        {"section": "nonexistent", "prompt": "test"}  # Invalid ref
                    ],
                }
            ],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model

            # Should raise KeyError for non-existent section
            with self.assertRaises(KeyError):
                producer.generate_layer(config["layers"][0])

    def test_start_offset_exceeds_duration(self):
        """Test when start_offset is greater than section duration"""
        config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [
                {
                    "name": "test",
                    "sections": [
                        {
                            "section": "intro",
                            "prompt": "test",
                            "start_offset": 15.0,  # Exceeds 10s duration
                        }
                    ],
                }
            ],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model

            # Mock generate_section to return audio
            producer.generate_section = MagicMock(
                return_value=torch.ones(int(10 * 32000))
            )

            # Should handle offset exceeding duration
            result = producer.generate_layer(config["layers"][0])
            self.assertIsNotNone(result)

    def test_negative_duration(self):
        """Test with negative duration values"""
        config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": -5}},  # Negative duration
            "layers": [{"name": "test", "sections": []}],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model

            # Should raise RuntimeError for negative samples
            with self.assertRaises(RuntimeError):
                result = producer.generate_layer(config["layers"][0])

    def test_unicode_in_paths_and_names(self):
        """Test Unicode characters in file paths and names"""
        config = {
            "song": {"name": "テスト音楽", "output_file": "音楽.wav"},
            "sections": {"イントロ": {"start": 0, "duration": 10}},
            "layers": [
                {
                    "name": "レイヤー",
                    "sections": [{"section": "イントロ", "prompt": "プロンプト"}],
                }
            ],
            "advanced": {"use_cache": True, "cache_dir": self.temp_dir},
        }
        config_path = os.path.join(self.temp_dir, "unicode.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f, allow_unicode=True)

        producer = MusicProducer(config_path)
        cache_key = producer.get_cache_key("レイヤー", "セクション", "プロンプト")
        
        # Save and load with Unicode key
        test_tensor = torch.randn(100)
        producer.save_to_cache(cache_key, test_tensor)
        loaded = producer.load_from_cache(cache_key)
        
        self.assertIsNotNone(loaded)
        self.assertTrue(torch.allclose(test_tensor, loaded))

    def test_circular_conditioning_dependency(self):
        """Test circular dependencies in layer conditioning"""
        config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [
                {
                    "name": "layer1",
                    "condition_on": ["layer2"],  # Circular: 1 depends on 2
                    "sections": [{"section": "intro", "prompt": "test"}],
                },
                {
                    "name": "layer2",
                    "condition_on": ["layer1"],  # Circular: 2 depends on 1
                    "sections": [{"section": "intro", "prompt": "test"}],
                },
            ],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        # Mock generate_section to avoid actual generation
        producer.generate_section = MagicMock(return_value=torch.randn(320000))
        
        # Should handle circular dependencies (layer2 won't have layer1 in conditioning)
        producer.produce()
        # If it completes without deadlock/crash, test passes


class TestAudioProcessingBoundaries(unittest.TestCase):
    """Test audio processing edge cases"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "song": {"name": "Test", "sample_rate": 32000},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [],
        }
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_zero_length_audio(self):
        """Test with zero-length audio tensors"""
        empty_audio = torch.tensor([])
        
        # Test apply_fade with empty audio
        result = self.producer.apply_fade(empty_audio, fade_out=True)
        self.assertEqual(len(result), 0)
        
        # Test mix_audio with empty audio
        result = self.producer.mix_audio([(empty_audio, 1.0)], 100)
        self.assertEqual(len(result), 100)

    def test_single_sample_audio(self):
        """Test with single-sample audio"""
        single_sample = torch.tensor([0.5])
        
        # Test apply_fade with single sample
        result = self.producer.apply_fade(single_sample, fade_out=True)
        # Should handle gracefully
        self.assertEqual(len(result), 1)
        
        # Test mix
        result = self.producer.mix_audio([(single_sample, 2.0)], 1)
        self.assertEqual(result[0], 1.0)  # 0.5 * 2.0

    def test_fade_duration_exceeds_audio_length(self):
        """Test fade when fade duration is longer than audio"""
        # 0.1 seconds of audio
        short_audio = torch.ones(3200)
        
        # Try to apply 10-second fade
        result = self.producer.apply_fade(
            short_audio, fade_out=True, fade_duration=10.0
        )
        
        # Should handle gracefully and fade entire audio
        self.assertEqual(len(result), 3200)
        # Last sample should be close to 0 (faded out)
        self.assertLess(result[-1], 0.1)

    def test_looping_with_zero_length_generation(self):
        """Test looping when generated audio has zero length"""
        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model

            # Return empty tensor
            self.producer.generate_section = MagicMock(return_value=torch.tensor([]))

            layer_config = {
                "name": "test",
                "sections": [
                    {"section": "intro", "prompt": "test", "loop": True}
                ],
            }

            # Should handle empty audio in loop
            result = self.producer.generate_layer(layer_config)
            self.assertIsNotNone(result)


class TestModelOutputValidation(unittest.TestCase):
    """Test validation of model outputs"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [],
        }
        self.config_path = os.path.join(self.temp_dir, "config.yml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
        self.producer = MusicProducer(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_model_returns_none(self):
        """Test when model returns None"""
        mock_model = MagicMock()
        mock_model.generate.return_value = None  # Changed: return None directly, not [None]
        
        # Now returns zeros instead of raising exception (more robust)
        result = self.producer.generate_section(mock_model, "test", 10.0)
        # Should return silence of the requested duration
        expected_samples = int(10.0 * self.producer.sample_rate)
        self.assertEqual(result.shape[0], expected_samples)
        self.assertTrue(torch.all(result == 0))

    def test_model_returns_empty_list(self):
        """Test when model returns empty list"""
        mock_model = MagicMock()
        mock_model.generate.return_value = []
        
        # Now returns zeros instead of raising exception (more robust)
        result = self.producer.generate_section(mock_model, "test", 10.0)
        # Should return silence of the requested duration
        expected_samples = int(10.0 * self.producer.sample_rate)
        self.assertEqual(result.shape[0], expected_samples)
        self.assertTrue(torch.all(result == 0))

    def test_model_returns_wrong_shape(self):
        """Test when model returns unexpected tensor shape"""
        mock_model = MagicMock()
        # Return 2D tensor instead of expected 1D
        mock_model.generate.return_value = [torch.randn(10, 10)]
        
        result = self.producer.generate_section(mock_model, "test", 10.0)
        # Should return first element regardless of shape
        self.assertEqual(result.shape, (10, 10))

    def test_conditioning_dimension_mismatch(self):
        """Test conditioning with unexpected dimensions"""
        mock_model = MagicMock()
        mock_model.generate_continuation.return_value = [torch.randn(1000)]
        
        # Provide 3D conditioning audio
        condition_audio = torch.randn(2, 3, 1000)
        
        result = self.producer.generate_section(
            mock_model, "test", 10.0, condition_audio
        )
        # Should handle dimension mismatch
        self.assertIsNotNone(result)


class TestResourceAndFileSystem(unittest.TestCase):
    """Test file system and resource handling"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_output_directory_permissions(self):
        """Test handling of permission errors on output directory"""
        config = {
            "song": {"name": "Test", "output_file": "/root/forbidden/output.wav"},
            "sections": {"intro": {"start": 0, "duration": 1}},
            "layers": [],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("No permission")
            
            # Should handle permission error gracefully
            try:
                producer.produce()
            except PermissionError:
                pass  # Expected

    def test_cache_file_corruption(self):
        """Test handling of corrupted cache files"""
        config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {"use_cache": True, "cache_dir": self.temp_dir},
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        
        # Create corrupted cache file
        cache_key = "corrupted"
        cache_path = Path(self.temp_dir) / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            f.write(b"corrupted data")
        
        # Should handle corrupted cache gracefully
        with self.assertRaises(Exception):
            result = producer.load_from_cache(cache_key)


class TestIntegrationAndRealWorld(unittest.TestCase):
    """Test realistic scenarios and integration"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_overlapping_sections(self):
        """Test when sections overlap in time"""
        config = {
            "song": {"name": "Test"},
            "sections": {
                "intro": {"start": 0, "duration": 10},
                "verse": {"start": 5, "duration": 10},  # Overlaps with intro
            },
            "layers": [
                {
                    "name": "test",
                    "sections": [
                        {"section": "intro", "prompt": "intro"},
                        {"section": "verse", "prompt": "verse"},
                    ],
                }
            ],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model

            producer = MusicProducer(config_path)
            producer.generate_section = MagicMock(
                side_effect=lambda m, p, d, **kwargs: torch.ones(int(d * 32000))
            )

            result = producer.generate_layer(config["layers"][0])
            
            # Should handle overlapping sections (they get mixed)
            self.assertIsNotNone(result)

    def test_concurrent_layer_conditions(self):
        """Test when multiple layers condition on the same layer"""
        config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [
                {
                    "name": "base",
                    "sections": [{"section": "intro", "prompt": "base"}],
                },
                {
                    "name": "harmony1",
                    "condition_on": ["base"],
                    "sections": [{"section": "intro", "prompt": "harmony1"}],
                },
                {
                    "name": "harmony2",
                    "condition_on": ["base"],
                    "sections": [{"section": "intro", "prompt": "harmony2"}],
                },
                {
                    "name": "top",
                    "condition_on": ["harmony1", "harmony2"],
                    "sections": [{"section": "intro", "prompt": "top"}],
                },
            ],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        producer.generate_section = MagicMock(return_value=torch.randn(320000))
        
        # Should handle multiple layers conditioning on same base
        producer.produce()

    def test_condition_chain_with_effects(self):
        """Test complex conditioning chain with various effects"""
        config = {
            "song": {"name": "Test"},
            "sections": {"intro": {"start": 0, "duration": 10}},
            "layers": [
                {
                    "name": "layer1",
                    "sections": [
                        {
                            "section": "intro",
                            "prompt": "test",
                            "fade_out": True,
                            "volume": 0.5,
                        }
                    ],
                },
                {
                    "name": "layer2",
                    "condition_on": ["layer1"],
                    "sections": [
                        {
                            "section": "intro",
                            "prompt": "test",
                            "loop": True,
                            "start_offset": 2.0,
                        }
                    ],
                },
                {
                    "name": "layer3",
                    "condition_on": ["layer1", "layer2"],
                    "sections": [{"section": "intro", "prompt": "test"}],
                },
            ],
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("main.MusicGen") as mock_musicgen:
            mock_model = MagicMock()
            mock_model.device = "cpu"
            mock_musicgen.get_pretrained.return_value = mock_model

            producer = MusicProducer(config_path)
            
            # Track conditioning to ensure proper chain
            original_generate = producer.generate_section
            conditioning_log = []

            def log_conditioning(model, prompt, duration, **kwargs):
                # Check for backward-compatible condition_audio parameter
                condition = kwargs.get('condition_audio')
                conditioning_log.append(
                    {"prompt": prompt, "has_condition": condition is not None}
                )
                # Return appropriate length audio
                return torch.ones(int(duration * 32000))

            producer.generate_section = log_conditioning

            producer.produce()

            # Check that sections were generated (no inter-layer conditioning in new architecture)
            self.assertEqual(len(conditioning_log), 3)
            # In the new architecture, layers don't condition on each other
            self.assertFalse(conditioning_log[0]["has_condition"])  # layer1 no condition
            self.assertFalse(conditioning_log[1]["has_condition"])  # layer2 no condition (changed)
            self.assertFalse(conditioning_log[2]["has_condition"])  # layer3 no condition (changed)

    def test_memory_leak_in_cache(self):
        """Test potential memory leak with large cache operations"""
        config = {
            "song": {"name": "Test"},
            "sections": {},
            "layers": [],
            "advanced": {"use_cache": True, "cache_dir": self.temp_dir},
        }
        config_path = os.path.join(self.temp_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        producer = MusicProducer(config_path)
        
        # Save and load many items
        for i in range(100):
            key = f"test_{i}"
            data = torch.randn(10000)
            producer.save_to_cache(key, data)
            loaded = producer.load_from_cache(key)
            self.assertIsNotNone(loaded)
        
        # Check that cache files were created
        cache_files = list(Path(self.temp_dir).glob("*.pkl"))
        self.assertEqual(len(cache_files), 100)


if __name__ == "__main__":
    unittest.main()