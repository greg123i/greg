import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.coherence import ContradictionDetector
from lib.data_loader import load_fim_data
from lib.brain.thalamus import Thalamus
from lib.brain.config import BrainConfig

class TestEnhancements(unittest.TestCase):
    
    def test_coherence_contradiction(self):
        detector = ContradictionDetector()
        
        # 1. Location Contradiction
        memory = "Greg is in the kitchen making coffee."
        gen = "Suddenly, Greg is in the garden."
        score = detector.check_contradiction(gen, memory)
        self.assertGreaterEqual(score, 0.5, "Should detect location contradiction")
        
        # 2. No Contradiction
        gen_ok = "He poured the milk."
        score_ok = detector.check_contradiction(gen_ok, memory)
        self.assertLess(score_ok, 0.5, "Should not flag consistent text")
        
        # 3. Sentiment Contradiction
        memory_sad = "I am so sad and miserable today."
        gen_happy = "I am feeling so happy and excited!"
        score_sent = detector.check_contradiction(gen_happy, memory_sad)
        self.assertGreaterEqual(score_sent, 0.5, "Should detect sentiment contradiction")

    def test_fim_data_format(self):
        # Generate small sample
        data = load_fim_data(samples=5, max_len=100)
        self.assertEqual(len(data), 5)
        
        # Check for presence of tag markers (roughly)
        # We can't easily check for string "<|prefix|>" in char codes without decoding
        # But we verify it runs without error and returns data
        self.assertTrue(len(data[0]) > 0)
        
    def test_mood_gru_gradients(self):
        thalamus = Thalamus()
        
        # Dummy inputs
        batch_size = 2
        vec_size = BrainConfig.VECTOR_SIZE
        
        r = torch.randn(batch_size, vec_size)
        t = torch.randn(batch_size, vec_size)
        w = torch.randn(batch_size, vec_size)
        token = torch.tensor([65, 66]) # 'A', 'B'
        
        # Forward
        _, _, _, _, _, mood, _ = thalamus(r, t, w, token)
        
        # Check output shape
        self.assertEqual(mood.shape, (batch_size, 256))
        
        # Check Gradients
        loss = mood.mean()
        loss.backward()
        
        self.assertIsNotNone(thalamus.mood_gru.weight_ih_l0.grad, "Mood GRU should have gradients")

if __name__ == '__main__':
    unittest.main()
