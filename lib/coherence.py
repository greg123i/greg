import re

class ContradictionDetector:
    """
    Hand-crafted rule set for detecting semantic contradictions in simple stories.
    Focuses on:
    1. Location consistency (Entity cannot be in two places at once).
    2. Sentiment consistency (Cannot be happy and sad simultaneously).
    """
    def __init__(self):
        self.locations = ["kitchen", "garden", "bedroom", "living room", "park", "school", "office"]
        self.sentiments_pos = ["happy", "excited", "glad", "joyful"]
        self.sentiments_neg = ["sad", "angry", "upset", "depressed", "miserable"]
        
    def check_contradiction(self, generated_text, memory_text):
        """
        Returns a probability of contradiction [0, 1].
        """
        gen_lower = generated_text.lower()
        mem_lower = memory_text.lower()
        
        score = 0.0
        
        # 1. Location Check
        # Find current location in memory
        mem_loc = None
        for loc in self.locations:
            if f"in the {loc}" in mem_lower or f"at the {loc}" in mem_lower:
                mem_loc = loc
                break
        
        # Check generated location
        if mem_loc:
            for loc in self.locations:
                if loc == mem_loc: continue
                # If generating a NEW location assertion for the same entity context
                # (Simplification: assuming 'I' or main character context)
                if f"is in the {loc}" in gen_lower or f"is at the {loc}" in gen_lower:
                    score += 0.8 # Strong contradiction
        
        # 2. Sentiment Check
        # Check memory sentiment
        mem_sent = None
        if any(s in mem_lower for s in self.sentiments_pos):
            mem_sent = "pos"
        elif any(s in mem_lower for s in self.sentiments_neg):
            mem_sent = "neg"
            
        # Check generated sentiment
        if mem_sent == "pos":
            if any(s in gen_lower for s in self.sentiments_neg):
                score += 0.6
        elif mem_sent == "neg":
            if any(s in gen_lower for s in self.sentiments_pos):
                score += 0.6
                
        return min(score, 1.0)
