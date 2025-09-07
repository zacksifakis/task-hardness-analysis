import numpy as np
from collections import defaultdict, Counter
import math
import pickle
from typing import List, Dict, Tuple, Union, Optional, TypeVar, Generic, Any

# Define a type variable for token types (integers or strings)
Token = TypeVar('Token', int, str)


class NgramModel(Generic[Token]):
    """
    A class for building and using n-gram language models.
    
    Attributes:
        n (int): The order of the n-gram model
        vocab (set): Set of all tokens seen in training
        counts (dict): Nested dictionary storing counts of n-grams
        smoothing (str): Smoothing method ('laplace', None)
    """
    
    def __init__(self, n: int, smoothing: Optional[str] = 'laplace'):
        """
        Initialize an n-gram model.
        
        Args:
            n (int): The order of the n-gram model (e.g., 1 for unigram, 2 for bigram)
            smoothing (str, optional): Smoothing method. Options: 'laplace', None
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        self.n = n
        self.vocab = set()
        self.counts = defaultdict(Counter)
        self.smoothing = smoothing
    
    def _get_ngrams(self, sequence: List[Token], n: int) -> List[Tuple[Token, ...]]:
        """
        Generate all n-grams of specified length from the sequence.
        
        Args:
            sequence (List[Token]): List of tokens (integers or strings)
            n (int): The length of n-grams to extract
            
        Returns:
            List[Tuple[Token, ...]]: List of n-gram tuples
        """
        return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]
    
    def train(self, sequences: List[List[Token]]):
        """
        Train the n-gram model on a corpus of sequences.
        
        Args:
            sequences (List[List[Token]]): List of token sequences (integers or strings)
        """
        # Gather all tokens for vocabulary
        for sequence in sequences:
            self.vocab.update(sequence)
        
        # Count n-grams of all orders up to n
        for sequence in sequences:
            for order in range(1, self.n + 1):
                ngrams = self._get_ngrams(sequence, order)
                for ngram in ngrams:
                    if order == 1:
                        # Unigram case
                        self.counts[()][ngram[0]] += 1
                    else:
                        # n-gram case where n > 1
                        context, token = ngram[:-1], ngram[-1]
                        self.counts[context][token] += 1
    
    def get_probability(self, context: Tuple[Token, ...], token: Token) -> float:
        """
        Get the probability of a token given its context.
        
        Args:
            context (Tuple[Token, ...]): The context n-gram (can be empty tuple for unigrams)
            token (Token): The token whose probability we want to calculate
            
        Returns:
            float: Probability P(token | context)
        """
        context_len = len(context)
        
        if context_len >= self.n:
            # Truncate context if it's longer than our n-gram model
            context = context[-(self.n-1):]
            context_len = len(context)
        
        # Get counts for the n-gram and its context
        token_count = self.counts[context][token]
        context_count = sum(self.counts[context].values())
        
        # Apply smoothing if needed
        if self.smoothing == 'laplace':
            # Laplace (add-one) smoothing
            vocab_size = len(self.vocab)
            return (token_count + 1) / (context_count + vocab_size)
        
        # import pdb; pdb.set_trace()
        if context_count == 0:
            # If context count is zero, check whether token is in self.counts[()].
            # If it is, return its count divided by the total count of unigrams.
            # Otherwise, return 0.0 (no probability mass)
            # if token in self.counts[()]:
            #     return self.counts[()][token] / sum(self.counts[()][t] for t in self.counts[()])
            # else:
            #     # No context and no token, return 0 probability
            return 0.0
        
        return token_count / context_count
    
    def log_probability(self, sequence: List[Token]) -> float:
        """
        Calculate the log probability of a sequence.
        
        Args:
            sequence (List[Token]): List of tokens (integers or strings)
            
        Returns:
            float: Log probability of the sequence under the model
        """
        if len(sequence) == 0:
            return 0.0
        
        log_prob = 0.0
                
        # For each position, calculate conditional probability based on available context
        for i in range(len(sequence)):
            # Determine context length based on model order and position
            context_len = min(i, self.n - 1)
            context = tuple(sequence[i - context_len:i]) if context_len > 0 else ()
            token = sequence[i]
            
            # Get probability of this token given context
            prob = self.get_probability(context, token)
            
            # Handle zero probability (shouldn't happen with proper smoothing)
            if prob <= 0:
                prob = 1e-10  # Small epsilon to avoid negative infinity
            
            log_prob += math.log2(prob)
        
        return log_prob
    
    def perplexity(self, sequence: List[Token]) -> float:
        """
        Calculate the perplexity of a sequence.
        
        Args:
            sequence (List[Token]): List of tokens (integers or strings)
            
        Returns:
            float: Perplexity of the sequence under the model
        """
        if len(sequence) == 0:
            return float('inf')
        
        log_prob = self.log_probability(sequence)
        return math.exp(-log_prob / len(sequence))
    
    def save(self, filepath: str):
        """
        Save the trained model to a file using pickle.
        
        Args:
            filepath (str): Path where the model will be saved
        """
        # Create a dictionary with all model attributes
        model_data = {
            'n': self.n,
            'vocab': self.vocab,
            'counts': self.counts,
            'smoothing': self.smoothing
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, filepath: str) -> 'NgramModel':
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            NgramModel: The loaded model instance
        """
        # Load the model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new model instance
        model = cls(
            n=model_data['n'],
            smoothing=model_data['smoothing']
        )
        
        # Restore model attributes
        model.vocab = model_data['vocab']
        model.counts = model_data['counts']
        
        return model





# Example usage
if __name__ == "__main__":
    # # Create a trigram model for integer tokens
    # model = NgramModel(n=3, smoothing='laplace')

    # # Train on multiple sequences
    # training_sequences = [
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 5],
    #     [1, 2, 3, 6],
    #     [1, 2, 3, 234232435],
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 6],
    #     [1, 2, 3, 4, 234232435],
    #     [1, 2, 3, 5, 6],
    #     [1, 2, 3, 5, 234232435],
    #     [1, 2, 3, 6, 234232435]
    # ]
    # model.train(training_sequences)

    # # Examine counts
    # print(f"Trigram counts:")
    # for context in model.counts:
    #     print(f"  Context {context}: {dict(model.counts[context])}")
    
    # # Save the model to a file
    # model.save("ngram_model.pkl")
    # print("Model saved to ngram_model.pkl")
    
    # # Load the model from the file
    # loaded_model = NgramModel.load("ngram_model.pkl")
    # print("Model loaded from ngram_model.pkl")
    
    # # Verify the loaded model has the same counts
    # print("\nVerifying loaded model:")
    # print(f"Loaded counts:")
    # for context in loaded_model.counts:
    #     print(f"  Context {context}: {dict(loaded_model.counts[context])}")
    
    # # Test the original model
    # test_sequence = [1, 2, 3, 4, 5, 6, 234232435]
    # print(f"\nTest sequence: {test_sequence}:")
    # print(f"  Log probability: {model.log_probability(test_sequence)}")
    # print(f"  Perplexity: {model.perplexity(test_sequence)}")
    
    # test_sequence = [2, 6, 3, 4, 5, 6, 12]
    # print(f"\nTest sequence: {test_sequence}:")
    # print(f"  Log probability: {model.log_probability(test_sequence)}")
    # print(f"  Perplexity: {model.perplexity(test_sequence)}")
    
    # Example with string tokens
    print("\n--- Example with string tokens ---")
    str_model = NgramModel(n=4, smoothing='laplace')
    
    # Train on string token sequences (e.g., words)
    # Use a different sequence for a better demonstration
    word_sequences = [
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["the", "quick", "brown", "fox"],
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    ]
    print("Here are the word sequences:")
    for seq in word_sequences:
        print(f"  {seq}")
    str_model.train(word_sequences)
    
    # Examine counts
    print(f"N-gram counts (string tokens):")
    for context in str_model.counts:
        print(f"  Context {context}: {dict(str_model.counts[context])}")
    
    # Test log probability and perplexity
    test_str_sequence = ["My", "name", "is", "the", "John", "Doe"]
    
    print(f"\nTest string sequence: {test_str_sequence}:")
    print(f"  Log probability: {str_model.log_probability(test_str_sequence)}")
    print(f"  Perplexity: {str_model.perplexity(test_str_sequence)}")
    
    # Debug by printing all probabilites of the test up to the length of the sequence
    print("\nProbabilities of each token in the test sequence:")
    for i in range(len(test_str_sequence)):
        context_len = min(i, str_model.n - 1)
        context = tuple(test_str_sequence[i - context_len:i]) if context_len > 0 else ()
        token = test_str_sequence[i]
        
        prob = str_model.get_probability(context, token)
        print(f"  P({token} | {context}) = {prob}")