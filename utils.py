"""
Yardımcı fonksiyonlar: Text preprocessing, tokenization, cleaning
"""
import re
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


# Download NLTK data (required on first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Text verilerini işleme ve sequence'lere çevirme"""

    def __init__(self,
                 remove_stopwords: bool = False,
                 lowercase: bool = True,
                 min_word_length: int = 2,
                 language: str = 'turkish'):
        """
        Args:
            remove_stopwords: Stopword'leri kaldır
            lowercase: Küçük harfe çevir
            min_word_length: Minimum kelime uzunluğu
            language: Dil (turkish/english)
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.min_word_length = min_word_length

        if language == 'turkish':
            # Turkish stopwords (simple list, can be extended if needed)
            self.stopwords = set(['ve', 'ile', 'bir', 'bu', 'şu', 'o', 'de', 'da', 'ki', 'mi', 'mu', 'mü'])
        else:
            self.stopwords = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Text temizleme"""
        if pd.isna(text):
            return ""

        text = str(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Clean special characters (preserve punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)

        # Clean extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_sentence(self, text: str) -> List[str]:
        """Cümleyi kelimelere ayır"""
        text = self.clean_text(text)

        # Sentence tokenization
        sentences = sent_tokenize(text)

        all_tokens = []
        for sentence in sentences:
            # Word tokenization
            tokens = word_tokenize(sentence)

            # Filtering
            filtered_tokens = []
            for token in tokens:
                # Skip punctuation marks
                if token in ['.', ',', '!', '?', ';', ':']:
                    continue

                # Minimum length check
                if len(token) < self.min_word_length:
                    continue

                # Stopword check
                if self.remove_stopwords and token in self.stopwords:
                    continue

                filtered_tokens.append(token)

            if filtered_tokens:
                all_tokens.extend(filtered_tokens)

        return all_tokens

    def extract_sequences(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Text'ten n-gram sequence'leri çıkar
        
        Args:
            text: İşlenecek text
            n: n-gram boyutu (2 = bigram, 3 = trigram, vb.)
        
        Returns:
            List of tuples: [(word1, word2), (word2, word3), ...]
        """
        tokens = self.tokenize_sentence(text)

        if len(tokens) < n:
            return []

        sequences = []
        for i in range(len(tokens) - n + 1):
            sequence = tuple(tokens[i:i+n])
            sequences.append(sequence)

        return sequences

    def extract_word_sequences(self, text: str) -> List[str]:
        """Text'ten kelime dizisini çıkar (path için)"""
        return self.tokenize_sentence(text)

    def process_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str,
                         category_column: str) -> pd.DataFrame:
        """
        DataFrame'i işle ve sequence'leri ekle
        
        Returns:
            DataFrame with 'tokens' and 'sequences' columns
        """
        processed_df = df.copy()

        # Tokenization
        processed_df['tokens'] = processed_df[text_column].apply(
            self.extract_word_sequences
        )

        # Bigram sequences (2-word paths)
        processed_df['bigrams'] = processed_df[text_column].apply(
            lambda x: self.extract_sequences(x, n=2)
        )

        # Trigram sequences (3-word paths)
        processed_df['trigrams'] = processed_df[text_column].apply(
            lambda x: self.extract_sequences(x, n=3)
        )

        return processed_df

